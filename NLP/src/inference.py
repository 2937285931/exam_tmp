#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
IMDB情感二分类实验 — 独立推理/预测脚本
加载训练好的 LSTM 模型，支持输入单条或多条文本，输出情感预测结果。
=============================================================================
"""

import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import CONFIG, set_seed
from src.data_utils import tokenize, texts_to_sequences, build_embedding_matrix
from src.model import TextLSTM


class SentimentPredictor:
    """
    情感预测器：加载训练好的 LSTM 模型、词表、Word2Vec 词向量，
    对外提供 predict() 和 predict_batch() 接口。
    """

    def __init__(self, results_dir: str = None):
        """
        初始化预测器，加载所有必要工件。
        参数:
          results_dir: 训练产出物目录，默认为 CONFIG["saved_dir"]（./results/）
        """
        self.cfg = CONFIG
        if results_dir is None:
            results_dir = self.cfg["saved_dir"]

        self.results_dir = results_dir
        self.device = torch.device(self.cfg["device"])

        # 1. 加载词汇表
        vocab_path = os.path.join(results_dir, "vocab.pkl")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"词汇表文件不存在: {vocab_path}\n请先运行 src/train.py 训练模型"
            )
        with open(vocab_path, "rb") as f:
            vocab_data = pickle.load(f)
        self.word2idx = vocab_data["word2idx"]
        self.idx2word = vocab_data.get("idx2word", {})
        print(f"已加载词汇表: {len(self.word2idx)} 个词")

        # 2. 加载 Word2Vec 模型并构建嵌入矩阵
        from gensim.models import Word2Vec as GensimWord2Vec
        w2v_path = os.path.join(results_dir, "w2v_model.gensim")
        if not os.path.exists(w2v_path):
            raise FileNotFoundError(
                f"Word2Vec模型文件不存在: {w2v_path}\n请先运行 src/train.py 训练模型"
            )
        w2v_model = GensimWord2Vec.load(w2v_path)
        self.embedding_matrix = build_embedding_matrix(
            self.word2idx, w2v_model, self.cfg["embed_dim"]
        )

        # 3. 构建 LSTM 模型并加载权重
        weights_path = os.path.join(results_dir, "lstm_model.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"LSTM模型权重文件不存在: {weights_path}\n请先运行 src/train.py 训练模型"
            )

        self.model = TextLSTM(
            len(self.word2idx), self.cfg["embed_dim"], self.cfg,
            embedding_matrix=self.embedding_matrix
        ).to(self.device)

        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"已加载 LSTM 最佳模型权重: {weights_path}")

    def _texts_to_tensor(self, texts: list) -> torch.Tensor:
        """将文本列表转换为模型输入张量。"""
        sequences = texts_to_sequences(texts, self.word2idx, self.cfg["max_seq_len"])
        return torch.from_numpy(sequences).long().to(self.device)

    def predict(self, text: str) -> dict:
        """
        对单条文本进行情感预测。
        参数:
          text: 英文影评文本
        返回:
          dict: {"sentiment": "Positive"/"Negative", "confidence": float,
                 "positive_prob": float, "negative_prob": float}
        """
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: list) -> list:
        """
        对多条文本批量进行情感预测。
        参数:
          texts: 英文影评文本列表
        返回:
          list[dict]: 每条文本的预测结果
        """
        input_tensor = self._texts_to_tensor(texts)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits)

        probs_np = probs.cpu().numpy()

        results = []
        for i, prob in enumerate(probs_np):
            positive_prob = float(prob)
            negative_prob = 1.0 - positive_prob
            sentiment = "Positive" if positive_prob >= 0.5 else "Negative"
            confidence = positive_prob if sentiment == "Positive" else negative_prob
            results.append({
                "sentiment": sentiment,
                "confidence": confidence,
                "positive_prob": positive_prob,
                "negative_prob": negative_prob,
            })

        return results


# ===========================================================================
# 命令行入口
# ===========================================================================

def main():
    set_seed(CONFIG["seed"])

    print("=" * 60)
    print("IMDB 情感二分类 — 推理预测")
    print("=" * 60)

    # 初始化预测器
    predictor = SentimentPredictor()

    # 交互模式
    print("\n进入交互推理模式（输入 'quit' 或 'exit' 退出）")
    print("-" * 60)

    while True:
        try:
            text = input("\n请输入英文影评文本: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出推理模式。")
            break

        if text.lower() in ("quit", "exit", "q"):
            print("退出推理模式。")
            break

        if not text:
            print("输入为空，请重新输入。")
            continue

        result = predictor.predict(text)
        print(f"\n  情感类别: {result['sentiment']}")
        print(f"  置信度:   {result['confidence']:.4f}")
        print(f"  Positive 概率: {result['positive_prob']:.4f}")
        print(f"  Negative 概率: {result['negative_prob']:.4f}")


if __name__ == "__main__":
    main()
