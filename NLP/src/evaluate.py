#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
IMDB情感二分类实验 — 独立评估脚本
加载 results/ 目录下训练好的最佳模型权重，评估测试集指标。
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

from src.config import CONFIG, DATA_ROOT, set_seed, ensure_dir
from src.data_utils import (
    load_imdb_from_local,
    texts_to_sequences,
    build_embedding_matrix,
    IMDBDataset,
)
from src.model import TextCNN, TextLSTM
from src.train_utils import evaluate
from torch.utils.data import DataLoader


def load_artifacts(results_dir: str):
    """
    从 results/ 目录加载评估所需的所有工件。
    返回:
      word2idx, embedding_matrix, test_loader, test_texts, test_labels
    """
    cfg = CONFIG
    device = torch.device(cfg["device"])

    # 1. 加载词汇表
    vocab_path = os.path.join(results_dir, "vocab.pkl")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"词汇表文件不存在: {vocab_path}，请先运行 train.py 训练模型")
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)
    word2idx = vocab_data["word2idx"]
    print(f"已加载词汇表: {len(word2idx)} 个词")

    # 2. 加载 Word2Vec 模型并构建嵌入矩阵
    from gensim.models import Word2Vec as GensimWord2Vec
    w2v_path = os.path.join(results_dir, "w2v_model.gensim")
    if not os.path.exists(w2v_path):
        raise FileNotFoundError(f"Word2Vec模型文件不存在: {w2v_path}，请先运行 train.py 训练模型")
    w2v_model = GensimWord2Vec.load(w2v_path)
    embedding_matrix = build_embedding_matrix(word2idx, w2v_model, cfg["embed_dim"])

    # 3. 加载测试数据
    _, _, test_texts, test_labels = load_imdb_from_local(DATA_ROOT)
    test_labels_np = np.array(test_labels, dtype=np.int64)
    print(f"已加载测试集: {len(test_texts)} 条")

    # 4. 序列化测试文本
    test_seqs = texts_to_sequences(test_texts, word2idx, cfg["max_seq_len"])
    print(f"测试序列: {test_seqs.shape}")

    # 5. 创建测试 DataLoader
    test_ds = IMDBDataset(test_seqs, test_labels_np)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    return word2idx, embedding_matrix, test_loader, test_texts, test_labels_np, device


def load_model(model_type: str, word2idx: dict, embedding_matrix, device, results_dir: str):
    """
    加载指定类型的模型并恢复最佳权重。
    参数:
      model_type: "cnn" 或 "lstm"
    返回:
      model
    """
    cfg = CONFIG
    if model_type == "cnn":
        model = TextCNN(
            len(word2idx), cfg["embed_dim"], cfg,
            embedding_matrix=embedding_matrix
        ).to(device)
        weights_path = os.path.join(results_dir, "cnn_model.pth")
    elif model_type == "lstm":
        model = TextLSTM(
            len(word2idx), cfg["embed_dim"], cfg,
            embedding_matrix=embedding_matrix
        ).to(device)
        weights_path = os.path.join(results_dir, "lstm_model.pth")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}，可选 'cnn' 或 'lstm'")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"模型权重文件不存在: {weights_path}，请先运行 train.py 训练模型"
        )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"已加载 {model_type.upper()} 最佳模型权重: {weights_path}")
    print(f"  {model_type.upper()} 参数量: {sum(p.numel() for p in model.parameters()):,}")

    return model


def main():
    set_seed(CONFIG["seed"])

    results_dir = CONFIG["saved_dir"]
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(
            f"results 目录不存在: {results_dir}，请先运行 train.py 训练模型"
        )

    print("=" * 60)
    print("IMDB 情感二分类 — 独立评估脚本")
    print("=" * 60)

    # 加载工件
    word2idx, embedding_matrix, test_loader, test_texts, test_labels_np, device = \
        load_artifacts(results_dir)

    # ---- 评估 CNN 模型 ----
    print(f"\n{'─'*60}")
    print("评估 Word2Vec + CNN 模型")
    print(f"{'─'*60}")
    cnn_model = load_model("cnn", word2idx, embedding_matrix, device, results_dir)
    cnn_test_loss, cnn_acc, cnn_f1, cnn_prec, cnn_rec, cnn_preds, cnn_labels = evaluate(
        cnn_model, test_loader, nn.BCEWithLogitsLoss(), device
    )
    print(f"  CNN Test Result → Acc: {cnn_acc:.4f}  F1: {cnn_f1:.4f}  "
          f"Prec: {cnn_prec:.4f}  Rec: {cnn_rec:.4f}")

    # ---- 评估 LSTM 模型 ----
    print(f"\n{'─'*60}")
    print("评估 Word2Vec + LSTM 模型")
    print(f"{'─'*60}")
    lstm_model = load_model("lstm", word2idx, embedding_matrix, device, results_dir)
    lstm_test_loss, lstm_acc, lstm_f1, lstm_prec, lstm_rec, lstm_preds, lstm_labels = evaluate(
        lstm_model, test_loader, nn.BCEWithLogitsLoss(), device
    )
    print(f"  LSTM Test Result → Acc: {lstm_acc:.4f}  F1: {lstm_f1:.4f}  "
          f"Prec: {lstm_prec:.4f}  Rec: {lstm_rec:.4f}")

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print("测试集评估结果汇总")
    print(f"{'='*60}")
    print(f"{'模型':<25} {'Accuracy':>10} {'F1-Score':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 65)
    print(f"{'Word2Vec + CNN':<25} {cnn_acc:>10.4f} {cnn_f1:>10.4f} {cnn_prec:>10.4f} {cnn_rec:>10.4f}")
    print(f"{'Word2Vec + LSTM':<25} {lstm_acc:>10.4f} {lstm_f1:>10.4f} {lstm_prec:>10.4f} {lstm_rec:>10.4f}")


if __name__ == "__main__":
    main()
