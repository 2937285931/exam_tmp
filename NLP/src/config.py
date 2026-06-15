#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全局配置、随机种子、NLTK资源下载、停用词表。
"""

import os
import sys
import random

import numpy as np
import torch

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ ★★★ 在这里修改路径！只需改这一行，指向你本机的 aclImdb_v1 文件夹 ★★★  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# DATA_ROOT 是数据集根目录，其下应有 aclImdb/train/pos、train/neg、test/pos、test/neg 四个子目录
# 示例（Windows）：DATA_ROOT = "F:/datasets/aclImdb_v1"
# 示例（Linux/macOS）：DATA_ROOT = "/home/user/data/aclImdb_v1"
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(_PROJECT_ROOT, "aclImdb_v1")

CONFIG = {
    # --- 文本预处理 ---
    "max_vocab_size": 25000,      # 最大词汇表大小（按词频截取）
    # 【改进点09】max_seq_len 256→384，配合头+尾联合截断策略保留更多上下文
    "max_seq_len": 384,           # 序列最大长度（截断/填充）— 原值256
    # --- Word2Vec 参数 ---
    # 【改进点03】embed_dim 200→300，增强词向量表达力
    "embed_dim": 300,             # 词向量维度 — 原值200
    # 【改进点03】w2v_window 5→8，扩大上下文窗口捕获更远词共现
    "w2v_window": 8,              # 上下文窗口大小 — 原值5
    # 【改进点03】w2v_min_count 5→3，保留更多低频情感词汇
    "w2v_min_count": 3,           # 最低词频阈值 — 原值5
    "w2v_workers": 4,             # 训练并行线程数
    # 【改进点03】w2v_epochs 10→20，充分训练词向量
    "w2v_epochs": 20,             # Word2Vec 训练轮数 — 原值10
    # --- CNN 超参数 ---
    # 【改进点14】cnn_filters 100→128，增强CNN特征提取能力
    "cnn_filters": 128,           # 每种卷积核数量 — 原值100
    "cnn_kernel_sizes": [3, 4, 5],  # 多尺度卷积核（等价 3-gram, 4-gram, 5-gram）
    # 【改进点15】cnn_dropout 0.5→0.3，减少过强正则化导致的信息瓶颈
    "cnn_dropout": 0.3,           # — 原值0.5
    # --- LSTM 超参数 ---
    # 【改进点14】lstm_hidden_dim 128→256，增强LSTM表达力
    "lstm_hidden_dim": 256,       # 隐藏层维度 — 原值128
    "lstm_layers": 2,             # LSTM 层数
    # 【改进点15】lstm_dropout 0.5→0.3，配合更宽隐藏层保持信息流
    "lstm_dropout": 0.3,          # — 原值0.5
    "lstm_bidirectional": True,   # 双向 LSTM
    # 【改进点12】注意力层维度设置
    "attention_dim": 128,         # 自注意力层隐射维度（新增参数）
    # --- 训练通用参数 ---
    "batch_size": 64,
    "learning_rate": 0.001,
    # 【改进点06】新增weight_decay用于AdamW的L2正则化
    "weight_decay": 1e-4,         # L2正则化系数（新增参数）
    # 【改进点10】标签平滑系数
    "label_smoothing": 0.05,      # 标签平滑因子，0表示不使用（新增参数）
    # 【改进点13】epochs 10→20，给模型更充分训练时间
    "epochs": 20,                 # 最大训练轮数 — 原值10
    # 【改进点13】patience 3→5，配合更多epochs
    "patience": 5,                # 早停耐心值 — 原值3
    # 【改进点16】grad_clip 5.0→3.0，更积极抑制梯度尖峰
    "grad_clip": 3.0,             # 梯度裁剪阈值 — 原值5.0
    # --- 系统 ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "saved_dir": "./results",
}


def set_seed(seed: int) -> None:
    """固定随机种子以确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    """若目录不存在则创建。"""
    os.makedirs(path, exist_ok=True)


def download_nltk_resources() -> None:
    """
    下载 NLTK punkt 分词模型和停用词语料库。
    原代码：仅下载punkt分词模型。
    【改进点04】新增停用词资源下载（stopwords）。
    若网络不可达则回退至简单空格分词（不影响实验整体进行）。
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            try:
                nltk.download("punkt", quiet=True)
            except Exception:
                pass  # 下载失败时回退至空格分词

    # 【改进点04】新增：下载停用词资源
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
        except Exception:
            pass


download_nltk_resources()


# 【改进点04】构建自定义停用词表：使用NLTK英文停用词，但保留情感相关的否定词和程度副词
def _build_stopwords() -> set:
    """构建自定义停用词集合，保留否定词和程度副词以维护情感极性。"""
    try:
        base_stopwords = set(stopwords.words("english"))
    except Exception:
        base_stopwords = set()
    # 保留对情感分析至关重要的词汇：否定词 + 程度副词 + 转折词
    keep_words = {
        "not", "no", "nor", "but", "very", "too", "only", "against",
        "up", "down", "most", "more", "least", "less", "few", "off",
        "over", "under", "again", "further", "then", "once", "here",
        "there", "all", "any", "each", "every", "both", "some", "such",
        "just", "so", "than", "until", "above", "below", "yet", "still",
    }
    return base_stopwords - keep_words


# 模块加载时构建停用词表，避免每次调用tokenize重复构建
_CUSTOM_STOPWORDS = _build_stopwords()
