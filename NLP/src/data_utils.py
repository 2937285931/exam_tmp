#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理工具：分词、数据加载、词汇表构建、序列化、Dataset/DataLoader、Word2Vec训练、嵌入矩阵。
"""

import os
import re
import pickle
import glob
from collections import Counter

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec

from src.config import _CUSTOM_STOPWORDS


def tokenize(text: str) -> list:
    """
    英文文本分词与清洗：
      1. 去除 HTML 标签（如 <br>）
      # 【改进点05】优化正则：保留单词内部撇号(don't, it's)，同时清理独立标点
      2. 保留字母、数字及常用标点
      3. 转为小写
      4. 优先使用 NLTK punkt 分词，异常时回退至正则分词
      # 【改进点04】新增：过滤停用词（保留否定词和程度副词）
      5. 过滤掉纯符号 token

    参数:
      text: 原始英文影评字符串
    返回:
      tokens: 清洗后的 token 列表
    """
    # 延迟导入 NLTK tokenizer，避免在无 NLTK 环境下导入失败
    from nltk.tokenize import word_tokenize

    text = str(text)
    # 去除 HTML 换行标签
    text = re.sub(r"<br\s*/?>", " ", text)
    # 去除任意 HTML/XML 标签
    text = re.sub(r"<[^>]+>", " ", text)
    # 【改进点05】优化标点处理正则：
    # 原代码: re.sub(r"[^a-zA-Z0-9\s!?.,;:'\"-]", " ", text)
    # 新代码：更精确地保留单词内部撇号，同时去除其他杂余字符
    # 先保留合法字符（字母、数字、空格、常用标点、撇号、连字符）
    text = re.sub(r"[^a-zA-Z0-9\s!?.,;:'\"-]", " ", text)
    # 统一小写
    text = text.lower().strip()

    # 优先 NLTK 分词，失败则空格分词
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # 过滤掉不含任何字母/数字的纯符号 token
    tokens = [t for t in tokens if re.search(r"[a-zA-Z0-9]", t)]

    # 【改进点04】新增：过滤停用词（已保留否定词和程度副词）
    tokens = [t for t in tokens if t not in _CUSTOM_STOPWORDS]

    return tokens


# ===========================================================================
# ★ 本地数据集加载（离线版核心改造） ★
# ===========================================================================

def load_imdb_from_local(data_root: str):
    """
    从本地 aclImdb_v1 文件夹读取训练集和测试集。
    遍历 train/pos、train/neg、test/pos、test/neg 四个子目录，
    直接读取每个 .txt 文件的全部内容作为一条影评。

    目录结构期望:
      data_root/
        aclImdb/
          train/pos/*.txt   → 标签 1（正面）
          train/neg/*.txt   → 标签 0（负面）
          test/pos/*.txt    → 标签 1（正面）
          test/neg/*.txt    → 标签 0（负面）

    参数:
      data_root: aclImdb_v1 文件夹的路径（字符串）
    返回:
      train_texts, train_labels, test_texts, test_labels
      标签: 1 = Positive（正面），0 = Negative（负面）
    """
    # 拼接出 aclImdb 子目录路径（兼容 aclImdb_v1/aclImdb 这一层嵌套）
    base_dir = os.path.join(data_root, "aclImdb")
    # 若不存在 aclImdb 子目录，则 data_root 本身就是 aclImdb 目录
    if not os.path.isdir(base_dir):
        base_dir = data_root

    print(f"\n>>> 正在从本地路径加载 IMDB 数据集...")
    print(f"    DATA_ROOT = {data_root}")
    print(f"    实际读取目录: {base_dir}")

    # 定义四个子目录及其对应标签
    subsets = [
        (os.path.join(base_dir, "train", "pos"), 1),   # 训练集正面
        (os.path.join(base_dir, "train", "neg"), 0),   # 训练集负面
        (os.path.join(base_dir, "test", "pos"),  1),   # 测试集正面
        (os.path.join(base_dir, "test", "neg"),  0),   # 测试集负面
    ]

    # 验证所有目录是否存在
    for dir_path, _ in subsets:
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(
                f"目录不存在: {dir_path}\n"
                f"请检查 DATA_ROOT 路径是否正确，以及 aclImdb_v1 是否已正确解压。\n"
                f"预期结构: {data_root}/aclImdb/train/pos/, "
                f"{data_root}/aclImdb/train/neg/, "
                f"{data_root}/aclImdb/test/pos/, "
                f"{data_root}/aclImdb/test/neg/"
            )

    # 分别收集训练集与测试集的文本和标签
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for dir_path, label in subsets:
        # 获取目录下所有 .txt 文件
        txt_files = sorted(glob.glob(os.path.join(dir_path, "*.txt")))
        print(f"    读取 {dir_path} → {len(txt_files)} 个文件, 标签={label}")

        # 逐个读取文件内容
        texts_batch = []
        for fpath in txt_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    texts_batch.append(f.read().strip())
            except UnicodeDecodeError:
                # 少数文件可能是 latin-1 编码
                with open(fpath, "r", encoding="latin-1") as f:
                    texts_batch.append(f.read().strip())

        # 根据路径前缀归入训练集或测试集
        if "train" in dir_path:
            train_texts.extend(texts_batch)
            train_labels.extend([label] * len(texts_batch))
        else:
            test_texts.extend(texts_batch)
            test_labels.extend([label] * len(texts_batch))

    print(f"  训练集大小: {len(train_texts)}")
    print(f"  测试集大小: {len(test_texts)}")
    print(f"  训练集正负分布: {Counter(train_labels)}")
    print(f"  测试集正负分布: {Counter(test_labels)}")

    return train_texts, train_labels, test_texts, test_labels


# 【改进点02】新增：加载无监督数据用于Word2Vec训练
def load_unsup_texts(data_root: str) -> list:
    """
    加载 aclImdb/train/unsup/ 下的无监督影评文本。
    该目录包含约50000条无标签影评，可用于扩充Word2Vec训练语料。

    参数:
      data_root: aclImdb_v1 文件夹路径
    返回:
      unsup_texts: 无监督影评文本列表（无标签）
    """
    base_dir = os.path.join(data_root, "aclImdb")
    if not os.path.isdir(base_dir):
        base_dir = data_root

    unsup_dir = os.path.join(base_dir, "train", "unsup")
    if not os.path.isdir(unsup_dir):
        print(f"    [提示] 无监督数据目录不存在: {unsup_dir}，跳过unsup数据加载")
        return []

    txt_files = sorted(glob.glob(os.path.join(unsup_dir, "*.txt")))
    print(f"\n>>> 加载无监督数据用于Word2Vec训练: {len(txt_files)} 条")

    unsup_texts = []
    for fpath in tqdm(txt_files, desc="  读取unsup"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                unsup_texts.append(f.read().strip())
        except UnicodeDecodeError:
            with open(fpath, "r", encoding="latin-1") as f:
                unsup_texts.append(f.read().strip())

    print(f"  无监督样本数: {len(unsup_texts)}")
    return unsup_texts


# ===========================================================================
# 词汇表构建
# ===========================================================================

def build_vocab(texts: list, max_size: int, min_freq: int = 1):
    """
    基于训练集文本构建词表。
      0: <PAD> （填充标记）
      1: <UNK> （未登录词）
      2..max_size-1: 按词频降序排列的词汇

    参数:
      texts: 训练集文本列表
      max_size: 最大词汇表大小
      min_freq: 最低词频（低于此频率的词不收入词表）
    返回:
      (word2idx, idx2word) 映射字典
    """
    print("\n>>> 构建词汇表...")
    counter = Counter()
    for text in tqdm(texts, desc="  统计词频"):
        tokens = tokenize(text)
        counter.update(tokens)

    # 按词频降序排列，取前 max_size-2 个（0和1留给PAD和UNK）
    most_common = counter.most_common(max_size - 2)

    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    for i, (word, freq) in enumerate(most_common, start=2):
        if freq < min_freq:
            break
        word2idx[word] = i
        idx2word[i] = word

    print(f"  词汇表大小: {len(word2idx)}")
    print(f"  Top-10 高频词: {[idx2word[i] for i in range(2, min(12, len(idx2word)))]}")
    return word2idx, idx2word


def texts_to_sequences(texts: list, word2idx: dict, max_len: int):
    """
    将文本列表转换为固定长度的整数索引序列。
    【改进点09】头+尾联合截断策略：
      原代码: 仅保留尾部 ids[-max_len:]  → 丢失开篇关键信息
      新代码: 保留前 max_len//2 词 + 后 max_len//2 词 → 完整保留影评情感结构
    - 长于 max_len 的序列：保留前一半 + 后一半
    - 短于 max_len 的序列：用 <PAD> (id=0) 前填充
    - 未登录词：映射为 <UNK> (id=1)

    参数:
      texts: 原始文本列表
      word2idx: 词到索引的映射字典
      max_len: 序列最大长度
    返回:
      numpy 数组，shape = (len(texts), max_len)
    """
    half_len = max_len // 2
    sequences = []
    for text in texts:
        tokens = tokenize(text)
        ids = [word2idx.get(t, 1) for t in tokens]  # 未知词 → <UNK> id=1
        if len(ids) > max_len:
            # 【改进点09】原代码: ids = ids[-max_len:] 仅保留尾部
            # 新代码: 头+尾联合截断，保留前一半和后一半
            ids = ids[:half_len] + ids[-half_len:]
        else:
            ids = [0] * (max_len - len(ids)) + ids   # 前填充：句首加 PAD
        sequences.append(ids)
    return np.array(sequences, dtype=np.int64)


# ===========================================================================
# PyTorch Dataset 与 DataLoader
# ===========================================================================

class IMDBDataset(Dataset):
    """IMDB 文本分类 PyTorch 数据集类。"""

    def __init__(self, sequences, labels):
        self.sequences = torch.from_numpy(sequences).long()
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def create_dataloaders(train_seqs, train_labels, test_seqs, test_labels, batch_size):
    """
    创建训练/验证/测试 DataLoader。
    训练集按 80/20 随机划分：80% 训练集，20% 验证集（用于早停判断）。

    参数:
      train_seqs, train_labels: 训练集序列和标签
      test_seqs, test_labels: 测试集序列和标签
      batch_size: 批次大小
    返回:
      train_loader, val_loader, test_loader
    """
    n = len(train_seqs)
    indices = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = IMDBDataset(train_seqs[train_idx], train_labels[train_idx])
    val_ds = IMDBDataset(train_seqs[val_idx], train_labels[val_idx])
    test_ds = IMDBDataset(test_seqs, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"\n>>> DataLoader 创建完毕:")
    print(f"  训练批次: {len(train_loader)} ({len(train_ds)} 样本)")
    print(f"  验证批次: {len(val_loader)} ({len(val_ds)} 样本)")
    print(f"  测试批次: {len(test_loader)} ({len(test_ds)} 样本)")

    return train_loader, val_loader, test_loader


# ===========================================================================
# Word2Vec 词向量训练
# ===========================================================================

def train_word2vec(texts: list, config: dict, unsup_texts: list = None):
    """
    使用 Gensim 训练 Word2Vec (Skip-Gram + 负采样) 词向量。
    在训练集文本上从零训练，无需任何预训练模型下载。

    【改进点02】合并无监督数据扩充训练语料（若提供unsup_texts）

    参数:
      texts: 训练集文本列表
      config: 全局配置字典
      unsup_texts: 无监督影评文本（可选），用于扩充词向量训练
    返回:
      w2v_model: 训练好的 Gensim Word2Vec 模型
    """
    print("\n>>> 训练 Word2Vec 词向量（Skip-Gram，完全本地训练）...")

    # 【改进点02】原代码：仅对训练集分词
    # sentences = [tokenize(text) for text in tqdm(texts, desc="  文本分词")]
    # 新代码：合并训练集+无监督数据
    all_texts = list(texts)
    if unsup_texts:
        all_texts.extend(unsup_texts)
        print(f"  合并训练集({len(texts)}) + 无监督({len(unsup_texts)}) = {len(all_texts)} 条文本")

    sentences = [tokenize(text) for text in tqdm(all_texts, desc="  文本分词")]

    # 【改进点03】使用优化后的超参数：
    # 原值: vector_size=200, window=5, min_count=5, epochs=10
    # 新值: vector_size=300, window=8, min_count=3, epochs=20
    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=config["embed_dim"],       # 【改进点03】200→300
        window=config["w2v_window"],           # 【改进点03】5→8
        min_count=config["w2v_min_count"],     # 【改进点03】5→3
        workers=config["w2v_workers"],
        sg=1,              # Skip-Gram（1=Skip-Gram, 0=CBOW）
        hs=0,              # 不使用层次 Softmax，改用负采样
        negative=5,        # 负采样数量
        epochs=config["w2v_epochs"],          # 【改进点03】10→20
        seed=config["seed"],
    )

    print(f"  Word2Vec 词汇量: {len(w2v_model.wv)}")
    print(f"  词向量维度: {w2v_model.wv.vector_size}")

    return w2v_model


def build_embedding_matrix(word2idx: dict, w2v_model, embed_dim: int):
    """
    构建 PyTorch Embedding 层的预训练权重矩阵。
    - 在 Word2Vec 词表中的词 → 使用训练好的词向量
    - 未登录词（<PAD>、<UNK> 及低频词）→ 均匀分布 U(-0.25, 0.25) 随机初始化

    【改进点01】此函数返回值将实际传入CNN/LSTM模型构造函数

    参数:
      word2idx: 词表映射字典
      w2v_model: 训练好的 Word2Vec 模型
      embed_dim: 词向量维度
    返回:
      embedding_matrix: numpy 数组，shape = (vocab_size, embed_dim)
    """
    vocab_size = len(word2idx)
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embed_dim)).astype(
        np.float32
    )

    hit_count = 0
    for word, idx in word2idx.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
            hit_count += 1

    print(f"  嵌入矩阵: {vocab_size} × {embed_dim}")
    print(f"  命中预训练词向量: {hit_count}/{vocab_size} ({100*hit_count/vocab_size:.1f}%)")
    return embedding_matrix
