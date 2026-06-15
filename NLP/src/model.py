#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型定义：TextCNN、SelfAttention、TextLSTM。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    TextCNN 模型 — Yoon Kim (2014) EMNLP.
    使用多尺度卷积核 (3, 4, 5) 捕获不同长度的 n-gram 局部特征。
    每个卷积核在句子矩阵上滑动，经 ReLU 激活 + 1-MaxPooling，
    将多尺度特征拼接后经 Dropout + BatchNorm + 全连接层输出二分类 logits。

    【改进点01】Embedding层使用预训练Word2Vec词向量初始化
    【改进点11】FC层前增加BatchNorm1d，稳定训练
    【改进点15】Dropout 0.5→0.3，配合更宽模型

    参数量估算（词表25000, embed_dim=300, filter=128, kernel=[3,4,5]）：
      ≈ 7.5M (Embedding) + 0.46M (Conv) = 约 8M
    """

    def __init__(self, vocab_size, embed_dim, config, embedding_matrix=None):
        super(TextCNN, self).__init__()

        # 【改进点01】原代码：从零随机初始化Embedding
        # self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 新代码：使用预训练Word2Vec词向量初始化，允许微调
        if embedding_matrix is not None:
            embedding_tensor = torch.from_numpy(embedding_matrix).float()
            self.embedding = nn.Embedding.from_pretrained(
                embedding_tensor, freeze=False, padding_idx=0
            )
            print(f"  CNN Embedding: 已加载预训练词向量 ({embedding_matrix.shape})")
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            print(f"  CNN Embedding: 从零随机初始化（未提供预训练词向量）")

        # 多尺度卷积核：每个 kernel_size 有 filters 个卷积核
        # Conv2d(in_channels=1 表示单通道文本, out_channels=filters)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=config["cnn_filters"],
                kernel_size=(ks, embed_dim),  # 卷积核宽度=词向量维度（整行卷积）
            )
            for ks in config["cnn_kernel_sizes"]
        ])

        self.dropout = nn.Dropout(config["cnn_dropout"])
        # 全连接输入维度 = 卷积核种类数 × 每种滤波器数
        fc_input_dim = len(config["cnn_kernel_sizes"]) * config["cnn_filters"]
        # 【改进点11】FC层前增加BatchNorm1d：
        # 原代码: self.fc = nn.Linear(fc_input_dim, 1)
        # 新代码: 加入BatchNorm稳定隐状态分布，加速收敛
        self.bn = nn.BatchNorm1d(fc_input_dim)
        self.fc = nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        embedded = embedded.unsqueeze(1)      # (batch, 1, seq_len, embed_dim)  增加通道维

        conv_outputs = []
        for conv in self.convs:
            out = F.relu(conv(embedded))      # (batch, filters, seq_len-ks+1, 1)
            out = out.squeeze(3)              # (batch, filters, seq_len-ks+1)
            out = F.max_pool1d(out, out.size(2))  # (batch, filters, 1)  全局最大池化
            conv_outputs.append(out.squeeze(2))   # (batch, filters)

        pooled = torch.cat(conv_outputs, dim=1)   # (batch, 3*filters)
        # 【改进点11】BatchNorm → Dropout → FC 顺序
        pooled = self.bn(pooled)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled).squeeze(1)        # (batch,)  logits（未过sigmoid）
        return logits


# 【改进点12】新增：自注意力层 — 缩放点积自注意力 (Scaled Dot-Product Self-Attention)
class SelfAttention(nn.Module):
    """
    缩放点积自注意力层。
    对BiLSTM所有时间步输出计算注意力权重，加权求和得到上下文向量。
    使模型能够"关注"序列中最具情感区分度的词/短语。

    原理：Attention(Q,K,V) = softmax(QK^T / √d_k) × V
    其中 Q=K=V = LSTM输出的线性投影（自注意力模式）
    """

    def __init__(self, input_dim: int, attention_dim: int):
        super(SelfAttention, self).__init__()
        self.attention_dim = attention_dim
        # 将输入映射到注意力空间
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, input_dim)  # 【修复】value投影维度应为input_dim(512)以匹配下游BatchNorm1d，非attention_dim(128)
        # 缩放因子
        self.scale = attention_dim ** 0.5

    def forward(self, lstm_outputs, mask=None):
        """
        参数:
          lstm_outputs: (batch, seq_len, input_dim) — BiLSTM所有时间步输出
          mask: (batch, seq_len) — True表示有效位置，False表示PAD（可选）
        返回:
          context: (batch, input_dim) — 注意力加权上下文向量
          attn_weights: (batch, seq_len) — 注意力权重分布（用于分析）
        """
        Q = self.query(lstm_outputs)   # (batch, seq_len, attention_dim)
        K = self.key(lstm_outputs)     # (batch, seq_len, attention_dim)
        V = self.value(lstm_outputs)   # (batch, seq_len, input_dim)

        # 计算注意力分数: (batch, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 对PAD位置施加极大负偏置，使其注意力权重趋近于0
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == False, -1e9)

        # softmax归一化得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, seq_len, seq_len)

        # 加权求和: 每个位置的输出是所有位置的加权和
        context = torch.matmul(attn_weights, V)  # (batch, seq_len, input_dim)

        # 对所有位置取平均得到句子级表示
        context = context.mean(dim=1)  # (batch, input_dim)

        return context, attn_weights


class TextLSTM(nn.Module):
    """
    BiLSTM 文本分类模型（优化版）。
    使用双层双向 LSTM 对序列进行编码，经自注意力层加权聚合所有时间步信息，
    再经 BatchNorm + Dropout + 全连接层输出二分类 logits。

    【改进点01】Embedding层使用预训练Word2Vec词向量初始化
    【改进点11】FC层前增加BatchNorm1d
    【改进点12】LSTM输出后增加自注意力层（Self-Attention）
      原代码: 直接取末层正反向最后时刻隐状态拼接 → 丢失早期关键信息
      新代码: 自注意力加权聚合所有时间步 → 模型关注关键情感词
    【改进点15】Dropout 0.5→0.3

    参数量估算（embed_dim=300, hidden=256, layers=2, bidirectional=True）：
      单向 LSTM 每层 ≈ 4 × ((300+256) × 256 + 256²) ≈ 832K
      双层双向 ≈ 832K × 2 × 2 ≈ 3.3M（不含 Embedding 约7.5M 和 Attention 约0.3M）
    """

    def __init__(self, vocab_size, embed_dim, config, embedding_matrix=None):
        super(TextLSTM, self).__init__()

        # 【改进点01】原代码：从零随机初始化Embedding
        # self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 新代码：使用预训练Word2Vec词向量初始化，允许微调
        if embedding_matrix is not None:
            embedding_tensor = torch.from_numpy(embedding_matrix).float()
            self.embedding = nn.Embedding.from_pretrained(
                embedding_tensor, freeze=False, padding_idx=0
            )
            print(f"  LSTM Embedding: 已加载预训练词向量 ({embedding_matrix.shape})")
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            print(f"  LSTM Embedding: 从零随机初始化（未提供预训练词向量）")

        # 双层 BiLSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=config["lstm_hidden_dim"],
            num_layers=config["lstm_layers"],
            batch_first=True,                                              # 输入格式 (batch, seq, feature)
            dropout=config["lstm_dropout"] if config["lstm_layers"] > 1 else 0.0,
            bidirectional=config["lstm_bidirectional"],
        )

        # LSTM 输出维度：hidden_dim × (双向?2:1)
        lstm_output_dim = config["lstm_hidden_dim"] * (
            2 if config["lstm_bidirectional"] else 1
        )

        # 【改进点12】新增自注意力层
        # 原代码：直接使用末层最后时刻隐状态，无注意力机制
        # 新代码：自注意力加权聚合所有时间步输出
        self.attention = SelfAttention(lstm_output_dim, config["attention_dim"])

        self.dropout = nn.Dropout(config["lstm_dropout"])
        # 【改进点11】FC层前增加BatchNorm1d：
        # 原代码: self.fc = nn.Linear(lstm_output_dim, 1)
        # 新代码: 加入BatchNorm稳定隐状态分布
        self.bn = nn.BatchNorm1d(lstm_output_dim)  # 修复BatchNorm维度匹配：512 = 256×2(bidirectional)
        self.fc = nn.Linear(lstm_output_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)                        # (batch, seq_len, embed_dim)

        # 构建PAD mask用于注意力层屏蔽填充位置
        pad_mask = (x != 0)  # (batch, seq_len) True=有效词, False=PAD

        lstm_out, (hidden, cell) = self.lstm(embedded)     # lstm_out: (batch, seq_len, lstm_output_dim)

        # 【改进点12】自注意力加权聚合所有时间步输出
        # 原代码（仅用末层最后时刻隐状态）：
        #   if self.lstm.bidirectional:
        #       hidden_fwd = hidden[-2, :, :]   # 正向最后一层
        #       hidden_bwd = hidden[-1, :, :]   # 反向最后一层
        #       hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        #   else:
        #       hidden_cat = hidden[-1, :, :]
        # 新代码：自注意力聚合所有时间步信息
        context, attn_weights = self.attention(lstm_out, pad_mask)

        hidden_cat = self.bn(context)                      # 【改进点11】BatchNorm
        hidden_cat = self.dropout(hidden_cat)              # 【改进点15】Dropout=0.3
        logits = self.fc(hidden_cat).squeeze(1)             # (batch,)  logits（未过sigmoid）
        return logits
