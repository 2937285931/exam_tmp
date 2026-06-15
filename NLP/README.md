# IMDB情感分析实验 — 离线版（Word2Vec + CNN / LSTM）

## 实验概述

对IMDB影评数据集（50000条）进行正面/负面情感二分类。实现三组对照模型：
- **TF-IDF + Logistic Regression**（传统机器学习基线）
- **Word2Vec + CNN**（TextCNN, Kim 2014）
- **Word2Vec + BiLSTM**（核心模型）

★ **本版已改为纯离线运行**：不再依赖 HuggingFace Datasets / torchtext 联网下载，直接读取本地解压后的 `aclImdb_v1` 文件夹中的 `.txt` 影评文件。

---

## 前置准备

### 1. 数据集下载与解压

从 Stanford 官网下载 aclImdb_v1 数据集压缩包：

```bash
# 方式一：浏览器直接下载
# https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

# 方式二：wget 命令行
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
```

解压到项目目录（或其他任意位置）：

```bash
tar -xzf aclImdb_v1.tar.gz
```

解压后的目录结构：

```
aclImdb_v1/
  aclImdb/
    train/
      pos/    ← 12500 条正面训练影评 (*.txt)
      neg/    ← 12500 条负面训练影评 (*.txt)
    test/
      pos/    ← 12500 条正面测试影评 (*.txt)
      neg/    ← 12500 条负面测试影评 (*.txt)
```

### 2. 修改 DATA_ROOT 路径

打开 `sentiment_analysis.py`，找到文件头部的 **第71行** 附近的 `DATA_ROOT` 变量（代码中有醒目标注），将其修改为你本机 `aclImdb_v1` 文件夹的实际路径：

```python
# ★★★ 在这里修改路径！只需改这一行 ★★★
DATA_ROOT = "C:/Users/你的用户名/datasets/aclImdb_v1"        # Windows 示例
# DATA_ROOT = "/home/你的用户名/data/aclImdb_v1"              # Linux/macOS 示例
```

> 路径末尾不要带 `/aclImdb`，代码会自动处理嵌套子目录。

### 3. 安装 Python 依赖

```bash
cd daima

# 安装 PyTorch (CPU 版本；如需 GPU 请访问 pytorch.org)
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# 安装其余依赖（推荐一键安装）
pip install -r requirements.txt
```

**依赖清单：**

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 运行环境 |
| PyTorch | 2.0.1 | 深度学习框架 |
| scikit-learn | 1.7.2 | TF-IDF + 逻辑回归基线 |
| gensim | 4.4.0 | Word2Vec词向量训练 |
| matplotlib | 3.10+ | 训练曲线可视化 |
| numpy | 2.2+ | 数值计算 |
| nltk | 3.9.4 | 英文分词 |
| tqdm | 4.67+ | 进度条 |


---

## 一键启动（单文件版）

```bash
cd daima
python sentiment_analysis.py
```

代码会依次执行：

1. **本地数据加载** — 遍历 `aclImdb_v1/aclImdb/` 下的四个子目录，读取全部 `.txt` 文件
2. **文本预处理** — 去HTML标签、去符号、小写化、NLTK分词
3. **词汇表构建** — 按词频统计，保留前 25000 个高频词
4. **Word2Vec 词向量训练** — Gensim Skip-Gram + 负采样，完全本地训练
5. **三组模型训练与评估**：
   - TF-IDF + Logistic Regression（机器学习基线）
   - Word2Vec + CNN（TextCNN, Kim 2014）
   - Word2Vec + BiLSTM（核心模型）
6. **训练曲线可视化** — CNN & LSTM 的 Loss/Acc 图
7. **LSTM 错误案例分析** — 8条典型错分样本深度分析
8. **结果汇总输出** — 指标对比、超参数记录、分析说明

**预计运行时间：** CPU环境约 30-60 分钟（视硬件配置而定）。

---

## 模块化启动方式（推荐）

项目已重构为模块化结构，所有核心代码位于 `src/` 目录下，支持独立运行各阶段。

### 1. 训练（train.py）

训练主脚本，执行完整实验流程：数据加载 → 预处理 → 词表构建 → Word2Vec训练 → 模型训练 → 评估 → 可视化 → 错误分析 → 结果汇总。

**启动方式：**

```bash
cd daima
python -m src.train
# 或
python src/train.py
```

**执行流程：**
1. 从本地 `aclImdb_v1/` 加载训练集和测试集
2. 加载 `train/unsup/` 无监督数据（约50000条）扩充Word2Vec训练语料
3. 构建词汇表（max_vocab_size=25000），保存至 `results/vocab.pkl`
4. 文本序列化（头+尾联合截断，max_seq_len=384）
5. 训练 Word2Vec 词向量（Skip-Gram, embed_dim=300, window=8, epochs=20），保存至 `results/w2v_model.gensim`
6. 构建预训练嵌入矩阵，供 CNN/LSTM 的 Embedding 层加载
7. 训练 TF-IDF + Logistic Regression 基线模型，保存 `results/tfidf_vectorizer.pkl` 和 `results/tfidf_lr_model.pkl`
8. 训练 Word2Vec + TextCNN 模型（含早停、学习率衰减），保存最佳权重至 `results/cnn_model.pth`
9. 训练 Word2Vec + BiLSTM + SelfAttention 模型，保存最佳权重至 `results/lstm_model.pth`
10. 绘制 CNN/LSTM 训练曲线，保存至 `results/loss_curve.png`
11. 分析 LSTM 错误案例（FP/FN 各4条），保存至 `results/error_cases.txt`
12. 生成结果汇总报告，保存至 `results/results.txt`

**输出内容与位置：**

| 输出文件 | 位置 | 说明 |
|----------|------|------|
| `vocab.pkl` | `./results/` | 词汇表（word2idx / idx2word 映射字典） |
| `w2v_model.gensim` | `./results/` | 训练好的 Gensim Word2Vec 词向量模型 |
| `tfidf_vectorizer.pkl` | `./results/` | TF-IDF 向量化器（unigram+bigram, max_features=50000） |
| `tfidf_lr_model.pkl` | `./results/` | TF-IDF + 逻辑回归分类器 |
| `cnn_model.pth` | `./results/` | TextCNN 模型最佳权重（验证集Acc最高时保存） |
| `lstm_model.pth` | `./results/` | BiLSTM+Attention 模型最佳权重（验证集Acc最高时保存） |
| `loss_curve.png` | `./results/` | CNN与LSTM训练/验证损失和准确率曲线（2×2子图） |
| `error_cases.txt` | `./results/` | LSTM错误分类样本分析（FP/FN各4条，含影评片段和混淆词） |
| `results.txt` | `./results/` | 完整实验结果汇总（指标表格+原理分析+超参数+参考文献） |

**运行时终端输出：**
- 每个 Epoch 的 Train Loss/Acc、Val Loss/Acc/F1、当前学习率、耗时
- 最佳模型更新提示（Val Acc 提升时）
- 早停触发提示
- 最终三模型测试集指标对比表格

---

### 2. 评估（evaluate.py）

独立评估脚本，加载 `results/` 目录下训练好的模型权重和工件，在测试集上评估 CNN 和 LSTM 模型性能。不执行训练，仅做推理评估。

**前置条件：** 必须已运行 `train.py` 并在 `results/` 目录下生成以下文件：
- `vocab.pkl`
- `w2v_model.gensim`
- `cnn_model.pth`
- `lstm_model.pth`

**启动方式：**

```bash
cd daima
python -m src.evaluate
# 或
python src/evaluate.py
```

**执行流程：**
1. 从 `results/` 加载词汇表 (`vocab.pkl`)
2. 从 `results/` 加载 Word2Vec 模型 (`w2v_model.gensim`) 并构建嵌入矩阵
3. 从本地 `aclImdb_v1/` 加载测试集（25000条）
4. 文本序列化并创建测试 DataLoader
5. 加载 CNN 最佳权重 (`results/cnn_model.pth`)，评估测试集指标
6. 加载 LSTM 最佳权重 (`results/lstm_model.pth`)，评估测试集指标
7. 输出两模型对比汇总表格

**输出内容与位置：**
- 本脚本不产生新文件，仅在终端输出评估结果

**运行时终端输出：**
- 词汇表加载信息（词数）
- 测试集加载信息（样本数）
- CNN 模型参数量
- CNN Test Result: Acc / F1 / Precision / Recall
- LSTM 模型参数量
- LSTM Test Result: Acc / F1 / Precision / Recall
- 最终汇总对比表格

---

### 3. 推理（inference.py）

独立推理脚本，加载训练好的 LSTM 模型，提供交互式命令行界面，支持输入单条英文影评文本，实时输出情感预测结果（正面/负面）及置信度。

**前置条件：** 必须已运行 `train.py` 并在 `results/` 目录下生成以下文件：
- `vocab.pkl`
- `w2v_model.gensim`
- `lstm_model.pth`

**启动方式：**

```bash
cd daima
python -m src.inference
# 或
python src/inference.py
```

**交互方式：**
- 运行后进入交互模式，提示 `请输入英文影评文本:`
- 输入英文影评文本，按回车获取预测结果
- 输入 `quit`、`exit` 或 `q` 退出
- 按 `Ctrl+C` 或 `Ctrl+D` 也可退出

**输出内容与位置：**
- 本脚本不产生新文件，仅在终端输出预测结果
- 每条输入输出包含：
  - 情感类别（Positive / Negative）
  - 置信度（0~1，越接近边界0.5表示模型越不确定）
  - Positive 概率
  - Negative 概率

**运行时终端输出示例：**
```
请输入英文影评文本: This movie was absolutely fantastic! Great acting and storyline.

  情感类别: Positive
  置信度:   0.9234
  Positive 概率: 0.9234
  Negative 概率: 0.0766
```

**编程调用方式：**

```python
from src.inference import SentimentPredictor

predictor = SentimentPredictor()

# 单条预测
result = predictor.predict("This movie was great!")
print(result)  # {"sentiment": "Positive", "confidence": 0.92, ...}

# 批量预测
results = predictor.predict_batch(["Great film!", "Terrible movie."])
for r in results:
    print(r["sentiment"], r["confidence"])
```

---

### 4. 训练工具函数（train_utils.py）

`train_utils.py` 是训练与评估框架模块，不独立运行，被 `train.py` 和 `evaluate.py` 导入使用。提供以下核心组件：

| 组件 | 说明 |
|------|------|
| `EarlyStopping` | 早停机制类。监控验证损失，在 `patience` 轮内无改善（下降超过 `delta`）时触发停止，防止过拟合 |
| `train_epoch()` | 执行单个训练轮次。包含前向传播、损失计算（支持标签平滑）、反向传播、梯度裁剪、参数更新 |
| `evaluate()` | 模型评估函数。返回 avg_loss, accuracy, F1(binary), precision, recall, all_preds, all_labels |
| `train_model()` | 通用训练循环。整合早停、AdamW优化器（含weight_decay L2正则）、ReduceLROnPlateau学习率衰减、BCEWithLogitsLoss（含pos_weight类别权重平衡）、最佳模型内存保存+磁盘持久化 |

**train_model() 训练的每一步输出（终端）：**
```
============================================================
训练 CNN 模型
============================================================
  正样本数: 10000, 负样本数: 10000, pos_weight: 1.000
  Epoch  1/20 | Train Loss: 0.4521 Acc: 0.7856 | Val Loss: 0.3612 Acc: 0.8412 F1: 0.8398 | LR: 0.001000 | Time: 12.3s
  >>> 更新最佳模型 (Val Acc: 0.8412)
  Epoch  2/20 | Train Loss: 0.3215 Acc: 0.8651 | Val Loss: 0.3102 Acc: 0.8623 F1: 0.8607 | LR: 0.001000 | Time: 11.8s
  >>> 更新最佳模型 (Val Acc: 0.8623)
  ...
  早停触发! 最佳验证准确率: 0.8789
  CNN 训练完成，最佳验证 Acc: 0.8789
```

---

## 输出文件

运行完成后，全部输出保存至 `./saved/` 目录：

| 文件 | 说明 |
|------|------|
| `vocab.pkl` | 词汇表字典（word2idx / idx2word 映射） |
| `w2v_model.gensim` | 训练好的 Word2Vec 词向量模型（Gensim格式） |
| `cnn_model.pth` | TextCNN 模型最佳权重 |
| `lstm_model.pth` | BiLSTM 模型最佳权重 |
| `tfidf_vectorizer.pkl` | TF-IDF 向量化器 |
| `tfidf_lr_model.pkl` | TF-IDF + LR 分类器 |
| `training_curves.png` | CNN & LSTM 训练/验证损失与准确率曲线（2×2子图） |
| `error_cases.txt` | LSTM 错误分类样本分析（假阳性/假阴性各4条） |
| `results.txt` | 完整实验结果汇总：指标表格、原理分析、超参数、参考文献 |

---

## 自定义超参数

打开 `sentiment_analysis.py`，修改 `CONFIG` 字典（第84-117行）中的参数：

```python
CONFIG = {
    "max_vocab_size": 25000,    # 词汇表大小
    "max_seq_len": 256,         # 序列最大长度
    "embed_dim": 200,           # 词向量维度
    "batch_size": 64,           # 批次大小
    "epochs": 10,               # 最大训练轮数
    "learning_rate": 0.001,     # 学习率
    "lstm_hidden_dim": 128,     # LSTM 隐藏层维度
    "lstm_layers": 2,           # LSTM 层数
    "cnn_filters": 100,         # CNN 每种卷积核数量
    # ... 更多参数见代码注释
}
```

---

## 模型架构说明

### TF-IDF + Logistic Regression（基线）
- 特征：unigram + bigram TF-IDF，最大50000维，sublinear TF缩放，去除英文停用词
- 分类器：L2正则化逻辑回归（C=1.0），L-BFGS求解器
- 角色：验证深度学习模型相比传统方法的提升幅度

### Word2Vec + TextCNN（Kim 2014）
- **原始版**：词嵌入：200维 Word2Vec 预训练向量（允许微调）；卷积层：3组并行卷积核（kernel=3/4/5，各100个滤波器）；池化：1-MaxPooling → 拼接 → Dropout(0.5) → 全连接
- **优化版**：词嵌入：300维预训练向量加载；卷积层：各128个滤波器；池化：1-MaxPooling → 拼接 → BatchNorm → Dropout(0.3) → 全连接

### Word2Vec + BiLSTM（核心模型）
- **原始版**：词嵌入：200维 Word2Vec 预训练向量（允许微调）；编码器：2层双向 LSTM（隐藏维度128）；分类：拼接末层正反向隐状态 → Dropout(0.5) → 全连接
- **优化版**：词嵌入：300维 Word2Vec 预训练向量（加载自unsup扩充语料训练）；编码器：2层双向 LSTM（隐藏维度256）；自注意力层：缩放点积自注意力聚合全部时间步；分类：BatchNorm → Dropout(0.3) → 全连接；损失：BCEWithLogitsLoss + 类别权重 + 标签平滑；优化器：AdamW + L2正则化

---

## 调优说明（2026-06-13 新增）

### 优化版代码

优化后代码位于 `sentiment_analysis_optimized.py`，与原始代码并行保留，不会覆盖原文件。

启动方式不变：
```bash
cd D:/exam_tmp-master/daima
python sentiment_analysis_optimized.py
```

### 17项优化概览

原始代码存在预训练词向量未加载、欠正则化/过度正则化并存、模型容量不足、缺少注意力机制等系统性缺陷，导致Acc/F1偏低。

优化后预期提升至 **89-92% Acc/F1**。详细改进方案对照表见 `docs/improvement_table.md`。

| 优先级 | 优化项 | 预期贡献 |
|---|---|---|
| **致命** | ① Embedding层加载预训练Word2Vec词向量 | Acc/F1 +3~5% |
| **高** | ⑥ AdamW weight_decay L2正则化 | Acc +1~2% |
| **高** | ⑫ LSTM增加自注意力(Self-Attention) | Acc/F1 +1~2% |
| **高** | ② Word2Vec加入unsup数据(50K)扩充语料 | Acc/F1 +1~2% |
| **中** | ⑭ LSTM hidden 128→256, CNN filters 100→128 | Acc +0.5~1% |
| **中** | ⑪ FC层前增加BatchNorm1d | Acc +0.5~1% |
| **中** | ③ Word2Vec超参数优化(embed_dim/window/min_count/epochs) | Acc +0.5~1% |
| **中** | ⑦ ReduceLROnPlateau patience 1→3 | Acc +0.5~1% |
| **中** | ⑨ 头+尾联合截断, max_seq_len 256→384 | Acc +0.5~1% |
| **中** | ⑬ epochs 10→20, patience 3→5 | Acc +0.5~1% |
| **低-中** | ④ 停用词过滤(保留否定词/程度副词) | Acc +0.5% |
| **低-中** | ⑧ pos_weight类别权重平衡 | F1 +0.5~1% |
| **低-中** | ⑩ 标签平滑 LabelSmoothing=0.05 | F1 +0.5% |
| **低-中** | ⑤ 优化标点处理正则 | Acc +0.3~0.5% |
| **低-中** | ⑮ Dropout 0.5→0.3 | Acc +0.5% |
| **低** | ⑯ 梯度裁剪 5.0→3.0 | 训练稳定性 |
| **低** | ⑰ F1改用binary平均 | 指标准确性 |

### 新增可调节超参数

以下为优化版新增/调整的超参数，可在 `sentiment_analysis_optimized.py` 的 `CONFIG` 字典中微调：

```python
CONFIG = {
    # ... 原有参数保持兼容 ...

    # === 新增超参数 ===
    "weight_decay": 1e-4,         # L2正则化系数。增大→更强正则化，范围[1e-5, 1e-3]
    "label_smoothing": 0.05,      # 标签平滑因子。增大→更平滑，范围[0, 0.1]
    "attention_dim": 128,         # 自注意力隐射维度。可以是64/128/256

    # === 调整超参数（括号内为原值） ===
    "embed_dim": 300,             # (原200) 词向量维度，需与attention模型匹配
    "max_seq_len": 384,           # (原256) 序列长度须为偶数（头+尾各半）
    "w2v_window": 8,              # (原5) 上下文窗口
    "w2v_min_count": 3,           # (原5) 最低词频
    "w2v_epochs": 20,             # (原10) Word2Vec训练轮数
    "cnn_filters": 128,           # (原100) CNN滤波数
    "cnn_dropout": 0.3,           # (原0.5) CNN Dropout
    "lstm_hidden_dim": 256,       # (原128) LSTM隐藏维度
    "lstm_dropout": 0.3,          # (原0.5) LSTM Dropout
    "epochs": 20,                 # (原10) 最大训练轮数
    "patience": 5,                # (原3) 早停耐心值
    "grad_clip": 3.0,             # (原5.0) 梯度裁剪阈值
}
```

### 进一步涨点建议

如果优化后仍希望进一步提升指标：

1. **词向量替换**：将Word2Vec替换为GloVe 300d预训练向量（需联网下载一次）
2. **数据增强**：对反讽、对比句式进行回译增强（英文→法文→英文），扩充训练集
3. **集成学习**：CNN + LSTM + Attention 三模型soft voting集成，通常可再提升1-2%
4. **对抗训练**：使用FGM/FGSM对Embedding层施加对抗扰动增强鲁棒性
5. **更大的词表**：max_vocab_size增大至50000（需配合更大的embed_dim）

### 错分样本对比说明

优化后预期变化：
- **总错分样本数**：减少30-50%（自注意力机制缓解长文本信息丢失）
- **假阳性(FP)**：减少显著（停用词过滤减少正面词汇引发的误判）
- **假阴性(FN)**：减少中等（预训练词向量改善低频情感词表示）
- **高置信度错误**：减少（标签平滑降低过度自信）

> 每次运行后可在 `saved/error_cases.txt` 查看具体错分案例对比。

---

## 参考文献

1. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. *EMNLP 2014*.
2. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
3. Mikolov, T. et al. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR 2013*.
4. Maas, A. L. et al. (2011). Learning Word Vectors for Sentiment Analysis. *ACL 2011*.
5. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
6. Bahdanau, D. et al. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR 2015*.

---

## 项目文件结构

```
daima/
├── sentiment_analysis.py              # 原始代码（完整保留，不做修改）
├── sentiment_analysis_optimized.py    # 优化版代码（17项改进，含【改进点】标记）
├── README.md                          # 本文件（含调优说明）
├── docs/
│   └── improvement_table.md           # 问题与改进方案对照表（详细）
├── src/                               # ★ 模块化源代码目录
│   ├── __init__.py                    # 包初始化文件
│   ├── config.py                      # 全局配置、随机种子、NLTK资源、停用词表
│   ├── data_utils.py                  # 数据处理：分词、数据加载、词表、序列化、DataLoader、Word2Vec训练
│   ├── model.py                       # 模型定义：TextCNN、SelfAttention、TextLSTM
│   ├── train_utils.py                 # 训练框架：EarlyStopping、train_epoch、evaluate、train_model
│   ├── train.py                       # ★ 训练主脚本（可独立运行）
│   ├── evaluate.py                    # ★ 独立评估脚本（可独立运行）
│   └── inference.py                   # ★ 独立推理脚本（可独立运行，含交互模式）
├── aclImdb_v1/                        # 数据集目录（需自行下载解压）
│   └── aclImdb/
│       ├── train/pos/  (12500 .txt)
│       ├── train/neg/  (12500 .txt)
│       ├── train/unsup/ (50000 .txt)  ★ 优化版Word2Vec训练使用
│       ├── test/pos/   (12500 .txt)
│       └── test/neg/   (12500 .txt)
└── results/                           # 运行输出目录（运行后生成）
    ├── vocab.pkl
    ├── w2v_model.gensim
    ├── cnn_model.pth / lstm_model.pth
    ├── tfidf_vectorizer.pkl / tfidf_lr_model.pkl
    ├── loss_curve.png
    ├── error_cases.txt
    └── results.txt
```

---

## src/ 模块文件详解

以下为 `src/` 目录下每个文件的功能说明，保证实验完全可复现。

### config.py — 全局配置与随机种子

- **功能**：定义全局配置字典 `CONFIG`（超参数集）、`DATA_ROOT` 路径、随机种子固定函数 `set_seed()`、目录创建函数 `ensure_dir()`、NLTK资源下载函数 `download_nltk_resources()`、自定义停用词表构建
- **CONFIG 包含**：词表大小 (25000)、序列长度 (384)、词向量维度 (300)、Word2Vec超参 (window=8, min_count=3, epochs=20)、CNN/LSTM结构参数、训练参数 (batch_size=64, lr=0.001, weight_decay=1e-4, label_smoothing=0.05, epochs=20, patience=5, grad_clip=3.0)、设备选择 (cuda/cpu)、随机种子 (42)、输出目录 (./results)
- **DATA_ROOT**：指向 `aclImdb_v1/` 数据集的绝对路径，需用户按本机实际路径修改
- **停用词表**：使用NLTK英文停用词表，但保留否定词(not/no/nor)、程度副词(very/too/only)、转折词(but/yet/still)等对情感分析至关重要的词汇
- **被导入于**：train.py, evaluate.py, inference.py, data_utils.py, model.py, train_utils.py

### data_utils.py — 数据处理与Word2Vec训练

- **功能**：提供从数据加载到模型输入的全套数据处理管线
- **核心函数**：
  - `tokenize(text)` — 文本清洗分词：去HTML标签→正则清理→小写→NLTK分词→去纯符号→停用词过滤
  - `load_imdb_from_local(data_root)` — 从本地 `aclImdb_v1/` 四子目录读取全部 .txt 影评，返回 (train_texts, train_labels, test_texts, test_labels)
  - `load_unsup_texts(data_root)` — 加载 `train/unsup/` 下约50000条无标签影评，用于扩充Word2Vec训练语料
  - `build_vocab(texts, max_size, min_freq)` — 词频统计并构建 word2idx/idx2word 映射（0=PAD, 1=UNK）
  - `texts_to_sequences(texts, word2idx, max_len)` — 文本→整数序列，使用头+尾联合截断策略（前max_len//2 + 后max_len//2），短序列前填充PAD
  - `IMDBDataset` — PyTorch Dataset 类，封装序列和标签
  - `create_dataloaders(train_seqs, train_labels, test_seqs, test_labels, batch_size)` — 创建 train/val/test DataLoader，训练集按80/20划分
  - `train_word2vec(texts, config, unsup_texts)` — 使用Gensim训练Skip-Gram+负采样Word2Vec，合并训练集和无监督数据
  - `build_embedding_matrix(word2idx, w2v_model, embed_dim)` — 构建PyTorch Embedding预训练权重矩阵，命中词用Word2Vec向量，未命中词用U(-0.25,0.25)随机初始化
- **被导入于**：train.py, evaluate.py, inference.py

### model.py — 模型定义

- **功能**：定义三个深度学习模块
- **TextCNN** (Kim 2014)：
  - Embedding层：支持加载预训练Word2Vec词向量（可微调）或从零随机初始化
  - 卷积层：3组并行Conv2d，kernel_size=(3,4,5)，各128个滤波器，单通道文本输入
  - 池化层：每组卷积后经ReLU → 1-MaxPooling → squeeze
  - 分类头：拼接3组特征 → BatchNorm1d → Dropout(0.3) → Linear → logits
  - 参数量：约8M（含Embedding 7.5M）
- **SelfAttention** (Vaswani 2017)：
  - 缩放点积自注意力：Q/K/V线性投影 → softmax(QK^T/√d_k) × V
  - 支持PAD mask（对填充位置施加-1e9偏置）
  - 返回加权上下文向量和注意力权重分布
  - Q/K投影至attention_dim(128)，V投影保持input_dim以匹配下游BatchNorm维度
- **TextLSTM** (核心模型)：
  - Embedding层：同TextCNN，支持预训练词向量加载
  - 编码器：2层双向LSTM（hidden_dim=256, batch_first=True）
  - 注意力层：SelfAttention加权聚合所有时间步输出（替代原仅取末层最后时刻隐状态）
  - 分类头：BatchNorm1d → Dropout(0.3) → Linear → logits
  - 参数量：约11M（Embedding 7.5M + LSTM 3.3M + Attention 0.3M）
- **被导入于**：train.py, evaluate.py, inference.py

### train_utils.py — 训练与评估框架

- **功能**：提供通用训练循环和评估组件，不独立运行
- **核心组件**：
  - `EarlyStopping` 类 — 监控验证损失，patience轮内无改善即触发早停
  - `train_epoch(model, dataloader, optimizer, criterion, device, clip, label_smoothing)` — 单个训练轮次：前向→损失计算（支持标签平滑，smoothing=0.05时标签从硬0/1软化为0.025/0.975）→反向→梯度裁剪→参数更新，返回平均损失和准确率
  - `evaluate(model, dataloader, criterion, device)` — 评估函数，返回 (avg_loss, accuracy, f1(binary), precision, recall, all_preds, all_labels)
  - `train_model(model, train_loader, val_loader, device, config, model_name, save_best_path)` — 完整训练循环：计算pos_weight类别权重→AdamW优化器(weight_decay=1e-4)→ReduceLROnPlateau(patience=3)→逐epoch训练/验证→最佳模型保存至磁盘→早停判断→训练结束后恢复最佳权重
- **被导入于**：train.py, evaluate.py

### train.py — 训练主脚本（可独立运行）

- **功能**：完整实验主入口，串联所有阶段
- **启动**：`python -m src.train` 或 `python src/train.py`
- **执行流程**：set_seed(42) → 本地数据加载 → unsup数据加载 → 词表构建 → 序列化 → Word2Vec训练 → DataLoader → TF-IDF+LR基线 → CNN训练 → LSTM训练 → 训练曲线可视化 → 错误案例分析 → 结果汇总
- **包含函数**：
  - `train_tfidf_lr()` — TF-IDF+LR训练与评估，持久化vectorizer和分类器
  - `plot_training_curves()` — 绘制2×2子图（CNN/LSTM的Loss和Acc曲线），保存loss_curve.png
  - `analyze_lstm_errors()` — 错误案例分析，区分FP/FN，按置信度排序，输出影评片段和混淆情感词
  - `generate_results_summary()` — 生成results.txt，包含指标表格+原理分析+17项优化清单+超参数+数据集说明+AI工具声明+参考文献
  - `main()` — 主流程编排，输出总耗时和最终模型对比
- **输出目录**：`./results/`（共9个文件）

### evaluate.py — 独立评估脚本（可独立运行）

- **功能**：加载已训练的模型权重，在测试集上独立评估，不执行训练
- **前置条件**：`results/` 目录下需已有 vocab.pkl, w2v_model.gensim, cnn_model.pth, lstm_model.pth
- **启动**：`python -m src.evaluate` 或 `python src/evaluate.py`
- **包含函数**：
  - `load_artifacts(results_dir)` — 加载词表、Word2Vec模型、测试数据、DataLoader
  - `load_model(model_type, word2idx, embedding_matrix, device, results_dir)` — 构建模型并加载最佳权重
  - `main()` — 依次评估CNN和LSTM，输出对比表格
- **输出**：仅终端输出，不产生新文件

### inference.py — 独立推理脚本（可独立运行）

- **功能**：加载LSTM模型，提供交互式命令行推理 + 编程调用接口
- **前置条件**：`results/` 目录下需已有 vocab.pkl, w2v_model.gensim, lstm_model.pth
- **启动**：`python -m src.inference` 或 `python src/inference.py`
- **核心类 `SentimentPredictor`**：
  - `__init__(results_dir)` — 加载词表、Word2Vec、LSTM模型权重
  - `predict(text)` — 单条文本预测，返回 {"sentiment", "confidence", "positive_prob", "negative_prob"}
  - `predict_batch(texts)` — 批量预测，返回 dict 列表
  - `_texts_to_tensor(texts)` — 内部方法，文本→序列→Tensor
- **交互模式**：输入英文影评 → 输出情感类别+置信度+正负概率，输入 quit/exit/q 退出
- **输出**：仅终端输出，不产生新文件

---

## AI工具使用声明

本实验在完成过程中使用了 Claude (Anthropic) 作为编程辅助工具，具体使用范围包括：代码框架生成与调试辅助、文档润色、文献检索。所有代码与实验内容已经过人工审查，本人对全部提交内容的理解与正确性负责。
