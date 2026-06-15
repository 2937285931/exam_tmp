# 如何提交PDF和代码

## 总览

将考核的 GitHub 仓库 Fork，按照以下结构化流程完成提交。

---

## 第一步：Fork 仓库

1. 打开考核仓库的 GitHub 页面

2. 点击右上角 **Fork** 按钮，将仓库 Fork 到你的个人 GitHub 账号下

3. `git clone` 你 Fork 后的仓库到本地：
   
   ```bash
   git clone https://github.com/<你的用户名>/AI-Assessment.git
   cd AI-Assessment
   ```

---

## 第二步：撰写 PDF 报告

- 报告内容涵盖 **文献综述（论文摘要）** 和 **实验分析**
- 三个方向任选其一，在 PDF 中写明所选方向
- PDF 控制在 **5-8 页**
- 文件命名：`<方向>_<姓名>.pdf`（例如 `NLP_张三.pdf`）
- 将 PDF 放到仓库**根目录**下

---

## 第三步：整理代码文件

根据你选的方向，将代码放到对应文件夹下。

### 方向选择

| 方向              | 文件夹           | 分数构成                |
| --------------- | ------------- | ------------------- |
| NLP（自然语言处理）     | `NLP/`        | 论文综述 40% + 代码实现 60% |
| MultiModal（多模态） | `MultiModal/` | 论文综述 40% + 代码实现 60% |
| CV（计算机视觉）       | `CV/`         | 论文综述 40% + 代码实现 60% |

### 代码结构要求

每个方向的文件夹已预设 `src/` 和 `results/` 子目录：

```
<方向>/
├── src/                 # 源代码
│   ├── train.py         # 训练脚本
│   ├── evaluate.py      # 评估脚本
│   └── inference.py     # 推理/预测脚本
├── results/             # 实验结果
│   ├── loss_curve.png   # 损失曲线
│   └── (其他图表/结果文件)
└── README.md            # （可选）运行说明
```

### 评分细则（代码部分）

| 维度       | 权重  | 说明                        |
| -------- | --- | ------------------------- |
| **代码质量** | 30% | 结构清晰，有注释，可复现              |
| **实验结果** | 30% | Accuracy / F1-Score 等指标   |
| **创新性**  | 30% | 尝试不同方法 / 对比实验 / 分析        |
| **拓展性**  | 10% | 额外尝试（如 Web 界面、Prompt 优化等） |

### 所有代码文件必须非空

- `train.py`、`evaluate.py`、`inference.py` **必须包含有效代码**，不可留空
- 如果某个文件不适用（例如 MultiModal 方向使用预训练模型推理，无需训练），则应在文件内注明原因并保留分析代码

---

## 第四步：更新 README.md（可选但推荐）

在仓库根目录的 `README.md` 中写明：

- 所选方向
- 运行环境依赖（Python 版本、PyTorch 版本等）
- 如何运行代码（train → evaluate → inference）
- 最终效果与关键结论

---

## 第五步：提交到 GitHub

```bash
git add .
git commit -m "提交 AI 考核：<方向>"
git push origin main
```

---

## 第六步：发送 Fork 仓库给测试者

将你的 Fork 仓库 URL 填到腾讯文档表格，格式：

```
https://github.com/<你的用户名>/AI-Assessment
```

提交链接：腾讯文档链接：【腾讯文档】考核提交汇总表格
https://docs.qq.com/sheet/DVWxPUnhTUnhFTHhX?tab=BB08J2

---

## 最终仓库结构示例

```
AI-Assessment/
├── NLP_张三.pdf              # ← 你的 PDF 报告（根目录）
├── Assessment.pdf            # 原考核题目（保留不动）
├── README.md                 # 运行说明
├── submission_format.md      # 本文件
├── CV/                       # 未选方向无需修改
├── MultiModal/               # 未选方向无需修改
└── NLP/                      # ← 选中的方向
    ├── src/
    │   ├── train.py          # 训练代码
    │   ├── evaluate.py       # 评估代码
    │   └── inference.py      # 推理代码
    └── results/
        ├── loss_curve.png    # 损失曲线图
        ├── confusion_matrix.png  # 混淆矩阵
        └── ...               # 其他结果文件
```
