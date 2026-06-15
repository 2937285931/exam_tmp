#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练与评估框架：EarlyStopping、train_epoch、evaluate、train_model。
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


class EarlyStopping:
    """
    早停机制：当验证损失在 patience 轮内没有改善（下降超过 delta）时触发停止。
    避免模型在训练集上过拟合。
    """

    def __init__(self, patience: int = 3, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_epoch(model, dataloader, optimizer, criterion, device, clip: float, label_smoothing: float = 0.0):
    """
    执行一个训练轮次。

    参数:
      model: PyTorch 模型
      dataloader: 训练数据加载器
      optimizer: 优化器
      criterion: 损失函数（BCEWithLogitsLoss）
      device: 计算设备
      clip: 梯度裁剪阈值
      label_smoothing: 标签平滑因子（0表示不使用）
    返回:
      (平均损失, 准确率)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences)

        # 【改进点10】标签平滑：将硬标签(0/1)软化为(smoothing/2, 1-smoothing/2)
        # 原代码: loss = criterion(logits, labels)
        # 新代码: 对labels做平滑后计算损失
        if label_smoothing > 0:
            smooth_labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
            loss = criterion(logits, smooth_labels)
        else:
            loss = criterion(logits, labels)

        loss.backward()

        # 【改进点16】梯度裁剪，防止梯度爆炸（阈值5.0→3.0，更积极抑制）
        # 原代码: nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item() * sequences.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """
    评估模型性能。

    【改进点17】F1 Score使用binary平均替代macro平均
      原代码: f1_score(..., average='macro')
      新代码: f1_score(..., average='binary')
      对于二分类任务，binary F1更直接反映分类器性能

    返回:
      avg_loss, accuracy, f1, precision, recall, all_preds, all_labels
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            logits = model(sequences)
            loss = criterion(logits, labels)

            total_loss += loss.item() * sequences.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = correct / total
    # 【改进点17】原代码: f1_score(..., average='macro')
    # 新代码: 二分类任务使用binary平均
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)

    return total_loss / total, acc, f1, prec, rec, all_preds, all_labels


def train_model(model, train_loader, val_loader, device, config, model_name: str,
                save_best_path: str = None):
    """
    通用训练循环：含早停、学习率衰减、模型保存、训练历史记录。

    【改进点06】Adam → AdamW + weight_decay (L2正则化)
    【改进点07】ReduceLROnPlateau patience=1→3，学习率衰减更稳定
    【改进点08】BCEWithLogitsLoss增加pos_weight类别权重平衡
    【改进点10】训练时应用标签平滑

    参数:
      model: 待训练的 PyTorch 模型
      train_loader / val_loader: 训练/验证 DataLoader
      device: 计算设备
      config: 全局配置
      model_name: 模型名称（用于保存文件名和日志，如 "cnn" / "lstm"）
      save_best_path: 最佳模型权重保存路径（可选，如 "./results/cnn_model.pth"）
    返回:
      (训练好的模型, history_dict)
    """
    # 【改进点08】计算正样本权重用于类别平衡
    # 原代码: criterion = nn.BCEWithLogitsLoss()
    # 新代码: 计算训练集中正负样本比例，设置pos_weight
    train_labels_list = []
    for _, labels in train_loader:
        train_labels_list.extend(labels.numpy().tolist())
    n_pos = sum(train_labels_list)
    n_neg = len(train_labels_list) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  正样本数: {n_pos}, 负样本数: {n_neg}, pos_weight: {pos_weight.item():.3f}")

    # 【改进点06】原代码: optim.Adam(model.parameters(), lr=config["learning_rate"])
    # 新代码: AdamW优化器，weight_decay实现L2正则化解耦
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    # 【改进点07】原代码: ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    # 新代码: patience=3，给模型更多时间在当前学习率下探索
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    early_stopping = EarlyStopping(patience=config["patience"])

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_val_acc = 0.0
    best_model_state = None  # 内存中保存最佳模型状态，避免读写模型文件

    print(f"\n{'='*60}")
    print(f"训练 {model_name.upper()} 模型")
    print(f"{'='*60}")

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()

        # 【改进点10】传入label_smoothing参数
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            config["grad_clip"], config["label_smoothing"]
        )
        val_loss, val_acc, val_f1, _, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - epoch_start
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:2d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s"
        )

        # 当验证准确率提升时在内存中保存最佳模型状态
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # 同时持久化到磁盘，供 evaluate.py / inference.py 独立加载
            if save_best_path is not None:
                torch.save(best_model_state, save_best_path)
            print(f"  >>> 更新最佳模型 (Val Acc: {val_acc:.4f})")

        if early_stopping(val_loss):
            print(f"  早停触发! 最佳验证准确率: {best_val_acc:.4f}")
            break

    # 训练结束后从内存恢复最佳权重（无需读取模型文件）
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"  {model_name.upper()} 训练完成，最佳验证 Acc: {best_val_acc:.4f}")
    return model, history
