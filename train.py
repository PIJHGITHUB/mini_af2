"""
train.py — Mini AlphaFold2 训练入口
====================================
运行方式：
    python train.py

功能：
  1. 自动下载并解析 ProteinNet CASP7 数据（首次运行约 2 分钟）
  2. 划分训练集 / 验证集（9:1）
  3. 训练 n_epochs 轮，余弦退火调度学习率
  4. 每轮结束后在验证集评估，保存最佳模型到 mini_af2_best.pt
"""

import torch

from config import C
from dataset import build_dataloaders
from model import MiniAlphaFold2, total_loss


def train():
    """完整训练流程。"""
    # ── 设备和基本信息 ──────────────────────────────────────────────
    print("=" * 65)
    print(" Mini AlphaFold2 — ProteinNet 真实数据训练")
    print("=" * 65)
    print(f"  设备   : {C.device}"
          + (" ← Mac MPS 加速 🚀" if C.device == "mps" else ""))
    print(f"  序列长度: L={C.L}（截断/填充），MSA 伪序列: {C.N_msa} 条")
    print(f"  模型维度: c_m={C.c_m}, c_z={C.c_z}, c_s={C.c_s}")
    print(f"  训练   : {C.n_epochs} epochs, batch={C.batch}, lr={C.lr}")
    print()

    # ── 1. 数据准备 ─────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(C.data_dir)

    # ── 2. 模型 ─────────────────────────────────────────────────────
    model        = MiniAlphaFold2().to(C.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量 : {total_params:,}（完整 AF2 约 93,000,000，缩减 "
          f"{93_000_000 // total_params}x）")
    print()

    # ── 3. 优化器 + 学习率调度 ──────────────────────────────────────
    # Adam：真实 AF2 用带 warmup 的 Adam，这里为简洁省略 warmup
    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr)
    # 余弦退火：lr 从 C.lr 渐减至 C.lr * 0.1，防止后期震荡
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=C.n_epochs, eta_min=C.lr * 0.1
    )

    # ── 4. 训练主循环 ────────────────────────────────────────────────
    best_val_loss = float("inf")
    save_path     = "mini_af2_best.pt"

    header = (f"{'Epoch':>6}  {'训练损失':>10}  {'验证损失':>10}  "
              f"{'FAPE':>8}  {'Distogram':>10}  {'lr':>8}")
    print(header)
    print("-" * 65)

    for epoch in range(1, C.n_epochs + 1):

        # ── 训练阶段 ────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            # 把所有张量移动到目标设备（CPU / MPS / CUDA）
            batch = {k: v.to(C.device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs          = model(batch)
            loss, _          = total_loss(outputs, batch)
            loss.backward()
            # 梯度裁剪：防止早期训练时梯度爆炸（AF2 用 max_norm=0.1）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum  += loss.item()

        train_loss_avg = train_loss_sum / len(train_loader)

        # ── 验证阶段 ────────────────────────────────────────────────
        model.eval()
        val_loss_sum  = 0.0
        fape_sum      = 0.0
        disto_sum     = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch          = {k: v.to(C.device) for k, v in batch.items()}
                outputs        = model(batch)
                loss, breakdown = total_loss(outputs, batch)
                val_loss_sum  += loss.item()
                fape_sum      += breakdown["fape"]
                disto_sum     += breakdown["distogram"]

        n_val          = len(val_loader)
        val_loss_avg   = val_loss_sum / n_val
        fape_avg       = fape_sum     / n_val
        disto_avg      = disto_sum    / n_val
        current_lr     = scheduler.get_last_lr()[0]

        scheduler.step()

        # ── 保存最佳模型 ─────────────────────────────────────────────
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), save_path)
            mark = " ★"   # 标记最佳
        else:
            mark = ""

        print(f"  {epoch:4d}  {train_loss_avg:10.4f}  {val_loss_avg:10.4f}  "
              f"{fape_avg:8.4f}  {disto_avg:10.4f}  {current_lr:8.2e}{mark}")

    print("-" * 65)
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型已保存至: {save_path}")
    print()
    print("期望的训练效果（真实 ProteinNet 数据）：")
    print("  · Distogram 损失从初始 ~4.16（ln64）下降到 ~3.0~3.5")
    print("  · FAPE 损失缓慢下降，但极简模型（2层）很难<5Å")
    print("  · 扩大 c_m/c_z、增加 n_evo/n_sm 可显著提升效果")
    print("  · 加载最佳模型: model.load_state_dict(torch.load('mini_af2_best.pt'))")


if __name__ == "__main__":
    train()
