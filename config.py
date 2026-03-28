"""
config.py — Mini AlphaFold2 全局超参数配置
==========================================
所有超参数集中管理，供 dataset.py、model/、train.py 共同引用。

使用方式：
    from config import Config, C
"""

import torch


def _detect_device() -> str:
    """自动检测最优计算设备：CUDA > MPS（Mac）> CPU"""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"   # Mac M 系列芯片
    return "cpu"


class Config:
    # ── 序列与 MSA ────────────────────────────────────────────────
    L        = 128    # 截断后的固定序列长度（超过则截断，不足则填充）
    N_msa    = 8      # 从 PSSM 采样的伪 MSA 条数
    n_aa     = 21     # 氨基酸种类（20种标准 + X未知）

    # ── 模型维度（保持小，确保 Mac CPU 可运行）────────────────────
    c_m      = 64     # MSA 表示维度（真实 AF2：256）
    c_z      = 32     # Pair 表示维度（真实 AF2：128）
    c_s      = 64     # 单链（single）表示维度（真实 AF2：384）
    c_ipa    = 16     # IPA 中间维度

    # ── 层数 ─────────────────────────────────────────────────────
    n_evo    = 4      # Evoformer 层数（真实：48）
    n_sm     = 4      # Structure Module 层数（真实：8）
    n_heads  = 4      # 注意力头数

    # ── 训练超参数 ────────────────────────────────────────────────
    batch    = 4      # 每批蛋白质数（Mac CPU 建议 4）
    lr       = 1e-3   # 学习率（Adam）
    n_epochs = 20     # 训练轮数

    # ── 设备（自动检测）──────────────────────────────────────────
    device   = _detect_device()

    # ── 数据配置（ProteinNet CASP7）──────────────────────────────
    data_dir     = "./proteinnet_data"  # 数据下载/缓存目录（可自定义）
    max_proteins = 500                  # 最多加载的蛋白质数
    min_len      = 40                   # 过滤：序列最短长度（太短无意义）
    max_len      = 128                  # 过滤：序列最长长度（超过则截断到 L）
    val_ratio    = 0.1                  # 验证集比例（10%）


C = Config()
