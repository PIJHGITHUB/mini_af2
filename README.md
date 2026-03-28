# Mini AlphaFold2

AlphaFold2 蛋白质结构预测的极简教学实现，使用 ProteinNet CASP7 真实数据训练。

## 项目结构

```
mini_af2/
├── config.py              # 全局超参数（序列、模型维度、训练、数据配置）
├── dataset.py             # ProteinNet 数据下载、解析、Dataset/DataLoader
├── model/
│   ├── __init__.py        # 导出 MiniAlphaFold2 和损失函数
│   └── mini_af2.py        # 模型架构（Evoformer + Structure Module + 预测头 + 损失函数）
├── train.py               # 训练循环入口
├── mini_af2_best.pt       # 训练保存的最佳模型权重
├── proteinnet_data/       # 数据缓存（首次运行自动下载）
└── README.md
```

## 快速开始

```bash
pip install torch numpy
python train.py            # 首次运行会自动下载 ProteinNet CASP7（约 30 MB）
```

设备自动检测优先级：CUDA GPU > MPS（Mac M 系列芯片）> CPU。

## 覆盖的 AlphaFold2 核心概念

**Evoformer（序列-结构协同进化）：**
MSA 行注意力（带 Pair 偏置）、MSA 列注意力、Outer Product Mean（MSA → Pair 融合）、三角乘法更新（outgoing / incoming，Algorithm 11/12）、三角注意力（outgoing / incoming，Algorithm 13/14）、Pair 过渡层。

**Structure Module（3D 结构预测）：**
Invariant Point Attention（IPA）、6D 旋转参数化（Gram-Schmidt）、刚体变换迭代更新。

**损失函数：**
FAPE（Frame Aligned Point Error，对旋转/平移等变）+ Distogram（残基间距离分布交叉熵）。

## 模型规模对比

| 参数 | Mini AF2 | 真实 AF2 |
|------|----------|----------|
| MSA 维度 c_m | 64 | 256 |
| Pair 维度 c_z | 32 | 128 |
| Evoformer 层数 | 4 | 48 |
| Structure Module 层数 | 4 | 8 |
| 总参数量 | ~572K | ~93M |

## 数据

使用 ProteinNet CASP7 的 training_30 子集（30% 序列同一性去冗余），默认加载 500 条蛋白质，序列长度 40~128 残基。数据首次运行时自动下载到 `proteinnet_data/` 目录。

## 各文件说明

**config.py** — 所有超参数集中管理。修改模型维度、训练轮数、学习率等只需编辑此文件。

**dataset.py** — 从 ProteinNet 文本格式解析氨基酸序列、PSSM、骨架原子坐标（N/Ca/C），计算局部坐标系（旋转矩阵 + 平移），从 PSSM 采样伪 MSA，封装为 PyTorch Dataset 和 DataLoader。

**model/mini_af2.py** — 完整模型定义，包含 InputEmbedder、Evoformer（6 个子模块）、InvariantPointAttention、StructureModuleBlock、DistogramHead、MiniAlphaFold2 顶层模型，以及 FAPE / Distogram / total_loss 损失函数。

**train.py** — 训练入口脚本，包含数据加载、模型初始化、Adam 优化器 + 余弦退火调度、训练/验证循环、最佳模型保存。

## 加载训练好的模型

```python
from config import C
from model import MiniAlphaFold2
import torch

model = MiniAlphaFold2().to(C.device)
model.load_state_dict(torch.load("mini_af2_best.pt", map_location=C.device))
model.eval()
```
