"""
mini_af2.py — AlphaFold2 训练流程极简实现（ProteinNet 真实数据版）
================================================================
约 750 行，只需 torch + numpy（数据下载仅需标准库）。

覆盖的核心概念：
  1. 真实数据加载：ProteinNet CASP7（首次运行自动下载，约 30 MB）
       - 氨基酸序列 → 索引编码
       - PSSM（21×L）→ 采样伪 MSA
       - N/Cα/C 骨架坐标 → 旋转矩阵（局部坐标系）
  2. 输入表示：序列 + MSA + 相对位置编码 → (m, z)
  3. Evoformer（2层简化版）：
       - MSA 行注意力（带 Pair 偏置）
       - MSA 列注意力
       - Outer Product Mean（MSA → Pair 融合）
       - 三角乘法更新（outgoing + incoming，Algorithm 11/12）
       - 三角注意力（outgoing + incoming，Algorithm 13/14）
       - Pair 过渡层
  4. Structure Module（2层简化版）：
       - Invariant Point Attention（IPA）
       - 刚体变换（旋转矩阵 + 平移）
  5. 损失函数：FAPE + Distogram
  6. 完整训练循环：DataLoader + 验证集 + 余弦退火调度 + 最佳模型保存

运行（Mac 本地）：
    pip install torch numpy        # 仅两个依赖
    python mini_af2.py             # 首次运行自动下载 ProteinNet 数据

设备优先级（自动检测）：
    CUDA GPU  → 最快
    MPS（Mac M 系列芯片）→ 比纯 CPU 快 3~5x，推荐 M1/M2/M3 用户
    CPU       → 兜底，~500 条蛋白质、20 epoch 约 30~60 分钟
"""

import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ════════════════════════════════════════════════════════════════════
# 0. 全局超参数 + 数据加载（从 dataset.py 导入）
# ════════════════════════════════════════════════════════════════════
from dataset import Config, C, AA_VOCAB, AA_TO_IDX, build_dataloaders

# ════════════════════════════════════════════════════════════════════
# 2. 输入嵌入层
# ════════════════════════════════════════════════════════════════════

class InputEmbedder(nn.Module):
    """
    将原始序列/MSA 嵌入为连续向量表示。

    真实 AF2：
      - target_feat (one-hot 氨基酸) → pair representation
      - msa_feat (MSA one-hot + deletion) → msa representation
      - 相对位置编码 → pair representation
    """
    def __init__(self):
        super().__init__()
        # MSA 嵌入：氨基酸 one-hot → c_m 维向量
        self.msa_emb = nn.Embedding(C.n_aa + 1, C.c_m)  # +1 for padding

        # 序列嵌入（用于 Pair 表示的初始化）
        self.seq_emb = nn.Embedding(C.n_aa + 1, C.c_z // 2)

        # 相对位置编码（残基 i 和 j 的相对距离 → pair 特征）
        # 真实 AF2 用 32 个 bin，这里简化为线性映射
        self.relpos_emb = nn.Embedding(65, C.c_z // 2)  # 64 个相对位置 bin + 1
        self.relpos_proj = nn.Linear(C.c_z // 2, C.c_z)

    def forward(self, aatype, msa):
        """
        输入：
          aatype: (B, L)     氨基酸类型
          msa:    (B, N, L)  MSA 矩阵
        输出：
          m: (B, N, L, c_m)  MSA 表示
          z: (B, L, L, c_z)  Pair 表示
        """
        B, N, L = msa.shape

        # ── MSA 表示 m ──────────────────────────────────────────────
        # 每个 MSA 位置嵌入为 c_m 维向量
        m = self.msa_emb(msa)  # (B, N, L, c_m)

        # ── Pair 表示 z ──────────────────────────────────────────────
        # 残基 i 和 j 的序列特征外积
        seq_feat = self.seq_emb(aatype)  # (B, L, c_z//2)
        # 拼接 i 和 j 的特征：z[i,j] = [seq_i, seq_j]
        a = seq_feat.unsqueeze(2)  # (B, L, 1, c_z//2) 代表 i 的特征
        zi = a.expand(B, L, L, -1)  # (B, L, L, c_z//2)
        zj = seq_feat.unsqueeze(1).expand(B, L, L, -1)  # (B, L, L, c_z//2)
        z = torch.cat([zi, zj], dim=-1)  # (B, L, L, c_z)

        # 加入相对位置编码：第 i 和第 j 残基的相对距离
        i_idx = torch.arange(L, device=msa.device)
        j_idx = torch.arange(L, device=msa.device)
        # 相对距离 clip 到 [-32, 32]，映射到 [0, 64]
        rel = (i_idx.unsqueeze(1) - j_idx.unsqueeze(0)).clamp(-32, 32) + 32  # (L, L)
        rel_feat = self.relpos_emb(rel)  # (L, L, c_z//2)
        # 用线性层把 c_z//2 映射到 c_z 并加到 z
        z = z + self.relpos_proj(rel_feat.unsqueeze(0).expand(B, -1, -1, -1))

        return m, z


# ════════════════════════════════════════════════════════════════════
# 3. Evoformer 模块
# ════════════════════════════════════════════════════════════════════

class MSARowAttentionWithPairBias(nn.Module):
    """
    MSA 行注意力（Row-wise Self-Attention with Pair Bias）

    对 MSA 的每一行（序列）做自注意力，同时加上 pair 表示作为偏置。
    直觉：残基 j 对残基 i 的重要性，除了看 MSA 本身，
         还参考 pair 表示里 i-j 的协同进化信号。
    """
    def __init__(self):
        super().__init__()
        self.norm_m = nn.LayerNorm(C.c_m)
        self.norm_z = nn.LayerNorm(C.c_z)
        # 生成 pair bias（c_z → n_heads）
        self.pair_bias = nn.Linear(C.c_z, C.n_heads, bias=False)
        self.attn = nn.MultiheadAttention(C.c_m, C.n_heads, batch_first=True)

    def forward(self, m, z):
        """
        m: (B, N, L, c_m)  MSA 表示
        z: (B, L, L, c_z)  Pair 表示（提供注意力偏置）
        """
        B, N, L, _ = m.shape

        # Pair bias: (B, L, L, n_heads) → (B*n_heads, L, L)
        bias = self.pair_bias(self.norm_z(z))           # (B, L, L, n_heads)
        bias = bias.permute(0, 3, 1, 2)                  # (B, n_heads, L, L)
        bias = bias.unsqueeze(1).expand(-1, N, -1, -1, -1)
        bias = bias.reshape(B * N * C.n_heads, L, L)

        # 对每行（序列）做自注意力
        m_norm = self.norm_m(m)
        m_flat = m_norm.reshape(B * N, L, C.c_m)

        # 加入 pair bias（作为 attention_mask）
        out, _ = self.attn(m_flat, m_flat, m_flat,
                           attn_mask=bias)        
        # out, _ = self.attn(m_flat, m_flat, m_flat,
        #                    attn_mask=bias.mean(0))        # 简化：对 batch 取均值
        out = out.reshape(B, N, L, C.c_m)
        return m + out  # 残差连接


class MSAColumnAttention(nn.Module):
    """
    MSA 列注意力（Column-wise Self-Attention）

    对 MSA 的每一列（位置）做自注意力。
    直觉：让同一个位置的不同同源序列互相交流，融合进化信息。
    """
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(C.c_m)
        self.attn = nn.MultiheadAttention(C.c_m, C.n_heads, batch_first=True)

    def forward(self, m):
        """m: (B, N, L, c_m)"""
        B, N, L, _ = m.shape

        m_norm = self.norm(m)
        # 转置：把列变成 "序列"维度来做注意力
        m_t = m_norm.permute(0, 2, 1, 3).reshape(B * L, N, C.c_m)
        out, _ = self.attn(m_t, m_t, m_t)
        out = out.reshape(B, L, N, C.c_m).permute(0, 2, 1, 3)
        return m + out


class OuterProductMean(nn.Module):
    """
    外积均值（Outer Product Mean）：MSA → Pair

    直觉：对所有 MSA 序列，计算位置 i 和位置 j 的特征外积，
         然后在 N 个序列上取平均，得到残基对的协同进化特征。
    这是 MSA 信息流入 Pair 表示的唯一通道。
    """
    def __init__(self, c=8):
        super().__init__()
        self.norm = nn.LayerNorm(C.c_m)
        # 把 c_m 映射到低维 c，再做外积，控制计算量
        self.proj_i = nn.Linear(C.c_m, c)
        self.proj_j = nn.Linear(C.c_m, c)
        self.out    = nn.Linear(c * c, C.c_z)

    def forward(self, m):
        """m: (B, N, L, c_m) → (B, L, L, c_z)"""
        m = self.norm(m)
        a = self.proj_i(m)  # (B, N, L, c)
        b = self.proj_j(m)  # (B, N, L, c)

        # 外积：对每对 (i, j)，计算 a[:,:,i,:] ⊗ b[:,:,j,:]
        # 然后在 N 上求平均
        # 用 einsum 实现：'bnid,bnjd->bijdc' 然后 reshape
        # outer = torch.einsum('bnid,bnjd->bijn', a, b) / C.N_msa
        # 这里用简化版：直接 einsum 得到 (B, L, L, c*c)
        a2 = a.mean(dim=1)  # (B, L, c)
        b2 = b.mean(dim=1)  # (B, L, c)
        outer2 = torch.einsum('bic,bjd->bijcd', a2, b2)  # (B, L, L, c, c)
        B, L, _, c, _ = outer2.shape
        outer2 = outer2.reshape(B, L, L, c * c)
        return self.out(outer2)  # (B, L, L, c_z)


class TriangleMultiplicativeUpdate(nn.Module):
    """
    三角乘法更新（Triangle Multiplicative Update）

    直觉：对残基对 (i,j) 的表示，融合三角形 (i,k)*(k,j) 的信息。
    有两种：outgoing（共享行 i）和 incoming（共享列 j）。
    这是 AlphaFold2 的核心创新之一。
    """
    def __init__(self, outgoing=True):
        super().__init__()
        self.outgoing = outgoing  # 决定是使用outgoing 还是 incoming 模式
        self.norm = nn.LayerNorm(C.c_z)
        c = C.c_z
        self.proj_a = nn.Linear(c, c)
        self.proj_b = nn.Linear(c, c)
        self.gate_a = nn.Linear(c, c)
        self.gate_b = nn.Linear(c, c)
        self.out    = nn.Linear(c, c)
        self.gate   = nn.Linear(c, c)
        self.norm2  = nn.LayerNorm(c)

    def forward(self, z):
        """z: (B, L, L, c_z)"""
        z_norm = self.norm(z)
        a = torch.sigmoid(self.gate_a(z_norm)) * self.proj_a(z_norm)
        b = torch.sigmoid(self.gate_b(z_norm)) * self.proj_b(z_norm)

        if self.outgoing:
            # z[i,j] += sum_k a[i,k] * b[k,j]
            # 即：矩阵乘法 (L,L,c) × (L,L,c) 在 L 维上收缩
            x = torch.einsum('bikc,bjkc->bijc', a, b)
        else:
            # z[i,j] += sum_k a[k,i] * b[k,j]
            x = torch.einsum('bkic,bkjc->bijc', a, b)

        x = self.norm2(x)
        x = torch.sigmoid(self.gate(z_norm)) * self.out(x)
        return z + x

    
class PairTransition(nn.Module):
    """Pair 表示的 FFN 层（位置逐点的两层 MLP）"""
    def __init__(self, n=4):
        super().__init__()
        self.norm = nn.LayerNorm(C.c_z)
        self.ff = nn.Sequential(
            nn.Linear(C.c_z, C.c_z * n),
            nn.ReLU(),
            nn.Linear(C.c_z * n, C.c_z),
        )

    def forward(self, z):
        return z + self.ff(self.norm(z))
    
class TriangleAttention(nn.Module):
    """
    三角注意力（Triangle Attention）— AF2 Algorithm 13 & 14

    几何直觉：
      蛋白质结构中，如果残基 i-j 之间有联系，i-k 之间有联系，
      那么 j-k 之间很可能也有联系（三角不等式约束）。
      三角注意力就是利用这一先验，在更新 z[i,j] 时，
      让它同时参考"三角形另外两条边"的信息。

    两种变体：
      outgoing=True  → 起始节点注意力（Algorithm 13）
                        更新 z[i,j]：在行 i 内对列 j 做注意力，
                        偏置来自"第三边" z[j,k]（j→k 的 pair 特征）

      outgoing=False → 终止节点注意力（Algorithm 14）
                        更新 z[i,j]：等价于先把矩阵转置，
                        用相同的起始节点逻辑处理，再转置回来
                        偏置来自"第三边" z[k,i]（k→i 的 pair 特征）

    参数说明：
      linear_bias : 把 c_z 维的 pair 特征映射为 n_heads 个标量偏置
      gate        : Sigmoid 门控，决定注意力输出有多少比例被保留
      attn        : 标准多头自注意力（PyTorch 内置）
    """
    def __init__(self, outgoing=True):
        super().__init__()
        self.outgoing    = outgoing                                       # True=起始节点, False=终止节点
        self.norm        = nn.LayerNorm(C.c_z)                           # 输入 LayerNorm
        self.linear_bias = nn.Linear(C.c_z, C.n_heads, bias=False)      # pair 特征 → 注意力偏置（每个 head 一个标量）
        self.gate        = nn.Linear(C.c_z, C.c_z)                      # 门控线性层
        self.attn        = nn.MultiheadAttention(C.c_z, C.n_heads,      # 多头自注意力
                                                 batch_first=True)

    def forward(self, z):
        """
        输入/输出：z: (B, L, L, c_z) — Pair 表示，输入输出形状相同

        计算流程（以 outgoing 为例）：
          1. 对 L×L 的 pair 矩阵的每一行 i，做跨列 j 的自注意力
          2. 注意力偏置由"对角线对边" z[b,j,k] 提供（第三边信息）
          3. 用 Sigmoid 门控缩放注意力输出
          4. 残差连接
        """
        B, L, _, _ = z.shape
        # ── 保存残差 ─────────────────────────────────────────────────
        # 必须在任何变换前保存，确保残差连接用的是原始输入
        residual = z

        # ── 终止节点（incoming）：通过转置复用起始节点代码路径 ──────
        # Algorithm 14 等价于：转置 z → 走 Algorithm 13 逻辑 → 再转置回来
        # 转置后 z[b, i, j] 变为原来的 z[b, j, i]，
        # 此时"行"变成了原来的"列"，注意力方向也随之反转
        if not self.outgoing:
            z = z.transpose(1, 2)              # (B, L, L, c_z)：行列互换

        # ── LayerNorm 归一化 ─────────────────────────────────────────
        z_norm = self.norm(z)                  # (B, L, L, c_z)
        
        # ── Triangle Bias（三角偏置）────────────────────────────────
        # bias[b, j, k, h] = linear(z_norm[b, j, k])
        # 语义：更新 z[b,i,j] 时，位置 j 对 k 的注意力权重
        #       受到"第三条边 j→k"（z[b,j,k]）的偏置影响
        # 重要性质：偏置只与 (b, j, k) 有关，与行 i 无关
        bias = self.linear_bias(z_norm)        # (B, L, L, n_heads)
        bias = bias.permute(0, 3, 1, 2)         # (B, n_heads, L, L)，方便后续广播
        bias = bias.unsqueeze(1).expand(-1, L, -1, -1, -1)  # (B, L, n_heads, L, L)，广播到每行 i
        bias = bias.reshape(B * L *C.n_heads, L, L)
        z_flat = z_norm.reshape(B * L, L, C.c_z)  # (B*L, L, c_z)，每行是一个"样本"
        attn_out=self.attn(z_flat, z_flat, z_flat, attn_mask=bias)[0]  # (B*L, L, c_z)
        attn_out=attn_out.reshape(B, L, L, C.c_z)  # (B, L, L, c_z)
        # ── 行方向自注意力（Row-wise Self-Attention）────────────────
        # 逐 batch 处理，每个蛋白质 b 使用自己的 (L, L) 偏置矩阵
        # 这样避免了 B*L*n_heads*L*L 的大型扩展（L=128 时约 1GB），
        # 同时保证 per-batch 偏置的正确性。
        # B 通常为 4，Python 循环开销可忽略。
        # attn_outputs = []
        # for b in range(B):
        #     # 对第 b 条蛋白质：attn_mask[j,k] = mean_head(bias[b,j,k,h])
        #     attn_bias_b = bias[b].mean(dim=-1)                  # (L, L) per-batch 偏置
        #     z_b         = z_norm[b]                             # (L, L, c_z)，每行是一个"样本"
        #     out_b, _    = self.attn(
        #         z_b, z_b, z_b,
        #         attn_mask=attn_bias_b                           # (L, L)，正确参数名（原 bug: att_mask）
        #     )
        #     attn_outputs.append(out_b)                          # (L, L, c_z)
        # attn_out = torch.stack(attn_outputs, dim=0)             # (B, L, L, c_z)
      

        # ── 门控（Gating）────────────────────────────────────────────
        # Sigmoid 门控：决定注意力结果中有多少比例被保留
        # 必须用 z_norm 而不是 z（z 在 incoming 分支已被转置，语义不同）
        gate     = torch.sigmoid(self.gate(z_norm))             # (B, L, L, c_z)，值域 (0,1)
        attn_out = gate * attn_out                              # 逐元素门控

        # ── 终止节点（incoming）：转置回原始方向 ─────────────────────
        # 恢复到与输入 z 相同的语义方向
        if not self.outgoing:
            attn_out = attn_out.transpose(1, 2)

        # ── 残差连接并返回 ───────────────────────────────────────────
        # 原bug：缺少 return 语句，导致函数返回 None，训练时会报 NoneType 错误
        return residual + attn_out                              # (B, L, L, c_z)
        


class EvoformerBlock(nn.Module):
    """
    一个完整的 Evoformer 块（AF2 Algorithm 6 简化版）

    数据流：
      MSA 表示 m (B,N,L,c_m)  ←→  Pair 表示 z (B,L,L,c_z)
      两者在整个 block 中相互更新、相互影响。

    更新顺序（严格遵循论文 Algorithm 6）：
    ┌─────────────────────────────────────────────────────────────┐
    │  MSA Stack（更新 MSA 表示 m）                               │
    │    Step 1. MSA 行注意力（带 Pair 偏置）                     │
    │            让同一序列内的残基互相通信，pair 表示提供偏置     │
    │    Step 2. MSA 列注意力                                     │
    │            让同一位置的不同同源序列互相通信                  │
    │    Step 3. MSA Transition（FFN）                            │
    │            逐位置的两层 MLP，增加非线性表达能力             │
    ├─────────────────────────────────────────────────────────────┤
    │  MSA → Pair 融合                                            │
    │    Step 4. Outer Product Mean                               │
    │            MSA 信息流入 pair 表示的唯一通道                 │
    ├─────────────────────────────────────────────────────────────┤
    │  Pair Stack（更新 Pair 表示 z）                             │
    │    Step 5. 三角乘法更新 outgoing （Algorithm 11）           │
    │            z[i,j] += Σ_k  a[i,k] * b[k,j]                 │
    │    Step 6. 三角乘法更新 incoming （Algorithm 12）           │
    │            z[i,j] += Σ_k  a[k,i] * b[k,j]                 │
    │    Step 7. 三角注意力 outgoing   （Algorithm 13）           │
    │            对每行 i，在列 j 方向做注意力，偏置来自 z[j,k]  │
    │    Step 8. 三角注意力 incoming   （Algorithm 14）           │
    │            对每列 j，在行 i 方向做注意力，偏置来自 z[k,i]  │
    │    Step 9. Pair Transition（FFN）                           │
    │            逐对位置的两层 MLP                               │
    └─────────────────────────────────────────────────────────────┘

    注意：三角乘法（Step5/6）和三角注意力（Step7/8）互补：
      - 三角乘法：计算效率高，O(L²c) 复杂度，全局信息传播
      - 三角注意力：更灵活的加权聚合，但内存更大，O(L³) 复杂度
    """
    def __init__(self):
        super().__init__()
        # ── MSA Stack 子模块 ──────────────────────────────────────────
        self.row_attn  = MSARowAttentionWithPairBias()          # Step 1：行注意力（带pair偏置）
        self.col_attn  = MSAColumnAttention()                   # Step 2：列注意力
        self.msa_ff    = nn.Sequential(                         # Step 3：MSA FFN 过渡层
            nn.LayerNorm(C.c_m),
            nn.Linear(C.c_m, C.c_m * 2),
            nn.ReLU(),
            nn.Linear(C.c_m * 2, C.c_m)
        )
        # ── MSA → Pair 融合 ───────────────────────────────────────────
        self.opm         = OuterProductMean()                   # Step 4：外积均值

        # ── Pair Stack 子模块 ─────────────────────────────────────────
        self.tri_mul_out  = TriangleMultiplicativeUpdate(outgoing=True)   # Step 5：三角乘法（outgoing）
        self.tri_mul_in   = TriangleMultiplicativeUpdate(outgoing=False)  # Step 6：三角乘法（incoming）
        self.tri_attn_out = TriangleAttention(outgoing=True)              # Step 7：三角注意力（outgoing，Algorithm 13）
        self.tri_attn_in  = TriangleAttention(outgoing=False)             # Step 8：三角注意力（incoming，Algorithm 14）
        self.pair_ff      = PairTransition()                              # Step 9：Pair FFN 过渡层

    def forward(self, m, z):
        """
        输入：
          m: (B, N, L, c_m) — MSA 表示
          z: (B, L, L, c_z) — Pair 表示
        输出：
          m, z — 更新后的 MSA 和 Pair 表示（形状不变）
        """
        # ── Step 1-3: MSA Stack ──────────────────────────────────────
        # 行注意力：每条序列内的残基互相交流，z 提供 pair 偏置
        m = self.row_attn(m, z)
        # 列注意力：同一位置的不同同源序列互相交流，融合进化信息
        m = self.col_attn(m)
        # MSA FFN：逐位置非线性变换，增强表达能力
        m = m + self.msa_ff(m)

        # ── Step 4: MSA → Pair 融合 ──────────────────────────────────
        # Outer Product Mean：将 MSA 协同进化信息注入 pair 表示
        z = z + self.opm(m)

        # ── Step 5-9: Pair Stack ─────────────────────────────────────
        # 三角乘法 outgoing：z[i,j] 融合 "i→k→j" 路径信息
        z = self.tri_mul_out(z)
        # 三角乘法 incoming：z[i,j] 融合 "k→i, k→j" 路径信息
        z = self.tri_mul_in(z)
        # 三角注意力 outgoing（Algorithm 13）：行注意力 + 第三边 z[j,k] 偏置
        z = self.tri_attn_out(z)
        # 三角注意力 incoming（Algorithm 14）：列注意力 + 第三边 z[k,i] 偏置
        z = self.tri_attn_in(z)
        # Pair FFN：逐对位置非线性变换
        z = self.pair_ff(z)

        return m, z


# ════════════════════════════════════════════════════════════════════
# 4. Structure Module（结构预测模块）
# ════════════════════════════════════════════════════════════════════

class InvariantPointAttention(nn.Module):
    """
    不变点注意力（IPA，接近 AF2 结构的轻量实现）

    核心 logit 由三部分组成：
      1) 单链标量注意力（q_s · k_s）
      2) pair 偏置（由 z 线性投影到各 head）
      3) 几何项（query/key 点在全局坐标中的平方距离，带可学习权重）

    输出融合三路信息：
      - 标量 value 聚合
      - 点 value 聚合后转回 query 局部坐标系
      - 局部点范数
    """
    def __init__(self):
        super().__init__()
        c_s, c_z, h = C.c_s, C.c_z, C.n_heads
        self.n_heads = h
        self.c_head = C.c_ipa
        self.n_qk_points = 4
        self.n_v_points = 8

        self.norm_s = nn.LayerNorm(c_s)
        self.norm_z = nn.LayerNorm(c_z)

        # 标量 q/k/v
        self.q_scalar = nn.Linear(c_s, h * self.c_head, bias=False)
        self.k_scalar = nn.Linear(c_s, h * self.c_head, bias=False)
        self.v_scalar = nn.Linear(c_s, h * self.c_head, bias=False)

        # 点 q/k/v（局部坐标系中的点）
        self.q_points = nn.Linear(c_s, h * self.n_qk_points * 3, bias=False)
        self.k_points = nn.Linear(c_s, h * self.n_qk_points * 3, bias=False)
        self.v_points = nn.Linear(c_s, h * self.n_v_points * 3, bias=False)

        # pair bias: z -> per-head logit bias
        self.pair_bias = nn.Linear(c_z, h, bias=False)
        # 每个 head 一个几何权重（softplus 保证非负）
        self.point_weight = nn.Parameter(torch.zeros(h))

        out_dim = (
            h * self.c_head +
            h * self.n_v_points * 3 +
            h * self.n_v_points
        )
        self.out = nn.Linear(out_dim, c_s)

    def forward(self, s, z, R, t):
        """
        s: (B, L, c_s)  单链表示
        z: (B, L, L, c_z) Pair 表示（提供注意力偏置）
        R: (B, L, 3, 3) 每个残基的旋转矩阵
        t: (B, L, 3)    每个残基的平移（坐标）
        """
        B, L, _ = s.shape
        s_norm = self.norm_s(s)
        z_norm = self.norm_z(z)
        h, d = self.n_heads, self.c_head

        # 标量 q/k/v: (B, L, H, D)
        q_s = self.q_scalar(s_norm).reshape(B, L, h, d)
        k_s = self.k_scalar(s_norm).reshape(B, L, h, d)
        v_s = self.v_scalar(s_norm).reshape(B, L, h, d)

        # 点 q/k/v（局部）: (B, L, H, P, 3)
        q_p_local = self.q_points(s_norm).reshape(B, L, h, self.n_qk_points, 3)
        k_p_local = self.k_points(s_norm).reshape(B, L, h, self.n_qk_points, 3)
        v_p_local = self.v_points(s_norm).reshape(B, L, h, self.n_v_points, 3)

        # 局部点 -> 全局点
        q_p = torch.einsum("blij,blhpj->blhpi", R, q_p_local) + t[:, :, None, None, :]
        k_p = torch.einsum("blij,blhpj->blhpi", R, k_p_local) + t[:, :, None, None, :]
        v_p = torch.einsum("blij,blhpj->blhpi", R, v_p_local) + t[:, :, None, None, :]

        # 标量 logits: (B, i, j, H)
        scalar_logits = torch.einsum("bihd,bjhd->bijh", q_s, k_s) / math.sqrt(d)
        pair_logits = self.pair_bias(z_norm)  # (B, L, L, H)

        # 几何 logits: -(w_h / 2) * ||q_i - k_j||^2
        diff = q_p.unsqueeze(2) - k_p.unsqueeze(1)         # (B, L, L, H, Pqk, 3)
        dist2 = (diff * diff).sum(dim=-1).sum(dim=-1)      # (B, L, L, H)
        w = F.softplus(self.point_weight).view(1, 1, 1, h)
        point_logits = -0.5 * w * dist2

        logits = scalar_logits + pair_logits + point_logits
        attn = F.softmax(logits, dim=2)                     # 对 key 维 j 归一化

        # 聚合标量 value: (B, L, H, D)
        out_s = torch.einsum("bijh,bjhd->bihd", attn, v_s)

        # 聚合点 value（先全局再转回 query 局部）: (B, L, H, Pv, 3)
        out_p_global = torch.einsum("bijh,bjhpc->bihpc", attn, v_p)
        out_p_local = torch.einsum(
            "blxy,blhpy->blhpx",
            R.transpose(-1, -2),
            out_p_global - t[:, :, None, None, :],
        )
        out_p_norm = torch.sqrt((out_p_local * out_p_local).sum(dim=-1) + 1e-8)

        out_cat = torch.cat(
            [
                out_s.reshape(B, L, -1),
                out_p_local.reshape(B, L, -1),
                out_p_norm.reshape(B, L, -1),
            ],
            dim=-1,
        )
        return s + self.out(out_cat)


class StructureModuleBlock(nn.Module):
    """
    Structure Module 的一个迭代步骤：
      1. IPA：让残基间互相交流（结合空间位置信息）
      2. 过渡层：更新单链表示
      3. Backbone 更新：用更新后的表示预测新的旋转/平移
    """
    def __init__(self):
        super().__init__()
        self.ipa = InvariantPointAttention()
        self.norm1 = nn.LayerNorm(C.c_s)
        self.ff = nn.Sequential(
            nn.Linear(C.c_s, C.c_s * 2),
            nn.ReLU(),
            nn.Linear(C.c_s * 2, C.c_s),
        )
        self.norm2 = nn.LayerNorm(C.c_s)

        # 预测旋转更新（用 6D 参数表示旋转，避免万向锁）
        # 输出 6 个数 → 两个 3D 向量 → Gram-Schmidt → 旋转矩阵
        self.update_R = nn.Linear(C.c_s, 6)
        # 预测平移更新
        self.update_t = nn.Linear(C.c_s, 3)

    def forward(self, s, z, R, t):
        # IPA + 过渡
        s = self.ipa(s, z, R, t)
        s = self.norm1(s)
        s = s + self.ff(s)
        s = self.norm2(s)

        # 更新骨架坐标（局部坐标系中的小更新，再转换回全局）
        dR_6d = self.update_R(s)  # (B, L, 6)
        dt    = self.update_t(s)  # (B, L, 3)

        # 把 6D 表示转为旋转矩阵
        a = dR_6d[..., :3]
        b = dR_6d[..., 3:]
        e1 = F.normalize(a, dim=-1)
        b  = b - (b * e1).sum(-1, keepdim=True) * e1
        e2 = F.normalize(b, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        dR = torch.stack([e1, e2, e3], dim=-1)  # (B, L, 3, 3)

        # 更新全局旋转：R_new = R_old @ dR旋转需要做矩阵乘法
        R_new = torch.bmm(
            R.reshape(-1, 3, 3),
            dR.reshape(-1, 3, 3)
        ).reshape(R.shape)

        # 更新全局平移：t_new = t_old + R_old @ dt
        dt_global = torch.bmm(
            R.reshape(-1, 3, 3),
            dt.reshape(-1, 3, 1)
        ).reshape(*dt.shape)
        t_new = t + dt_global

        return s, R_new, t_new


# ════════════════════════════════════════════════════════════════════
# 5. 预测头（Prediction Heads）
# ════════════════════════════════════════════════════════════════════

class DistogramHead(nn.Module):
    """
    Distogram 预测头：从 pair 表示预测残基间 Cβ 距离分布
    一个bin就是一个区间
    将连续距离离散化为 64 个 bin（真实 AF2 为 [2.3125, 21.6875] Å 内的 64 个 bin）。
    """
    def __init__(self, n_bins=64):
        super().__init__()
        self.n_bins = n_bins
        # 对称化 pair 表示后预测距离分布
        self.proj = nn.Linear(C.c_z, n_bins)

    def forward(self, z):
        """z: (B, L, L, c_z) → (B, L, L, n_bins)"""
        # pair 表示对称化（真实 AF2 做法）
        z_sym = (z + z.transpose(1, 2)) / 2
        return self.proj(z_sym)  # (B, L, L, 64) 为每对残基预测距离分布


# ════════════════════════════════════════════════════════════════════
# 6. 完整模型
# ════════════════════════════════════════════════════════════════════

class MiniAlphaFold2(nn.Module):
    """
    AlphaFold2 极简实现。数据流：

      (aatype, msa)
          ↓ InputEmbedder
      (m: MSA表示, z: Pair表示)
          ↓ Evoformer × n_evo
      (m, z)  →  s = m[0] (第一行MSA作为单链表示)
          ↓ StructureModule × n_sm
      (R, t: 每个残基的旋转+位置)
          ↓ 各预测头
      (坐标, distogram, ...)
    """
    def __init__(self):
        super().__init__()
        self.embedder  = InputEmbedder()
        self.evoformer = nn.ModuleList([EvoformerBlock() for _ in range(C.n_evo)])
        self.sm_blocks = nn.ModuleList([StructureModuleBlock() for _ in range(C.n_sm)])
        self.distogram = DistogramHead()

        # 把 MSA 表示投影到单链维度（作为 Structure Module 输入）
        self.m_to_s = nn.Linear(C.c_m, C.c_s)

    def forward(self, batch):
        aatype = batch["aatype"]   # (B, L)
        msa    = batch["msa"]      # (B, N, L)
        B, L   = aatype.shape

        # ── 1. 嵌入 ─────────────────────────────────────────────────
        m, z = self.embedder(aatype, msa)

        # ── 2. Evoformer ─────────────────────────────────────────────
        for block in self.evoformer:
            m, z = block(m, z)

        # ── 3. 初始化 Structure Module 的输入 ────────────────────────
        # 单链表示 s = 第一条序列（查询序列）的 MSA 表示
        s = self.m_to_s(m[:, 0, :, :])  # (B, L, c_s)

        # 初始化所有残基为"单位帧"（位于原点，无旋转）
        R = torch.eye(3, device=aatype.device).unsqueeze(0).unsqueeze(0)\
              .expand(B, L, -1, -1).clone()
        t = torch.zeros(B, L, 3, device=aatype.device)

        # ── 4. Structure Module ────────────────────────────────────────
        for block in self.sm_blocks:
            s, R, t = block(s, z, R, t)

        # t 即为预测的 Cα 坐标
        pred_positions = t  # (B, L, 3)
        pred_R         = R  # (B, L, 3, 3)

        # ── 5. 预测头 ───────────────────────────────────────────────
        logits_distogram = self.distogram(z)  # (B, L, L, 64)

        return {
            "pred_positions":  pred_positions,
            "pred_R":          pred_R,
            "distogram_logits": logits_distogram,
            "single":          s,
            "pair":            z,
        }


# ════════════════════════════════════════════════════════════════════
# 7. 损失函数
# ════════════════════════════════════════════════════════════════════

def fape_loss(pred_R, pred_t, gt_R, gt_t, clamp_dist=10.0, eps=1e-8):
    """
    FAPE（Frame Aligned Point Error）损失

    核心思想：在每个残基 i 的局部坐标系中，
             计算所有其他残基 j 的坐标误差。
    这样对全局旋转/平移不变（等变），是 AF2 最重要的损失。

    数学：
      FAPE = (1/NL) Σ_i Σ_j || R_i^T(t_j - t_i) - R_i^GT_T(t_j^GT - t_i^GT) ||₂

    参数：
      pred_R/pred_t: 预测的旋转矩阵和平移向量 (B, L, 3, 3) / (B, L, 3)
      gt_R/gt_t:     真实的旋转矩阵和平移向量
      clamp_dist:    距离截断（防止梯度爆炸，真实 AF2 用 10Å）
    """
    B, L, _, _ = pred_R.shape

    # 把所有残基 j 的坐标变换到每个残基 i 的局部坐标系
    # 局部坐标 = R_i^T @ (t_j - t_i)
    diff_pred = pred_t.unsqueeze(2) - pred_t.unsqueeze(1)  # (B, L, L, 3)  t_j - t_i
    diff_gt   = gt_t.unsqueeze(2)   - gt_t.unsqueeze(1)    # (B, L, L, 3)

    # 投影到局部坐标系：R_i^T @ diff
    R_T = pred_R.transpose(-2, -1)  # (B, L, 3, 3)
    local_pred = torch.einsum('bilmn,biljn->bilm',
                               R_T.unsqueeze(2).expand(-1,-1,L,-1,-1),
                               diff_pred.unsqueeze(-1)).squeeze(-1)

    R_T_gt = gt_R.transpose(-2, -1)
    local_gt = torch.einsum('bilmn,biljn->bilm',
                              R_T_gt.unsqueeze(2).expand(-1,-1,L,-1,-1),
                              diff_gt.unsqueeze(-1)).squeeze(-1)

    # 计算 L2 距离误差
    err = (local_pred - local_gt).norm(dim=-1)  # (B, L, L)

    # 截断（原论文用 sqrt(x² + ε) 软截断）
    err = torch.sqrt(err ** 2 + eps)
    err = torch.clamp(err, max=clamp_dist)

    return err.mean()


def distogram_loss(logits, gt_positions):
    """
    Distogram 损失：预测残基间 Cβ 距离的分类分布

    步骤：
      1. 从真实坐标计算实际距离
      2. 将连续距离离散化为 bin 索引
      3. 交叉熵损失

    参数：
      logits:       (B, L, L, n_bins) 距离分布预测值
      gt_positions: (B, L, 3)        真实坐标
    """
    # logits 形状：(B, L, L, n_bins)，必须拆出全部 4 维
    # 原代码 shape[:3] 只拿到 (B, L, L)，把 L 误当成 n_bins——
    # 当 L=64=n_bins 时凑巧正确，L=128 时暴露错误。
    B, L, _, n_bins = logits.shape   # _ 是第二个 L（行列方向相同）

    # 计算真实残基对距离
    diff = gt_positions.unsqueeze(2) - gt_positions.unsqueeze(1)  # (B,L,L,3)
    dist = diff.norm(dim=-1)  # (B, L, L)

    # 离散化：将距离映射到 [0, n_bins-1] 的整数 bin
    # 真实 AF2：bins 在 [2.3125, 21.6875] Å，这里简化为 [0, 20] Å
    d_min, d_max = 0.0, 20.0
    bins = ((dist - d_min) / (d_max - d_min) * n_bins).long()
    bins = bins.clamp(0, n_bins - 1)  # (B, L, L)

    # 交叉熵损失（只看上三角，避免重复）
    loss = F.cross_entropy(
        logits.reshape(-1, n_bins),
        bins.reshape(-1)
    )
    return loss


def total_loss(outputs, batch):
    """
    综合损失 = 0.5 * FAPE + 0.5 * Distogram

    真实 AF2 的损失还包括：
      - pLDDT 置信度损失
      - Masked MSA 重建损失（类似 BERT）
      - 违规损失（键长/键角约束）
      - TM-score 损失（Multimer 版本）
    """
    loss_fape = fape_loss(
        outputs["pred_R"],
        outputs["pred_positions"],
        batch["gt_R"],
        batch["gt_t"],
    )
    loss_distogram = distogram_loss(
        outputs["distogram_logits"],
        batch["gt_positions"],
    )
    return 0.5 * loss_fape + 0.5 * loss_distogram, {
        "fape":      loss_fape.item(),
        "distogram": loss_distogram.item(),
    }


# ════════════════════════════════════════════════════════════════════
# 8. 训练循环（真实数据版）
# ════════════════════════════════════════════════════════════════════

def train():
    """
    完整训练流程：
      1. 自动下载并解析 ProteinNet CASP7 数据（首次运行约 2 分钟）
      2. 划分训练集 / 验证集（9:1）
      3. 训练 n_epochs 轮，余弦退火调度学习率
      4. 每轮结束后在验证集评估，保存最佳模型到 mini_af2_best.pt
    """
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
