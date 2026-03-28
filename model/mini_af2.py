"""
model/mini_af2.py — AlphaFold2 模型架构极简实现
================================================
包含所有模型组件和损失函数：

  1. InputEmbedder          — 序列/MSA 嵌入 + 相对位置编码
  2. Evoformer 模块          — MSA/Pair 交互更新
     - MSARowAttentionWithPairBias
     - MSAColumnAttention
     - OuterProductMean
     - TriangleMultiplicativeUpdate
     - TriangleAttention
     - PairTransition
     - EvoformerBlock
  3. Structure Module        — IPA + 刚体变换
     - InvariantPointAttention
     - StructureModuleBlock
  4. DistogramHead           — 距离分布预测头
  5. MiniAlphaFold2          — 顶层模型
  6. 损失函数                — FAPE + Distogram + total_loss
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import C


# ════════════════════════════════════════════════════════════════════
# 1. 输入嵌入层
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
# 2. Evoformer 模块
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
      outgoing=False → 终止节点注意力（Algorithm 14）
    """
    def __init__(self, outgoing=True):
        super().__init__()
        self.outgoing    = outgoing
        self.norm        = nn.LayerNorm(C.c_z)
        self.linear_bias = nn.Linear(C.c_z, C.n_heads, bias=False)
        self.gate        = nn.Linear(C.c_z, C.c_z)
        self.attn        = nn.MultiheadAttention(C.c_z, C.n_heads,
                                                 batch_first=True)

    def forward(self, z):
        """
        输入/输出：z: (B, L, L, c_z) — Pair 表示，输入输出形状相同
        """
        B, L, _, _ = z.shape
        residual = z

        # 终止节点（incoming）：通过转置复用起始节点代码路径
        if not self.outgoing:
            z = z.transpose(1, 2)

        z_norm = self.norm(z)

        # Triangle Bias（三角偏置）
        bias = self.linear_bias(z_norm)        # (B, L, L, n_heads)
        bias = bias.permute(0, 3, 1, 2)         # (B, n_heads, L, L)
        bias = bias.unsqueeze(1).expand(-1, L, -1, -1, -1)
        bias = bias.reshape(B * L * C.n_heads, L, L)
        z_flat = z_norm.reshape(B * L, L, C.c_z)
        attn_out = self.attn(z_flat, z_flat, z_flat, attn_mask=bias)[0]
        attn_out = attn_out.reshape(B, L, L, C.c_z)

        # 门控（Gating）
        gate     = torch.sigmoid(self.gate(z_norm))
        attn_out = gate * attn_out

        # 终止节点（incoming）：转置回原始方向
        if not self.outgoing:
            attn_out = attn_out.transpose(1, 2)

        return residual + attn_out


class EvoformerBlock(nn.Module):
    """
    一个完整的 Evoformer 块（AF2 Algorithm 6 简化版）

    数据流：
      MSA 表示 m (B,N,L,c_m)  ←→  Pair 表示 z (B,L,L,c_z)

    更新顺序（严格遵循论文 Algorithm 6）：
      Step 1. MSA 行注意力（带 Pair 偏置）
      Step 2. MSA 列注意力
      Step 3. MSA Transition（FFN）
      Step 4. Outer Product Mean（MSA → Pair 融合）
      Step 5. 三角乘法更新 outgoing （Algorithm 11）
      Step 6. 三角乘法更新 incoming （Algorithm 12）
      Step 7. 三角注意力 outgoing   （Algorithm 13）
      Step 8. 三角注意力 incoming   （Algorithm 14）
      Step 9. Pair Transition（FFN）
    """
    def __init__(self):
        super().__init__()
        # ── MSA Stack 子模块 ──────────────────────────────────────────
        self.row_attn  = MSARowAttentionWithPairBias()
        self.col_attn  = MSAColumnAttention()
        self.msa_ff    = nn.Sequential(
            nn.LayerNorm(C.c_m),
            nn.Linear(C.c_m, C.c_m * 2),
            nn.ReLU(),
            nn.Linear(C.c_m * 2, C.c_m)
        )
        # ── MSA → Pair 融合 ───────────────────────────────────────────
        self.opm         = OuterProductMean()

        # ── Pair Stack 子模块 ─────────────────────────────────────────
        self.tri_mul_out  = TriangleMultiplicativeUpdate(outgoing=True)
        self.tri_mul_in   = TriangleMultiplicativeUpdate(outgoing=False)
        self.tri_attn_out = TriangleAttention(outgoing=True)
        self.tri_attn_in  = TriangleAttention(outgoing=False)
        self.pair_ff      = PairTransition()

    def forward(self, m, z):
        """
        输入：
          m: (B, N, L, c_m) — MSA 表示
          z: (B, L, L, c_z) — Pair 表示
        输出：
          m, z — 更新后的 MSA 和 Pair 表示（形状不变）
        """
        # ── Step 1-3: MSA Stack ──────────────────────────────────────
        m = self.row_attn(m, z)
        m = self.col_attn(m)
        m = m + self.msa_ff(m)

        # ── Step 4: MSA → Pair 融合 ──────────────────────────────────
        z = z + self.opm(m)

        # ── Step 5-9: Pair Stack ─────────────────────────────────────
        z = self.tri_mul_out(z)
        z = self.tri_mul_in(z)
        z = self.tri_attn_out(z)
        z = self.tri_attn_in(z)
        z = self.pair_ff(z)

        return m, z


# ════════════════════════════════════════════════════════════════════
# 3. Structure Module（结构预测模块）
# ════════════════════════════════════════════════════════════════════

class InvariantPointAttention(nn.Module):
    """
    不变点注意力（IPA，接近 AF2 结构的轻量实现）

    核心 logit 由三部分组成：
      1) 单链标量注意力（q_s · k_s）
      2) pair 偏置（由 z 线性投影到各 head）
      3) 几何项（query/key 点在全局坐标中的平方距离，带可学习权重）
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

        # 更新全局旋转：R_new = R_old @ dR
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
# 4. 预测头（Prediction Heads）
# ════════════════════════════════════════════════════════════════════

class DistogramHead(nn.Module):
    """
    Distogram 预测头：从 pair 表示预测残基间 Cb 距离分布
    一个bin就是一个区间
    将连续距离离散化为 64 个 bin（真实 AF2 为 [2.3125, 21.6875] Å 内的 64 个 bin）。
    """
    def __init__(self, n_bins=64):
        super().__init__()
        self.n_bins = n_bins
        self.proj = nn.Linear(C.c_z, n_bins)

    def forward(self, z):
        """z: (B, L, L, c_z) → (B, L, L, n_bins)"""
        z_sym = (z + z.transpose(1, 2)) / 2
        return self.proj(z_sym)


# ════════════════════════════════════════════════════════════════════
# 5. 完整模型
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
        s = self.m_to_s(m[:, 0, :, :])  # (B, L, c_s)

        R = torch.eye(3, device=aatype.device).unsqueeze(0).unsqueeze(0)\
              .expand(B, L, -1, -1).clone()
        t = torch.zeros(B, L, 3, device=aatype.device)

        # ── 4. Structure Module ────────────────────────────────────────
        for block in self.sm_blocks:
            s, R, t = block(s, z, R, t)

        pred_positions = t
        pred_R         = R

        # ── 5. 预测头 ───────────────────────────────────────────────
        logits_distogram = self.distogram(z)

        return {
            "pred_positions":  pred_positions,
            "pred_R":          pred_R,
            "distogram_logits": logits_distogram,
            "single":          s,
            "pair":            z,
        }


# ════════════════════════════════════════════════════════════════════
# 6. 损失函数
# ════════════════════════════════════════════════════════════════════

def fape_loss(pred_R, pred_t, gt_R, gt_t, clamp_dist=10.0, eps=1e-8):
    """
    FAPE（Frame Aligned Point Error）损失

    核心思想：在每个残基 i 的局部坐标系中，
             计算所有其他残基 j 的坐标误差。

    数学：
      FAPE = (1/NL) Σ_i Σ_j || R_i^T(t_j - t_i) - R_i^GT_T(t_j^GT - t_i^GT) ||₂
    """
    B, L, _, _ = pred_R.shape

    diff_pred = pred_t.unsqueeze(2) - pred_t.unsqueeze(1)
    diff_gt   = gt_t.unsqueeze(2)   - gt_t.unsqueeze(1)

    R_T = pred_R.transpose(-2, -1)
    local_pred = torch.einsum('bilmn,biljn->bilm',
                               R_T.unsqueeze(2).expand(-1,-1,L,-1,-1),
                               diff_pred.unsqueeze(-1)).squeeze(-1)

    R_T_gt = gt_R.transpose(-2, -1)
    local_gt = torch.einsum('bilmn,biljn->bilm',
                              R_T_gt.unsqueeze(2).expand(-1,-1,L,-1,-1),
                              diff_gt.unsqueeze(-1)).squeeze(-1)

    err = (local_pred - local_gt).norm(dim=-1)
    err = torch.sqrt(err ** 2 + eps)
    err = torch.clamp(err, max=clamp_dist)

    return err.mean()


def distogram_loss(logits, gt_positions):
    """
    Distogram 损失：预测残基间 Cb 距离的分类分布

    步骤：
      1. 从真实坐标计算实际距离
      2. 将连续距离离散化为 bin 索引
      3. 交叉熵损失
    """
    B, L, _, n_bins = logits.shape

    diff = gt_positions.unsqueeze(2) - gt_positions.unsqueeze(1)
    dist = diff.norm(dim=-1)

    d_min, d_max = 0.0, 20.0
    bins = ((dist - d_min) / (d_max - d_min) * n_bins).long()
    bins = bins.clamp(0, n_bins - 1)

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
