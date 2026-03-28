"""
dataset.py — ProteinNet CASP7 数据加载与预处理
================================================
从 mini_af2.py 中拆分出来的数据处理模块，包含：

  1. ProteinNet 数据集下载与解压
  2. 文本格式解析（序列、PSSM、骨架坐标、掩码）
  3. 骨架坐标 → 局部坐标系（旋转矩阵 + 平移）
  4. PSSM → 伪 MSA 采样
  5. PyTorch Dataset + DataLoader 封装

使用方式：
    from dataset import build_dataloaders, Config
    train_loader, val_loader = build_dataloaders()
"""

import math
import urllib.request
import tarfile
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ════════════════════════════════════════════════════════════════════
# 0. 全局超参数（数据相关部分）
# ════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════
# 1. 氨基酸字母表
# ════════════════════════════════════════════════════════════════════

# 氨基酸字母表（与 ProteinNet 对应，索引 0~20）
AA_VOCAB  = "ACDEFGHIKLMNPQRSTVWYX"   # 20 种标准氨基酸 + X（未知/填充）
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}


# ════════════════════════════════════════════════════════════════════
# 2. ProteinNet 数据下载与解析
# ════════════════════════════════════════════════════════════════════

# ProteinNet CASP7 人类可读版（最小版本，约 30 MB 压缩包）
_PROTEINNET_URL = (
    "https://sharehost.hms.harvard.edu/sysbio/alquraishi/"
    "proteinnet/human_readable/casp7.tar.gz"
)


def download_proteinnet(data_dir: str = C.data_dir) -> str:
    """
    下载并解压 ProteinNet CASP7 数据集。
    如果已存在则直接返回路径（增量安全）。

    返回值：training_30 文件的绝对路径
      - training_30：约 4 400 条蛋白质（30% 序列同一性去冗余）
    """
    data_dir      = Path(data_dir)
    training_file = data_dir / "casp7" / "training_30"

    # ── 已有缓存，直接返回 ──────────────────────────────────────────
    if training_file.exists():
        print(f"[数据] 找到缓存: {training_file}")
        return str(training_file)

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "casp7.tar.gz"

    # ── 下载压缩包 ─────────────────────────────────────────────────
    if not archive.exists():
        print(f"[数据] 开始下载 ProteinNet CASP7（约 30 MB）...")
        print(f"      URL : {_PROTEINNET_URL}")
        print(f"      目标: {archive}")

        def _progress(block_num, block_size, total_size):
            done = block_num * block_size
            if total_size > 0:
                pct = min(100, done * 100 // total_size)
                bar = "\u2588" * (pct // 5) + "\u2591" * (20 - pct // 5)
                print(f"\r      [{bar}] {pct:3d}%  {done // 1_048_576} / "
                      f"{total_size // 1_048_576} MB", end="", flush=True)

        urllib.request.urlretrieve(_PROTEINNET_URL, archive, reporthook=_progress)
        print()  # 换行

    # ── 解压 ────────────────────────────────────────────────────────
    print("[数据] 解压中...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=data_dir)

    if not training_file.exists():
        raise FileNotFoundError(
            f"解压后未找到 {training_file}，请检查压缩包内容。"
        )
    print(f"[数据] 解压完成: {training_file}")
    return str(training_file)


def diagnose_proteinnet(filepath: str, n_lines: int = 60) -> None:
    """
    打印文件的前 n_lines 行，用于快速诊断 ProteinNet 格式。

    常见问题诊断：
      - PRIMARY 是空格分隔（"M V H L T..."）还是连续串（"MVHLT..."）？
      - TERTIARY 有几行？3行（3x3L格式）还是9行（9xL格式）？
      - 记录之间是否有空行？

    调用方式：
        diagnose_proteinnet("proteinnet_data/casp7/training_30")
    """
    print(f"\n{'='*60}")
    print(f" 文件格式诊断: {Path(filepath).name}")
    print(f"{'='*60}")
    try:
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                if i >= n_lines:
                    print(f"  ... (只显示前 {n_lines} 行)")
                    break
                # 用 repr 显示特殊字符（\r、\t 等）
                print(f"  {i+1:3d}: {repr(line.rstrip())}")
    except FileNotFoundError:
        print(f"  ✗ 文件不存在: {filepath}")
    print(f"{'='*60}\n")


def parse_proteinnet(filepath: str,
                     max_proteins: int = C.max_proteins,
                     min_len: int      = C.min_len,
                     max_len: int      = C.max_len) -> list:
    """
    解析 ProteinNet 文本格式，返回蛋白质记录列表。

    ProteinNet 文本格式（每条记录以空行分隔）：
      [ID]           蛋白质 ID（如 1ABC_1_A）
      [PRIMARY]      氨基酸序列（单字母码，长度 L）
      [EVOLUTIONARY] PSSM：21 行 x L 列（log-score，每行对应一种氨基酸）
      [TERTIARY]     骨架坐标：9 行 x L 列（单位：Å）
                     行顺序：N_x, N_y, N_z, Ca_x, Ca_y, Ca_z, C_x, C_y, C_z
      [MASK]         '+' = 坐标已解析，'-' = 坐标缺失

    返回每条记录的字典：
      id      : str          蛋白质 ID
      seq     : list[int]    氨基酸索引（长度 L_prot）
      pssm    : ndarray      (21, L_prot) 浮点矩阵
      N_pos   : ndarray      (L_prot, 3) N 原子坐标 Å
      CA_pos  : ndarray      (L_prot, 3) Ca 原子坐标 Å
      C_pos   : ndarray      (L_prot, 3) C 原子坐标 Å
      mask    : ndarray      (L_prot,) bool，True = 坐标有效
    """
    records       = []
    current       = {}
    current_field = None
    field_lines   = []

    def _save_field():
        nonlocal current_field, field_lines
        if current_field:
            current[current_field] = field_lines[:]
        current_field = None
        field_lines   = []

    def _finish_record():
        nonlocal current
        _save_field()

        # 必须含序列和坐标才有效
        if "PRIMARY" not in current or "TERTIARY" not in current:
            current = {}
            return

        # ── PRIMARY：兼容两种格式 ─────────────────────────────────
        # 格式 A（连续字符串）: "MVHLTPEEKS..."
        # 格式 B（空格分隔）:   "M V H L T P E E K S..."  <- ProteinNet 实际使用此格式
        seq_str = current["PRIMARY"][0].strip()
        tokens  = seq_str.split()
        if len(tokens) > 1 and all(len(t) == 1 for t in tokens):
            amino_acids = tokens        # 空格分隔的单字母
        else:
            amino_acids = list(seq_str) # 连续字符串逐字符

        L_prot = len(amino_acids)

        # ── 长度过滤 ────────────────────────────────────────────────
        if not (min_len <= L_prot <= max_len):
            current = {}
            return

        # ── 序列编码 ────────────────────────────────────────────────
        seq = [AA_TO_IDX.get(aa, AA_TO_IDX["X"]) for aa in amino_acids]#把字符串转成数字索引，未知的氨基酸用 'X' 的索引表示

        # ── PSSM（21 x L_prot）──────────────────────────────────────
        evo_lines = current.get("EVOLUTIONARY", [])
        if len(evo_lines) >= 21:
            try:
                pssm = np.array(
                    [list(map(float, row.split()))[:L_prot]
                     for row in evo_lines[:21]],
                    dtype=np.float32,
                )  # (21, L_prot)，（i，j）表示该第j个位置是第i种氨基酸的log概率
                if pssm.shape[1] != L_prot:
                    pssm = np.full((21, L_prot), math.log(1.0/21), dtype=np.float32)
            except ValueError:
                pssm = np.full((21, L_prot), math.log(1.0/21), dtype=np.float32)
        else:
            pssm = np.full((21, L_prot), math.log(1.0 / 21), dtype=np.float32)

        # ── TERTIARY：兼容两种坐标存储格式 ──────────────────────────
        # 格式 A（9行 x L列）: 每行一个原子坐标分量
        #   行 0,1,2 -> N(x,y,z)；行 3,4,5 -> Ca(x,y,z)；行 6,7,8 -> C(x,y,z)
        # 格式 B（3行 x 3L列）: 每行一个空间坐标轴，N/Ca/C 在列方向交替
        #   行 0 -> x: [N_x_1, Ca_x_1, C_x_1, N_x_2, ...]
        #   行 1 -> y: [N_y_1, Ca_y_1, C_y_1, ...]
        #   行 2 -> z: [N_z_1, Ca_z_1, C_z_1, ...]
        tert_lines = current["TERTIARY"]
        try:
            n_rows = len(tert_lines)
            if n_rows >= 9:
                # ── 格式 A ───────────────────────────────────────────
                tert = np.array(
                    [list(map(float, l.split())) for l in tert_lines[:9]],
                    dtype=np.float32,
                )  # (9, ncols)
                if tert.shape[1] < L_prot:
                    current = {}; return
                N_pos  = tert[0:3, :L_prot].T.astype(np.float32)  # (L_prot, 3)
                CA_pos = tert[3:6, :L_prot].T.astype(np.float32)
                C_pos  = tert[6:9, :L_prot].T.astype(np.float32)
            elif n_rows >= 3:
                # ── 格式 B ───────────────────────────────────────────
                tert = np.array(
                    [list(map(float, l.split())) for l in tert_lines[:3]],
                    dtype=np.float32,
                )  # (3, 3*ncols_residues)
                if tert.shape[1] < 3 * L_prot:
                    current = {}; return
                # N：每隔3取一，起始 0；Ca：起始 1；C：起始 2
                N_pos  = np.stack([tert[0, 0::3][:L_prot],
                                   tert[1, 0::3][:L_prot],
                                   tert[2, 0::3][:L_prot]], axis=-1)
                CA_pos = np.stack([tert[0, 1::3][:L_prot],
                                   tert[1, 1::3][:L_prot],
                                   tert[2, 1::3][:L_prot]], axis=-1)
                C_pos  = np.stack([tert[0, 2::3][:L_prot],
                                   tert[1, 2::3][:L_prot],
                                   tert[2, 2::3][:L_prot]], axis=-1)
            else:
                current = {}; return
        except (ValueError, IndexError):
            current = {}; return

        # ── 解析 MASK ─────────────────────────────────────────────
        mask_str = current.get("MASK", ["+"])[0].strip()
        mask     = np.array([c == "+" for c in mask_str[:L_prot]], dtype=bool)

        records.append({
            "id":     current.get("ID", ["unknown"])[0].strip(),
            "seq":    seq,
            "pssm":   pssm,
            "N_pos":  N_pos.astype(np.float32),
            "CA_pos": CA_pos.astype(np.float32),
            "C_pos":  C_pos.astype(np.float32),
            "mask":   mask,
        })
        current = {}

    # ── 逐行解析 ───────────────────────────────────────────────────
    print(f"[数据] 解析 {Path(filepath).name}（最多取 {max_proteins} 条）...")
    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\r\n")   # 同时处理 \r\n 和 \r 行尾
            if line.startswith("[") and line.endswith("]"):
                _save_field()
                current_field = line[1:-1]
            elif line == "":
                _finish_record()
                if len(records) >= max_proteins:
                    break
            else:
                field_lines.append(line)
    _finish_record()  # 处理文件末尾可能没有空行的情况

    print(f"[数据] 解析完成，有效蛋白质: {len(records)} 条")
    return records[:max_proteins]


# ════════════════════════════════════════════════════════════════════
# 3. 骨架坐标 → 局部坐标系
# ════════════════════════════════════════════════════════════════════

def backbone_to_frames(N_pos: np.ndarray,
                       CA_pos: np.ndarray,
                       C_pos: np.ndarray):
    """
    从骨架原子坐标计算每个残基的局部坐标系（AF2 Algorithm 21 简化版）。

    约定：以 Ca 为原点，用 N-Ca-C 三原子定义右手坐标系
      e1 = normalize(Ca - N)          N->Ca 方向（主轴）
      n  = normalize(e1 x (C - N))    肽平面法向量
      e2 = n x e1                     在肽平面内与 e1 正交
    旋转矩阵 R = [e1 | e2 | n]（列为基向量）
    平移     t = Ca 坐标（局部系原点）

    输入：N_pos, CA_pos, C_pos — (L, 3) numpy 数组，单位 Å
    输出：R (L, 3, 3) 旋转矩阵，t (L, 3) 平移向量
    """
    eps = 1e-8

    e1  = CA_pos - N_pos                                              # (L, 3) N->Ca
    e1  = e1  / (np.linalg.norm(e1,  axis=-1, keepdims=True) + eps)  # 归一化

    u   = C_pos - N_pos                                               # (L, 3) N->C
    n   = np.cross(e1, u)                                             # (L, 3) 法向量向量

    e2  = np.cross(n, e1)                                             # (L, 3) 第三轴

    R = np.stack([e1, e2, n], axis=-1).astype(np.float32)            # (L, 3, 3)
    t = CA_pos.astype(np.float32).copy()                              # (L, 3)
    return R, t


# ════════════════════════════════════════════════════════════════════
# 4. PSSM → 伪 MSA 采样
# ════════════════════════════════════════════════════════════════════

def pssm_to_msa(pssm: np.ndarray,
                n_seqs: int,
                query_seq: list) -> np.ndarray:
    """
    从 PSSM 矩阵采样 n_seqs 条伪序列，用于构造假 MSA。

    策略：
      第 0 行 = 真实查询序列（不采样，直接赋值）
      第 1~N 行 = 对每个位置独立地按 PSSM 概率分布采样

    输入：
      pssm:      (21, L) log-score 矩阵
      n_seqs:    要生成的总序列数
      query_seq: list[int]，长度 L，真实序列的氨基酸索引
    输出：
      msa: (n_seqs, L) int64 numpy 数组
    """
    L = pssm.shape[1]

    # log-score -> 概率（softmax，数值稳定）
    shifted  = pssm - pssm.max(axis=0, keepdims=True)
    probs    = np.exp(shifted)
    probs   /= probs.sum(axis=0, keepdims=True) + 1e-8  # (21, L)

    msa    = np.zeros((n_seqs, L), dtype=np.int64)
    msa[0] = query_seq  # 第一行 = 真实序列

    for i in range(1, n_seqs):
        for pos in range(L):
            msa[i, pos] = np.random.choice(21, p=probs[:, pos])
    return msa


# ════════════════════════════════════════════════════════════════════
# 5. PyTorch Dataset
# ════════════════════════════════════════════════════════════════════

class ProteinNetDataset(Dataset):
    """
    ProteinNet 蛋白质结构数据集（PyTorch Dataset）。

    对每条记录的处理流程：
      1. 将序列截断到 max_len（或保持原长），不足则用 0（'A'）填充
      2. 用 pssm_to_msa 从 PSSM 采样 N_msa 条伪序列作为 MSA
      3. 由骨架原子坐标（N, Ca, C）计算局部旋转矩阵 R 和平移 t

    返回的字典与原合成数据 make_batch 键名/形状完全兼容：
      aatype       (L,)        氨基酸类型索引（0~20）
      msa          (N_msa, L)  MSA 矩阵（第 0 行 = 真实序列）
      gt_positions (L, 3)      真实 Ca 坐标，Å
      gt_R         (L, 3, 3)   真实旋转矩阵（由骨架计算）
      gt_t         (L, 3)      真实平移（= Ca 坐标）
    """
    def __init__(self,
                 records:  list,
                 max_len:  int = C.max_len,
                 n_msa:    int = C.N_msa):
        self.records = records
        self.max_len = max_len
        self.n_msa   = n_msa

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec    = self.records[idx]
        L_prot = len(rec["seq"])
        L      = min(L_prot, self.max_len)      # 实际使用的长度（可能截断）

        # ── 氨基酸序列（截断 + 填充）────────────────────────────────
        seq       = np.zeros(self.max_len, dtype=np.int64)
        seq[:L]   = rec["seq"][:L]

        # ── 伪 MSA（从 PSSM 采样）───────────────────────────────────
        msa_raw   = pssm_to_msa(rec["pssm"][:, :L],
                                 self.n_msa,
                                 rec["seq"][:L])      # (N_msa, L)
        msa       = np.zeros((self.n_msa, self.max_len), dtype=np.int64)
        msa[:, :L] = msa_raw

        # ── 骨架坐标（截断 + 填充）──────────────────────────────────
        CA_full   = np.zeros((self.max_len, 3), dtype=np.float32)
        N_full    = np.zeros((self.max_len, 3), dtype=np.float32)
        C_full    = np.zeros((self.max_len, 3), dtype=np.float32)
        CA_full[:L] = rec["CA_pos"][:L]
        N_full[:L]  = rec["N_pos"][:L]
        C_full[:L]  = rec["C_pos"][:L]

        # ── 局部坐标系（旋转矩阵 + 平移）────────────────────────────
        R_valid, t_valid = backbone_to_frames(
            N_full[:L], CA_full[:L], C_full[:L]
        )  # (L, 3, 3), (L, 3)

        # 填充区域用单位矩阵/零向量占位（不参与损失计算）
        R_full      = np.tile(np.eye(3, dtype=np.float32), (self.max_len, 1, 1))
        t_full      = np.zeros((self.max_len, 3), dtype=np.float32)
        R_full[:L]  = R_valid
        t_full[:L]  = t_valid

        return {
            "aatype":       torch.from_numpy(seq),        # (max_len,)
            "msa":          torch.from_numpy(msa),        # (N_msa, max_len)
            "gt_positions": torch.from_numpy(CA_full),    # (max_len, 3)
            "gt_R":         torch.from_numpy(R_full),     # (max_len, 3, 3)
            "gt_t":         torch.from_numpy(t_full),     # (max_len, 3)
        }


# ════════════════════════════════════════════════════════════════════
# 6. DataLoader 构建
# ════════════════════════════════════════════════════════════════════

def build_dataloaders(data_dir: str = C.data_dir) -> tuple:
    """
    完整的数据准备流水线：下载 → 解析 → 划分 → 封装 DataLoader。

    返回：(train_loader, val_loader)
    """
    # 下载并解析
    training_file = download_proteinnet(data_dir)
    records       = parse_proteinnet(training_file)

    if len(records) == 0:
        # 自动打印文件头部，帮助诊断格式问题
        print("\n[错误] 解析到 0 条有效蛋白质。正在打印文件前 60 行以诊断格式...\n")
        diagnose_proteinnet(training_file)
        raise RuntimeError(
            "没有找到符合条件的蛋白质。\n"
            "可能原因：\n"
            "  1. PRIMARY 字段格式不符（上方诊断输出可见）\n"
            "  2. TERTIARY 行数不足（需要 ≥3 行）\n"
            "  3. 所有蛋白质长度都超出 [min_len, max_len] 范围\n"
            f"  当前过滤范围: {C.min_len} ≤ L ≤ {C.max_len}，可在 Config 中调整。"
        )

    # 打乱并划分训练/验证集
    rng = np.random.default_rng(seed=42)
    rng.shuffle(records)

    n_val          = max(1, int(len(records) * C.val_ratio))
    train_records  = records[n_val:]
    val_records    = records[:n_val]

    print(f"[数据] 训练集: {len(train_records)} 条 | 验证集: {len(val_records)} 条\n")

    train_ds = ProteinNetDataset(train_records)#把数据集变成PyTorch Dataset对象，方便后续DataLoader加载
    val_ds   = ProteinNetDataset(val_records)

    train_loader = DataLoader(train_ds, batch_size=C.batch,
                              shuffle=True,  num_workers=0)
    
    val_loader   = DataLoader(val_ds,   batch_size=C.batch,
                              shuffle=False, num_workers=0)
    return train_loader, val_loader
