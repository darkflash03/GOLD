import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from pathlib import Path

from utils.constants import SIDE_CHAIN_ATOMS, RESD_SIDE_CHAIN_ATOM_NAME, NUCLEOTIDES
from utils.rigid_utils import torsions

# =======================================================================
# 1. 常量定义
# =======================================================================

# --- 新增：常见 RNA 修饰碱基映射表 ---
# 这是一个相对完整的列表，涵盖了 PDB 中绝大多数修饰 RNA
MODIFIED_MAP = {
    # A variants
    '1MA': 'A', 'M1A': 'A', 'ADP': 'A', 'ATP': 'A', 'A23': 'A',
    '2MA': 'A', 'LCA': 'A', '6IA': 'A', 'RIA': 'A',

    # G variants
    'GTP': 'G', 'GDP': 'G', 'GMP': 'G', 'G7M': 'G', '1MG': 'G',
    '2MG': 'G', 'M2G': 'G', '7MG': 'G', 'OMG': 'G', 'YG': 'G',

    # C variants
    '5MC': 'C', 'OMC': 'C', 'CCC': 'C', '4OC': 'C', 'CBV': 'C',

    # U variants
    'URA': 'U', 'UMP': 'U', 'UDP': 'U', 'UTP': 'U',
    'PSU': 'U', 'H2U': 'U', '5MU': 'U', 'OMU': 'U', '4SU': 'U', 'UR3': 'U',

    # DNA aliases (以防万一混入)
    'DA': 'A', 'DT': 'U', 'DG': 'G', 'DC': 'C',

    # Common Ions/Water (Explicitly Ignore)
    'HOH': None, 'MG': None, 'NA': None, 'CL': None, 'K': None
}

BB_COARSE_DEF = {
    'A': ["C4'", "C1'", "N9"],
    'G': ["C4'", "C1'", "N9"],
    'C': ["C4'", "C1'", "N1"],
    'U': ["C4'", "C1'", "N1"]
}

TORSION_DEF = {
    'A': ["C4'", "C1'", "N9", "C4"],
    'G': ["C4'", "C1'", "N9", "C4"],
    'C': ["C4'", "C1'", "N1", "C2"],
    'U': ["C4'", "C1'", "N1", "C2"]
}

BB_OBSTACLE_ATOMS = ['P', "O5'", "C5'", "C3'", "O3'", "C1'", "C4'", "N1", "N9"]


# =======================================================================
# 2. 辅助函数
# =======================================================================

def load_pdb_wrapper(filename: str, pdbid: str = ""):
    if not pdbid:
        pdbid = Path(filename).stem.split('_')[0]
    try:
        from fr3d.pdb.pdb_reader import PDBStructure
        with open(filename, 'r') as raw:
            return PDBStructure(pdbid, raw), None
    except ImportError:
        return None, "Error: fr3d-python not installed."
    except Exception as ex:
        return None, f"Error loading {filename}: {ex}"


def parse_dot_bracket(sec_str):
    stack = []
    pairs = []
    for i, char in enumerate(sec_str):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                pairs.append([stack.pop(), i])
    if len(pairs) == 0:
        return np.empty((2, 0), dtype=np.int64)
    return np.array(pairs, dtype=np.int64).T


# =======================================================================
# 3. 核心特征提取
# =======================================================================

def make_feature(pdb_path: str, sec_info=None):
    pdb_wrapper, error = load_pdb_wrapper(pdb_path)
    if error: raise ValueError(error)

    # 1. 筛选 RNA Residues
    res_list = [res for res in pdb_wrapper.residues if res.type == "RNA linking"]
    res_list.sort(key=lambda res: (res.chain, res.number))

    N = len(res_list)
    if N == 0: raise ValueError("No RNA residues found")

    # 2. 初始化
    coords_coarse = np.zeros((N, 3, 3), dtype=np.float32)
    mask_coarse = np.zeros((N, 3), dtype=bool)
    coords_gt = np.zeros((N, 14, 3), dtype=np.float32)
    mask_gt = np.zeros((N, 14), dtype=bool)
    torsion_sincos = np.zeros((N, 2), dtype=np.float32)
    mask_torsion = np.zeros(N, dtype=bool)
    torsion_sincos[:, 1] = 1.0
    coords_obstacle = np.full((N, 8, 3), np.nan, dtype=np.float32)

    seq_chars = []

    # 3. 遍历提取
    for i, res in enumerate(res_list):
        # --- 核心修复：序列名称映射 ---
        orig_name = res.sequence  # 可能是 'PSU', '5MC' 等
        if orig_name in NUCLEOTIDES:
            std_name = orig_name
        elif orig_name in MODIFIED_MAP:
            std_name = MODIFIED_MAP[orig_name]
            # 如果映射为 None (如水分子混入)，则跳过或标记
            if std_name is None: std_name = 'A'
        else:
            # 实在不认识的，才标记为 N 或 A
            # print(f"Unknown residue: {orig_name}")
            std_name = 'A'  # 建议用 N，比 A 更诚实

        seq_chars.append(std_name)

        # 为了后续查找常量表（BB_COARSE_DEF），必须用标准名
        # 如果是 'N'，我们暂时用 'A' 的几何定义来尝试提取骨架，以防崩溃
        lookup_name = std_name if std_name in NUCLEOTIDES else 'A'

        atoms_dict = {atom.name: np.array([atom.x, atom.y, atom.z], dtype=np.float32)
                      for atom in res.atoms()}

        # A. 粗粒度骨架
        targets = BB_COARSE_DEF[lookup_name]
        for k, atom_name in enumerate(targets):
            if atom_name in atoms_dict:
                coords_coarse[i, k] = atoms_dict[atom_name]
                mask_coarse[i, k] = True

        # B. 侧链 Ground Truth
        if lookup_name in RESD_SIDE_CHAIN_ATOM_NAME:
            for atom_name in RESD_SIDE_CHAIN_ATOM_NAME[lookup_name]:
                if atom_name in SIDE_CHAIN_ATOMS and atom_name in atoms_dict:
                    idx = SIDE_CHAIN_ATOMS[atom_name]
                    coords_gt[i, idx] = atoms_dict[atom_name]
                    mask_gt[i, idx] = True

        # C. 扭转角
        tor_names = TORSION_DEF[lookup_name]
        if all(name in atoms_dict for name in tor_names):
            p = [torch.tensor(atoms_dict[name]) for name in tor_names]
            sin_val, cos_val = torsions(*p)
            torsion_sincos[i] = [sin_val, cos_val]
            mask_torsion[i] = True

        # D. 障碍物
        std_bb = ['P', "O5'", "C5'", "C3'", "O3'", "C1'", "C4'"]
        n_conn = 'N1' if lookup_name in ['C', 'U'] else 'N9'
        std_bb.append(n_conn)

        for k, atom_name in enumerate(std_bb):
            if atom_name in atoms_dict:
                coords_obstacle[i, k] = atoms_dict[atom_name]

    # 4. 组装
    str_seq = "".join(seq_chars)
    assert len(str_seq) == coords_coarse.shape[0] and len(str_seq) == mask_coarse.shape[0] and len(str_seq) > 0
    letter_to_num = dict(zip(
        NUCLEOTIDES,
        list(range(len(NUCLEOTIDES)))
    ))
    seq2num = [letter_to_num[residue] for residue in str_seq]
    seq = torch.tensor(seq2num, dtype=torch.int64)

    if sec_info is None or len(sec_info) != N:
        ss_pairs = np.empty((2, 0), dtype=np.int64)
    else:
        ss_pairs = parse_dot_bracket(sec_info)

    return {
        'str_seq': str_seq,
        "seq": seq,
        "coords_coarse": coords_coarse,
        "mask_coarse": mask_coarse,
        "coords_gt": coords_gt,
        "mask_gt": mask_gt,
        "angles_gt": torsion_sincos,
        "mask_angles": mask_torsion,
        "ss_pairs": ss_pairs,
        "coords_obstacle": coords_obstacle
    }