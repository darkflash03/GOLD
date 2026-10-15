import warnings
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

# =============================================================================
# 1. RNA residue 标准化映射
# -----------------------------------------------------------------------------
# 这部分吸收了 pdb_feature.py 的 modified RNA 处理思路：
# - 将常见修饰碱基映射回 A/U/G/C
# - 将 DNA 别名映射到 RNA 字母（例如 DT -> U）
# - 明确忽略水、离子等非 RNA 成分
# =============================================================================

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

    # DNA aliases
    'DA': 'A', 'DT': 'U', 'DG': 'G', 'DC': 'C',

    # Explicitly ignore common solvent / ions
    'HOH': None, 'WAT': None, 'H2O': None,
    'MG': None, 'NA': None, 'CL': None, 'K': None, 'CA': None, 'ZN': None,
}

STANDARD_RNA = {'A', 'U', 'G', 'C'}


# =============================================================================
# 2. 几何计算
# =============================================================================

def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Compute optimal rotation matrix that aligns P to Q.
    P, Q: (N, 3)
    """
    C = P.transpose(-1, -2) @ Q
    V, _, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        V[:, -1] = -V[:, -1]
    U = V @ W
    return U


def get_optimal_transform(src_atoms: np.ndarray, tgt_atoms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optimal rigid transform (rotation + translation) that aligns src to tgt.
    """
    src_center = src_atoms.mean(-2)[None, :]
    tgt_center = tgt_atoms.mean(-2)[None, :]
    r = kabsch_rotation(src_atoms - src_center, tgt_atoms - tgt_center)
    x = tgt_center - src_center @ r
    return r, x


def compute_rmsd(true_atom_pos: np.ndarray, pred_atom_pos: np.ndarray, eps: float = 1e-6) -> float:
    sd = np.square(true_atom_pos - pred_atom_pos).sum(axis=-1)
    msd = np.mean(sd)
    return float(np.sqrt(msd + eps))


def compute_tm(true_atom_pos: np.ndarray, pred_atom_pos: np.ndarray, eps: float = 1e-6) -> float:
    sd = np.square(true_atom_pos - pred_atom_pos).sum(axis=-1)
    num_res = true_atom_pos.shape[0]
    d0 = 1.24 * (max(num_res, 19) - 15) ** (1.0 / 3) - 1.8
    nsd = 1.0 / (1.0 + sd / (d0 ** 2.0))
    return float(nsd.mean())


def compute_gdt(true_atom_pos: np.ndarray, pred_atom_pos: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    d = np.sqrt(np.square(true_atom_pos - pred_atom_pos).sum(axis=-1))

    def p(dist: np.ndarray, k: float) -> float:
        return float((dist <= k).astype(np.float32).sum() / dist.size)

    p0_5 = p(d, 0.5)
    p1 = p(d, 1.0)
    p2 = p(d, 2.0)
    p4 = p(d, 4.0)
    p8 = p(d, 8.0)
    gdt_ts = 0.25 * (p1 + p2 + p4 + p8)
    gdt_ha = 0.25 * (p0_5 + p1 + p2 + p4)
    return float(gdt_ts), float(gdt_ha)


def compute_lddt(
    true_atom_pos: np.ndarray,
    pred_atom_pos: np.ndarray,
    cutoff: float = 15.0,
    eps: float = 1e-10,
) -> float:
    n = true_atom_pos.shape[-2]
    dmat_true = np.sqrt(
        eps + np.sum(
            (true_atom_pos[..., None, :] - true_atom_pos[..., None, :, :]) ** 2,
            axis=-1,
        )
    )
    dmat_pred = np.sqrt(
        eps + np.sum(
            (pred_atom_pos[..., None, :] - pred_atom_pos[..., None, :, :]) ** 2,
            axis=-1,
        )
    )
    dists_to_score = (dmat_true < cutoff).astype(np.float32) * (1.0 - np.eye(n))
    dist_l1 = np.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).astype(np.float32)
        + (dist_l1 < 1.0).astype(np.float32)
        + (dist_l1 < 2.0).astype(np.float32)
        + (dist_l1 < 4.0).astype(np.float32)
    ) * 0.25

    norm = 1.0 / (eps + np.sum(dists_to_score, axis=-1))
    score = norm * (eps + np.sum(dists_to_score * score, axis=-1))
    return float(score.mean())


# =============================================================================
# 3. 兼容旧代码的工具函数（保留，但 compute_monomer 不再依赖 residue-name 对齐）
# =============================================================================

def recursive_perm(intervals, cur_idx=0):
    if cur_idx >= len(intervals):
        return [()]
    ret = []
    for cur_perm in permutations(intervals[cur_idx]):
        for right in recursive_perm(intervals, cur_idx + 1):
            ret.append(cur_perm + right)
    return ret


def generate_perm(entity):
    intervals = []
    pre_eid = -1
    for i, eid in enumerate(entity):
        if eid != pre_eid:
            intervals.append([])
        intervals[-1].append(i)
        pre_eid = eid
    return recursive_perm(intervals)


# =============================================================================
# 4. RNA 解析辅助函数
# -----------------------------------------------------------------------------
# 设计原则：
# - GT / Pred 使用统一的 Bio.PDB / MMCIFParser 解析逻辑
# - 只保留 RNA-like residues
# - 将 modified bases 标准化成 A/U/G/C
# - 保留 chain_id、residue key、atoms，供后续按“位置”建立 correspondence
# =============================================================================

def _normalize_resname(resname: str) -> Optional[str]:
    """
    Map residue name to standard RNA one-letter code.
    Return None if the residue should be ignored.
    """
    name = str(resname).strip().upper()
    if name in STANDARD_RNA:
        return name
    if name in MODIFIED_MAP:
        return MODIFIED_MAP[name]
    return None


def _normalize_icode(icode) -> str:
    if icode is None:
        return ""
    icode = str(icode)
    return "" if icode == " " else icode


def _residue_sort_key(residue) -> Tuple[int, str]:
    """
    Sort residue by (resseq, insertion_code).
    """
    rid = residue.get_id()
    resseq = rid[1]
    icode = _normalize_icode(rid[2])
    return int(resseq), icode


def _is_rna_like_residue(residue) -> bool:
    """
    Heuristic RNA residue filter.

    We want to mimic the spirit of pdb_feature.py:
    keep RNA polymer residues (including common modified residues) and
    exclude water / ions / non-RNA clutter.

    Rules:
    1) residue name must be mappable to A/U/G/C
    2) residue must contain C4' (sugar atom)
    3) residue must contain either N1 or N9 (base anchor atom)
    """
    std_name = _normalize_resname(residue.get_resname())
    if std_name is None:
        return False

    atom_names = {atom.get_name() for atom in residue.get_atoms()}

    # sugar atom present
    if "C4'" not in atom_names:
        return False

    # base anchor present
    if ("N1" not in atom_names) and ("N9" not in atom_names):
        return False

    return True


def _parse_structure(filename: str, is_cif: bool = False) -> List[Dict]:
    """
    Parse PDB/mmCIF file into a unified chain/residue representation.

    Return format:
    [
        {
            "chain_id": "A",
            "residues": [
                {
                    "key": (resseq, icode),
                    "resname": "A" / "U" / "G" / "C",
                    "orig_resname": original residue name,
                    "atoms": {atom_name: np.ndarray(3,)},
                },
                ...
            ]
        },
        ...
    ]
    """
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    structure = parser.get_structure("tmp", filename)

    try:
        model = next(structure.get_models())
    except StopIteration:
        raise ValueError(f"No model found in structure file: {filename}")

    chains_out: List[Dict] = []

    # 按 chain id 排序，保证 GT / Pred 的遍历顺序尽量稳定
    for chain in sorted(model, key=lambda c: c.id):
        residues_out = []

        residues = sorted(list(chain.get_residues()), key=_residue_sort_key)
        for res in residues:
            if not _is_rna_like_residue(res):
                continue

            std_name = _normalize_resname(res.get_resname())
            if std_name is None:
                continue

            rid = res.get_id()
            resseq = int(rid[1])
            icode = _normalize_icode(rid[2])

            atoms = {}
            for atom in res.get_atoms():
                atoms[atom.get_name()] = np.asarray(atom.get_coord(), dtype=np.float64)

            residues_out.append({
                "key": (resseq, icode),
                "resname": std_name,
                "orig_resname": str(res.get_resname()).strip(),
                "atoms": atoms,
            })

        if residues_out:
            chains_out.append({
                "chain_id": chain.id,
                "residues": residues_out,
            })

    if len(chains_out) == 0:
        raise ValueError(f"No RNA-like residues found in file: {filename}")

    return chains_out


def get_pdb(filename: str) -> List[Dict]:
    """
    Backward-compatible entry for PDB parsing.
    """
    return _parse_structure(filename, is_cif=False)


def get_cif(filename: str) -> List[Dict]:
    """
    Backward-compatible entry for mmCIF parsing.
    """
    return _parse_structure(filename, is_cif=True)


# =============================================================================
# 5. Residue correspondence
# -----------------------------------------------------------------------------
# v2 的核心变化：
# - 不再按 residue_name 序列做全局比对
# - 默认按 "chain 顺序 + residue 顺序" 建立对应关系
#
# 为什么这样更适合 inverse folding:
# - inverse folding 允许碱基 identity 改变
# - 但目标 usually 是保留同一个 RNA 拓扑/位点框架
# - 所以 GT 的第 i 个 residue 应该与 Pred 的第 i 个 residue 比
#
# 额外做了一个小的鲁棒处理：
# - 若同一 chain 的 residue 数量不一致，优先尝试按 residue key 交集配对
# - 若 key 交集不够，再退回到“按顺序截断到 min_len”
# =============================================================================

def _pair_chain_residues(gt_residues: List[Dict], pred_residues: List[Dict], chain_name: str = ""):
    """
    Pair residues within one chain.

    Priority:
    1) if lengths are equal: pair by order
    2) else if residue keys overlap sufficiently: pair by key
    3) else fallback to pair by order up to min length
    """
    if len(gt_residues) == 0 or len(pred_residues) == 0:
        return []

    # Case 1: ideal case, same length -> strict position-wise pairing
    if len(gt_residues) == len(pred_residues):
        return list(zip(gt_residues, pred_residues))

    # Case 2: try residue-key pairing if numbering overlaps enough
    gt_by_key = {res["key"]: res for res in gt_residues}
    pred_by_key = {res["key"]: res for res in pred_residues}

    common_keys = [res["key"] for res in gt_residues if res["key"] in pred_by_key]
    min_len = min(len(gt_residues), len(pred_residues))

    # “足够重合”阈值可以稍微保守一些
    if len(common_keys) >= max(1, int(0.8 * min_len)):
        warnings.warn(
            f"[align_residues] Chain {chain_name}: residue counts differ "
            f"(gt={len(gt_residues)}, pred={len(pred_residues)}), "
            f"fall back to residue-key matching on {len(common_keys)} common residues."
        )
        return [(gt_by_key[k], pred_by_key[k]) for k in common_keys]

    # Case 3: final fallback -> position-wise truncation
    warnings.warn(
        f"[align_residues] Chain {chain_name}: residue counts differ "
        f"(gt={len(gt_residues)}, pred={len(pred_residues)}), and residue keys "
        f"do not overlap enough. Fallback to position-wise pairing on first {min_len} residues."
    )
    return list(zip(gt_residues[:min_len], pred_residues[:min_len]))


def align_residues(gt: List[Dict], pred: List[Dict]) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """
    Align residues between two structures based on topology/order, NOT residue_name.

    Return two flattened dicts:
        filtered_gt[idx] = residue_dict
        filtered_pred[idx] = residue_dict

    This keeps the output style close to the old version so that downstream code
    such as extract_c4_coords can stay simple.
    """
    filtered_gt: Dict[int, Dict] = {}
    filtered_pred: Dict[int, Dict] = {}

    if len(gt) != len(pred):
        warnings.warn(
            f"[align_residues] Number of chains differ: gt={len(gt)}, pred={len(pred)}. "
            f"Only the first {min(len(gt), len(pred))} chains will be compared in order."
        )

    n_chain = min(len(gt), len(pred))
    counter = 0

    for i in range(n_chain):
        gt_chain = gt[i]
        pred_chain = pred[i]
        chain_name = f"{gt_chain['chain_id']} vs {pred_chain['chain_id']}"

        pairs = _pair_chain_residues(gt_chain["residues"], pred_chain["residues"], chain_name=chain_name)
        for gt_res, pred_res in pairs:
            filtered_gt[counter] = gt_res
            filtered_pred[counter] = pred_res
            counter += 1

    print("aligned res:", counter)

    if counter == 0:
        raise ValueError("No aligned RNA residues found between ground-truth and prediction.")

    return filtered_gt, filtered_pred


def extract_c4_coords(filtered_gt: Dict[int, Dict], filtered_pred: Dict[int, Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract C4' coordinates from aligned residue pairs.

    Only keep positions where both GT and Pred contain C4'.
    """
    gt_coords = []
    pred_coords = []

    for key in filtered_gt:
        if key not in filtered_pred:
            continue

        gt_res = filtered_gt[key]
        pred_res = filtered_pred[key]

        if "C4'" in gt_res["atoms"] and "C4'" in pred_res["atoms"]:
            gt_coords.append(gt_res["atoms"]["C4'"])
            pred_coords.append(pred_res["atoms"]["C4'"])

    if len(gt_coords) == 0 or len(pred_coords) == 0:
        raise ValueError("No common C4' coordinates found after residue alignment.")

    gt_coords = np.asarray(gt_coords, dtype=np.float64)
    pred_coords = np.asarray(pred_coords, dtype=np.float64)
    return gt_coords, pred_coords


def get_coords(gt, pred):
    """
    Backward-compatible helper.
    In v2 this follows the new topology/order-based alignment policy.
    """
    filtered_gt, filtered_pred = align_residues(gt, pred)
    return extract_c4_coords(filtered_gt, filtered_pred)


# =============================================================================
# 6. 主接口
# -----------------------------------------------------------------------------
# compute_monomer 的语义保持不变：
# - 读 GT / Pred
# - 建立 residue correspondence
# - 提取 C4'
# - Kabsch 对齐
# - 计算 RMSD / lDDT
#
# 但 v2 的 correspondence 规则已经改为：
# position-wise / topology-wise
# =============================================================================

def compute_monomer(gt_pdb: str, pred_pdb: str) -> Dict[str, float]:
    """
    Compute monomer metrics for RNA structure comparison.

    Assumptions:
    - GT and Pred represent the same RNA topology / residue positions.
    - We evaluate whether Pred recovers the target fold at corresponding positions.
    - Therefore residue correspondence is established by topology/order, not by base identity.

    Args:
        gt_pdb: path to ground-truth PDB/mmCIF file
        pred_pdb: path to predicted PDB/mmCIF file

    Returns:
        metrics dict with:
            - rmsd
            - lddt
    """
    gt = get_cif(gt_pdb) if gt_pdb.endswith(".cif") else get_pdb(gt_pdb)
    pred = get_cif(pred_pdb) if pred_pdb.endswith(".cif") else get_pdb(pred_pdb)

    filtered_gt, filtered_pred = align_residues(gt, pred)
    gt_coords, pred_coords = extract_c4_coords(filtered_gt, filtered_pred)

    # rigid-body alignment: pred -> gt
    r, x = get_optimal_transform(pred_coords, gt_coords)
    pred_coords_aligned = pred_coords @ r + x

    best_rmsd = compute_rmsd(gt_coords, pred_coords_aligned)
    best_lddt = compute_lddt(gt_coords, pred_coords_aligned)

    metrics = {
        "rmsd": float(best_rmsd),
        "lddt": float(best_lddt),
    }
    return metrics