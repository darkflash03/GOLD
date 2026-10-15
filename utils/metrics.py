import torch
import os
import RNA
from utils.predict_3d_structure import batch_rhofold
from utils.evaluate_3d_structure import compute_monomer

def evaluate_clash_score(pred_coords_single, mask_atoms_single, clash_threshold=2.0):
    """
    计算评估用的 Clash Score (每 1000 个原子的碰撞数)
    严格屏蔽共价键与相邻核苷酸
    """
    L = pred_coords_single.shape[0]
    if L < 3:
        return 0.0, []

    res_idx_matrix = torch.arange(L, device=pred_coords_single.device).view(-1, 1).expand(L, 14)
    valid_coords = pred_coords_single[mask_atoms_single]
    valid_res_idx = res_idx_matrix[mask_atoms_single]

    num_valid_atoms = len(valid_coords)
    if num_valid_atoms < 2:
        return 0.0, []

    dist_matrix = torch.cdist(valid_coords, valid_coords)

    # 屏蔽自对角线与近邻核苷酸 (|i - j| <= 1)
    diff_matrix = torch.abs(valid_res_idx.unsqueeze(0) - valid_res_idx.unsqueeze(1))
    ignore_mask = (diff_matrix <= 1)
    dist_matrix.masked_fill_(ignore_mask, 999.0)

    # === 提取碰撞残基对 ===
    # 找到所有距离小于阈值的原子对索引 (Nx2)
    clash_mask = dist_matrix < clash_threshold
    clash_indices = torch.nonzero(clash_mask)

    clashing_res_pairs = set()
    for idx in clash_indices:
        atom_i, atom_j = idx[0].item(), idx[1].item()
        res_i = valid_res_idx[atom_i].item()
        res_j = valid_res_idx[atom_j].item()

        # 只保留唯一的残基对，且 i < j 避免重复记录 (A,B) 和 (B,A)
        if res_i < res_j:
            # 【重要细节】：Python 是 0-based，但我们导出的 PDB 里 res_seq 是从 1 开始的
            # 为了让你在 PyMOL 里直接用，这里直接给它们 +1
            clashing_res_pairs.add((res_i + 1, res_j + 1))


    # 统计发生碰撞的原子对数量 (单边计数)
    num_clashes = (dist_matrix < clash_threshold).sum().item() / 2.0

    # 转换为每 1000 个原子的标准 Clash Score
    clash_score = (num_clashes / num_valid_atoms) * 1000.0
    return clash_score, sorted(list(clashing_res_pairs))

def seq_recovery(gt_seq, pred_seq):
    ind = (gt_seq == pred_seq.argmax(dim=-1))
    recovery = ind.sum() / ind.shape[0]
    return recovery, ind.cpu()

def secondary_structure_metric(gt_seq, pred_seq):
    pred_structure, pred_mfe = RNA.fold(pred_seq)
    gt_structure, gt_mfe = RNA.fold(gt_seq)

    similarity_count = 0
    min_len = min(len(pred_structure), len(gt_structure))
    for i in range(min_len):
        if pred_structure[i] == gt_structure[i]:
            similarity_count += 1

    secondary_structure_similarity = (similarity_count / min_len) * 100 if min_len > 0 else 0.0
    return secondary_structure_similarity, pred_mfe

def three_dimension_metric(pred_seq, pdb_name, pdb_data_dir, out_dir, n_jobs=1):
    output_files = batch_rhofold(pred_seq, pdb_name, out_dir, n_jobs=n_jobs)
    rmsd_list = []
    lddt_list = []
    for output_file in output_files:
        gt_pdb_file = os.path.join(pdb_data_dir, pdb_name + ".pdb")
        metrics = compute_monomer(gt_pdb_file, output_file)
        rmsd = metrics['rmsd']
        lddt = metrics['lddt']
        rmsd_list.append(rmsd)
        lddt_list.append(lddt)

    return rmsd_list, lddt_list

def af3_metric(pdb_name, pdb_data_dir, af3_pdb_data_dir):
    gt_pdb_file = os.path.join(pdb_data_dir, pdb_name + ".pdb")
    af3_pdb_file = os.path.join(af3_pdb_data_dir, pdb_name + ".pdb")
    metrics = compute_monomer(gt_pdb_file, af3_pdb_file)
    rmsd = metrics['rmsd']
    lddt = metrics['lddt']
    return rmsd, lddt

def compute_sequence_similarity(pred_seq, gt_seq):
    assert len(pred_seq) == len(gt_seq)
    eq_cnt = 0
    for i in range(len(pred_seq)):
        if pred_seq[i] == gt_seq[i]:
            eq_cnt += 1
    recovery = eq_cnt / len(pred_seq)
    return recovery