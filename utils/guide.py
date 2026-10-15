import torch
from utils.constants import NUCLEOTIDES
import torch.nn.functional as F

def compute_semantic_guidance(seq_probs, ss_pairs):
    if not ss_pairs:        #没有二级结构约束的配对引导
        return torch.tensor(0.0, device=seq_probs.device)

    letter_to_num = dict(zip(
        NUCLEOTIDES,
        list(range(len(NUCLEOTIDES)))
    ))

    loss = 0.0

    #只计算三种主流的配对关系 (A, U) (G, C) (U, G)
    for (i, j) in ss_pairs:
        p_i = seq_probs[i]
        p_j = seq_probs[j]
        valid_prob = (p_i[letter_to_num['A']] * p_j[letter_to_num['U']] + p_i[letter_to_num['U']] * p_j[letter_to_num['A']] +
                      p_i[letter_to_num['G']] * p_j[letter_to_num['C']] + p_i[letter_to_num['C']] * p_j[letter_to_num['G']] +
                      p_i[letter_to_num['G']] * p_j[letter_to_num['U']] + p_i[letter_to_num['U']] * p_j[letter_to_num['G']])
        loss += -torch.log(valid_prob + 1e-8)
    return loss / len(ss_pairs)


def compute_clash_guidance(pred_coords, mask_atoms, clash_threshold=2.0):
    """
    计算侧链碰撞排斥引导的能量 (物理正确版：屏蔽共价键与近邻残基)
    因为 batch_size=1，所以 pred_coords 就是单条序列的坐标 [L, 14, 3]
    """
    L = pred_coords.shape[0]  # 序列长度
    if L < 3:
        return torch.tensor(0.0, device=pred_coords.device)

    # 1. 记录每个原子属于第几个残基 (0 到 L-1)
    res_idx_matrix = torch.arange(L, device=pred_coords.device).view(-1, 1).expand(L, 14)

    # 2. 提取出所有真实存在的有效原子及其残基归属
    valid_coords = pred_coords[mask_atoms]  # [N_valid_atoms, 3]
    valid_res_idx = res_idx_matrix[mask_atoms]  # [N_valid_atoms]

    if len(valid_coords) < 2:
        return torch.tensor(0.0, device=pred_coords.device)

    # 3. 计算所有有效原子两两之间的距离矩阵
    dist_matrix = torch.cdist(valid_coords, valid_coords)

    # 4. 【核心物理规则】非键合掩码 (Non-bonded Mask)
    # 屏蔽属于同一个残基(差值0) 或 相邻残基(差值1) 的原子对
    diff_matrix = torch.abs(valid_res_idx.unsqueeze(0) - valid_res_idx.unsqueeze(1))
    ignore_mask = (diff_matrix <= 1)

    dist_matrix.masked_fill(ignore_mask, 999.0)  # 把合法近距离变成大数，避开惩罚

    # 5. 计算排斥能量
    clash_loss = F.relu(clash_threshold - dist_matrix)
    energy = torch.sum(clash_loss ** 2) / 2.0

    return energy