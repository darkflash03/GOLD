import numpy as np
import random
import os
import torch
import yaml
from .constants import NUCLEOTIDES
import math

def seeding(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("seeding done !")

def load_config(config_path):
    with open(config_path, 'r', encoding="utf-8") as f:
        running_config = yaml.load(f, Loader=yaml.FullLoader)
    return running_config

def save_config(config_path, config):
    with open(config_path, 'w', encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

def to_tensor(val, dtype=torch.float32):
    if isinstance(val, np.ndarray):
        return torch.from_numpy(val).type(dtype)
    elif isinstance(val, torch.Tensor):
        return val.type(dtype)
    return val

idx_to_base = {i: base for i, base in enumerate(NUCLEOTIDES)}
def decode_seq_tensor(seq_tensor):
    """将 Tensor [0, 1, 3...] 转回字符串 'AGCU...'"""
    indices = seq_tensor.cpu().numpy().tolist()
    return "".join([idx_to_base.get(i, '?') for i in indices])

def parse_dot_bracket(dot_bracket_str):
    stack = []
    pairs = []
    for i, char in enumerate(dot_bracket_str):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))  # (左括号索引, 右括号索引)
    return pairs

# 侧链原子索引映射
IDX_TO_ATOM = {
    0: 'C4', 1: 'N1', 2: 'C2', 3: 'N3', 4: 'C5', 5: 'C6',
    6: 'N6', 7: 'N7', 8: 'C8', 9: 'N2', 10: 'O6', 11: 'O2',
    12: 'O4', 13: 'N4'
}

# 每种碱基“应该存在”的合法侧链原子
RESD_SIDE_CHAIN_ATOM_NAME = {
    'A': {'C2', 'C4', 'C5', 'C6', 'C8', 'N1', 'N3', 'N6', 'N7'},
    'G': {'C2', 'C4', 'C5', 'C6', 'C8', 'N1', 'N2', 'N3', 'N7', 'O6'},
    'U': {'C2', 'C4', 'C5', 'C6', 'N3', 'O2', 'O4'},
    'C': {'C2', 'C4', 'C5', 'C6', 'N3', 'N4', 'O2'}
}

def export_full_rna_pdb(coords_coarse, pred_coords, seq_tensor, out_filepath, clashing_pairs=None):
    num2letter = {0: 'A', 1: 'U', 2: 'G', 3: 'C'}
    os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

    # ==================== 1. 写入 PDB ====================
    with open(out_filepath, 'w') as f:
        atom_serial = 1
        for i in range(pred_coords.shape[0]):
            res_idx = int(seq_tensor[i].item())
            res_name = num2letter.get(res_idx, 'A')
            res_seq_num = i + 1
            chain_id = 'A'

            # 骨架锚点 (C4', C1', N)
            bb_names = ["C4'", "C1'", "N9"] if res_name in ['A', 'G'] else ["C4'", "C1'", "N1"]
            for k, atom_name in enumerate(bb_names):
                x, y, z = coords_coarse[i, k].tolist()
                if math.isnan(x) or math.isnan(y) or math.isnan(z): continue
                if abs(x) < 1e-4 and abs(y) < 1e-4 and abs(z) < 1e-4: continue

                element = atom_name[0]
                atom_name_str = f" {atom_name:<3s}" if len(atom_name) < 4 else f"{atom_name:<4s}"
                f.write(
                    f"ATOM  {atom_serial:5d} {atom_name_str} {res_name:>3s} {chain_id}{res_seq_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00          {element:>2s}\n")
                atom_serial += 1

            # 侧链原子
            valid_atoms_for_res = RESD_SIDE_CHAIN_ATOM_NAME[res_name]
            for a in range(pred_coords.shape[1]):
                atom_name = IDX_TO_ATOM[a]
                if atom_name in valid_atoms_for_res:
                    x, y, z = pred_coords[i, a].tolist()
                    if math.isnan(x) or math.isnan(y) or math.isnan(z): continue

                    element = atom_name[0]
                    atom_name_str = f" {atom_name:<3s}" if len(atom_name) < 4 else f"{atom_name:<4s}"
                    f.write(
                        f"ATOM  {atom_serial:5d} {atom_name_str} {res_name:>3s} {chain_id}{res_seq_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00          {element:>2s}\n")
                    atom_serial += 1
        f.write("TER\nEND\n")

    # ==================== 2. 写入 PML 脚本 (顶级审美版) ====================
    pml_filepath = out_filepath.replace('.pdb', '.pml')
    with open(pml_filepath, 'w') as f:
        pdb_name = os.path.basename(out_filepath)
        f.write(f"load {pdb_name}, my_rna\n")
        f.write("hide all\n")
        f.write("set bond_sym_cutoff, 0.8\n")

        # 强制连键逻辑 (保持不变)
        seq_len = pred_coords.shape[0]
        for i in range(1, seq_len):
            f.write(f"bond my_rna and resi {i} and name C4', my_rna and resi {i + 1} and name C4'\n")
        for i in range(1, seq_len + 1):
            f.write(f"bond my_rna and resi {i} and name C4', my_rna and resi {i} and name C1'\n")
            f.write(f"bond my_rna and resi {i} and name C1', my_rna and resi {i} and name N9\n")
            f.write(f"bond my_rna and resi {i} and name C1', my_rna and resi {i} and name N1\n")

        # 🌟 基础显示与光影美颜
        f.write("show sticks, my_rna\n")
        f.write("util.cbac my_rna\n")  # 健康 RNA 使用清冷的青蓝色(Cyan)碳原子
        f.write("bg_color white\n")
        f.write("set stick_radius, 0.12\n")  # 键调得更细致一点
        f.write("set stick_ball, on\n")  # 开启圆滑关节
        f.write("set stick_ball_ratio, 1.2\n")  # 关节比例
        f.write("set specular, 0.1\n")  # 降低贼光，提升哑光高级感
        f.write("set ambient, 0.4\n")

        # 骨架样式
        f.write("color gray60, name C4'\n")
        f.write("color gray60, name C1'\n")
        f.write("set_bond stick_radius, 0.04, my_rna and name C4', my_rna and name C4'\n")
        f.write("set_bond stick_color, gray80, my_rna and name C4', my_rna and name C4'\n")

        # 🌟 ==================== 高级碰撞高亮 ==================== 🌟
        if clashing_pairs and len(clashing_pairs) > 0:
            clash_residues = set()
            for (res_i, res_j) in clashing_pairs:
                clash_residues.add(res_i)
                clash_residues.add(res_j)

            resi_str = "+".join(str(idx) for idx in sorted(list(clash_residues)))

            f.write("\n# --- Highlight Clashes ---\n")
            f.write(f"select bad_clashes, my_rna and resi {resi_str}\n")

            # 1. 碰撞残基的碳原子涂成高级的鲑鱼粉/警示红
            f.write("color salmon, bad_clashes and elem C\n")

            # 2. 显示球体，并进行“瘦身”和半透明处理
            f.write("show spheres, bad_clashes\n")
            f.write("set sphere_scale, 0.45, bad_clashes\n")  # 核心！把球缩小到 45%
            f.write("set sphere_transparency, 0.5, bad_clashes\n")  # 柔和的半透明

            # 聚焦并稍微拉远一点点，留出留白
            f.write("zoom bad_clashes, buffer=3.0\n")
        else:
            f.write("zoom my_rna\n")

        # 顺便去个选中状态的粉色小点
        f.write("deselect\n")