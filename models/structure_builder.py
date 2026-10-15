import torch
import torch.nn as nn
from utils.rigid_utils import calc_angl_rot_tsl, merge_rot_tsl, calc_rot_tsl_batch

class StructureBuilder(nn.Module):
    def __init__(self, template_path="templates.pt"):
        """
        全原子结构构建器 (Differentiable)
        用于从 (序列, 角度, 骨架) 恢复全原子坐标。

        Args:
            template_path: 预计算好的模板文件路径
        """
        super().__init__()

        # 加载模板
        # content: {'pos': [4, 14, 3], 'mask': [4, 14]}
        if not torch.cuda.is_available():
            # 防止在没有 GPU 的机器上加载出错
            data = torch.load(template_path, map_location='cpu')
        else:
            data = torch.load(template_path)

        # 注册为 Buffer，这样会被自动移动到 model.device
        # templates: [4, 14, 3] -> (A, U, G, C)
        self.register_buffer('templates', data['pos'].float())
        self.register_buffer('template_masks', data['mask'].bool())

    def forward(self, seq, angles, coords_coarse):
        """
        前向传播重建坐标。

        Args:
            seq: [N] LongTensor (序列索引 0-3) 或 [N, 4] FloatTensor (Softmax概率)
                 N = Sum of lengths in batch
            angles: [N, 2] FloatTensor (sin, cos 或 cos, sin，需与训练一致)
            coords_coarse: [N, 3, 3] FloatTensor (C4', C1', N)

        Returns:
            pred_coords: [N, 14, 3] 重建的全原子坐标
            pred_mask: [N, 14] 对应的原子存在掩码
        """
        # 1. 获取局部模板 (Local Templates)
        #print("seq", seq.shape)
        if seq.dim() == 1 or seq.shape[-1] == 1:
            # Hard Mode: 输入是索引 [N]
            # [N, 14, 3]
            local_pos = self.templates[seq.long()]
            # [N, 14]
            local_mask = self.template_masks[seq.long()]
        else:
            # Soft Mode: 输入是概率 [N, 4] (用于采样或 Soft Loss)
            # [N, 4] x [4, 14, 3] -> [N, 14, 3]
            local_pos = torch.einsum('ni,ijk->njk', seq, self.templates)
            # Mask 也可以软化，或者取概率最大的
            # 这里简单处理：如果对应的 Base 概率 > 0.5 且模板里有该原子，则认为存在
            # 为了计算 Loss 方便，通常我们只关心 Ground Truth 存在的原子，
            # 所以这里的 pred_mask 更多是用于可视化。
            # 简单策略：用 argmax 决定 mask
            hard_ids = seq.argmax(dim=-1)
            local_mask = self.template_masks[hard_ids]

        # 2. 计算骨架坐标系 (Backbone Frame)
        # coords_coarse: [N, 3, 3] -> p1(C4'), p2(C1'), p3(N)
        p1 = coords_coarse[:, 0]
        p2 = coords_coarse[:, 1]
        p3 = coords_coarse[:, 2]

        rot_bb, trans_bb = calc_rot_tsl_batch(p1, p2, p3)

        # 3. 计算侧链旋转 (Torsion Rotation)
        # 输入 angles: [N, 2]
        # 注意：rigid_utils.calc_angl_rot_tsl 内部通常期望 input 是 [cos, sin]
        # 如果您的模型输出或 GT 是 [sin, cos]，在这里不需要变，因为我们之前验证过 pipeline 是通的。
        rot_tor, trans_tor = calc_angl_rot_tsl(angles)

        # 4. 合并变换 (Global Frame)
        # Global = R_bb * R_tor * Local + T
        rot_glob, trans_glob = merge_rot_tsl(rot_bb, trans_bb, rot_tor, trans_tor)

        # 5. 应用变换 (Apply Transform)
        # [N, 3, 3] x [N, 14, 3] -> [N, 14, 3]
        # rot_vec_mul: (R @ v.T).T
        # 手动 einsum 更稳健: R[n,i,j] * v[n,k,j] -> out[n,k,i] (注意转置关系)
        # 旋转矩阵通常是列向量基，所以 v_glob = R @ v_loc
        global_pos = torch.einsum('nij,nkj->nki', rot_glob, local_pos) + trans_glob.unsqueeze(1)

        return global_pos, local_mask