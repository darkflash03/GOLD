import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    def __init__(self, train_config):
        super(VAELoss, self).__init__()
        self.weights = train_config['weights']

    def forward(self, output, batch):
        w = self.weights
        # 1. Flat Losses (直接对齐 Valid Nodes)
        l_seq = F.cross_entropy(output['seq_logits'], batch.seq_idx)
        zero = l_seq.new_zeros(())

        # Angle loss
        l_angle = zero
        if output.get("pred_angles") is not None:
            l_angle = F.mse_loss(output['pred_angles'], batch.angles_gt)

        # Coordinate loss
        l_coord = zero
        if output.get("pred_coords") is not None:
            mask_atoms = batch.mask_gt.bool()
            l_coord = F.mse_loss(output['pred_coords'][mask_atoms], batch.coords_gt[mask_atoms])

        # 2. Dense Losses (需要处理 Padding Mask)
        mask = output['mask']  # [B, L]

        # KL Divergence
        # kld_per_token: [B, L]
        kld_per_token = -0.5 * torch.sum(1 + output['logvar'] - output['mu'].pow(2) - output['logvar'].exp(), dim=-1)
        # 只取有效 token 的平均
        l_kl = (kld_per_token * mask).sum() / mask.sum().clamp_min(1)

        # --- RepA Alignment: Cosine ---
        l_repa = zero
        # 取出 Student 和 Teacher 的向量
        if (output.get("z_teacher") is not None and output.get("z_student") is not None):
            z_s = output['z_student']  # [B, L, D]
            z_t = output['z_teacher']  # [B, L, D]

            # 计算 Cosine Similarity: [B, L]
            # dim=-1 表示沿着特征维度计算
            # eps=1e-6 防止除零
            cos_sim = F.cosine_similarity(z_s, z_t, dim=-1, eps=1e-6)

            # Loss = 1 - Cosine (范围 0~2, 0表示完全重合)
            repa_loss_per_token = 1.0 - cos_sim

            # Apply Mask
            l_repa = (repa_loss_per_token * mask).sum() / mask.sum().clamp_min(1)

        # # Weighted Sum
        # loss = (self.weights['w_seq'] * l_seq +
        #         self.weights['w_angle'] * l_angle +
        #         self.weights['w_coord'] * l_coord +
        #         self.weights['w_repa'] * l_repa +
        #         self.weights['w_kl'] * l_kl)

        # Do not use 0 * NaN.
        loss = w["w_seq"] * l_seq

        if w["w_angle"] != 0:
            if output.get("pred_angles") is None:
                raise ValueError(
                    "w_angle > 0 but pred_angles is None."
                )
            loss = loss + w["w_angle"] * l_angle

        if w["w_coord"] != 0:
            if output.get("pred_coords") is None:
                raise ValueError(
                    "w_coord > 0 but pred_coords is None."
                )
            loss = loss + w["w_coord"] * l_coord

        if w["w_repa"] != 0:
            loss = loss + w["w_repa"] * l_repa

        if w["w_kl"] != 0:
            loss = loss + w["w_kl"] * l_kl

        return loss, {
            "loss": loss.item(),
            "l_seq": l_seq.item(),
            "l_angle": l_angle.item(),
            "l_coord": l_coord.item(),
            "l_repa": l_repa.item(),
            "l_kl": l_kl.item()
        }