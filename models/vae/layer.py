import torch
import torch.nn as nn
import torch.nn.functional as F

class RotationAwareEncodeLayer(nn.Module):
    def __init__(self, d_model):
        super(RotationAwareEncodeLayer, self).__init__()
        self.d_model = d_model

    def forward(self, seq_embedding, angle):
        '''
        :param seq_embedding: [N, d_model] 序列语义嵌入
        :param angle: [N, 2] (sin, cos)
        :return:
        '''
        # 1. 准备旋转矩阵
        sin_a, cos_a = angle[:, 0:1], angle[:, 1:2]  # [N, 1]

        # 2. 将embedding拆成 [x,y]，应用se(2)的旋转平移作用在embedding上
        h_pairs = seq_embedding.view(seq_embedding.shape[0], -1, 2)
        x = h_pairs[..., 0]
        y = h_pairs[..., 1]

        # 3. 应用se(2)
        # x' = x*cos - y*sin
        # y' = x*sin + y*cos
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a

        # 4. 合并回原来的维度
        geo_embedding = torch.stack([x_rot, y_rot], dim=-1).view(seq_embedding.shape[0], self.d_model)
        return geo_embedding

class RotationAwareDecodeLayer(nn.Module):
    '''
    1. 序列 (Identity) 是 SE(2) 不变量 -> 从向量模长 (Magnitude) 解码
    2. 角度 (Torsion) 是 SE(2) 等变概 -> 从向量相位 (Phase) 解码
    '''

    def __init__(self, d_model):
        super(RotationAwareDecodeLayer, self).__init__()
        self.d_model = d_model

        # --- 1. Sequence Head (Invariant) ---
        # 输入维度减半，因为只取模长
        self.seq_mlp = nn.Sequential(
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 4)
        )  # 输出 'A', 'U', 'G', 'C'的概率

        # --- 2. Angle Head (Equivariant) ---
        # 保留相位信息，使用完整维度
        self.angle_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
        ) # [sin(a), cos(a)]

    def forward(self, z):
        # --- A. 几何预处理 ---
        # 将 z 视为复数向量场 [..., D/2, 2]
        z_complex = z.view(*z.shape[:-1], -1, 2)
        x, y = z_complex[..., 0], z_complex[..., 1]

        # --- B. 序列解码 (基于模长 - Magnitude) ---
        # 模长是旋转不变的： sqrt(x^2 + y^2)
        magnitude = torch.sqrt(x ** 2 + y ** 2 + 1e-6)  # [..., D/2]
        seq_logits = self.seq_mlp(magnitude)

        # --- C. 角度解码 (基于全特征 - Phase Sensitive) ---
        raw_vec = self.angle_mlp(z)

        # 稳健归一化：强制约束 sin^2 + cos^2 = 1
        pred_angles = F.normalize(raw_vec, p=2, dim=-1, eps=1e-6)
        return seq_logits, pred_angles

class LatentProjectionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LatentProjectionLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.layer(x)

class BaselineEncodeLayer(nn.Module):
    def __init__(self, d_model):
        super(BaselineEncodeLayer, self).__init__()
        self.d_model = d_model

        # 输入维度是 d_model (序列) + 2 (角度的 sin, cos)
        self.proj = nn.Sequential(
            nn.Linear(d_model + 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, seq_embedding, angle):
        '''
        :param seq_embedding: [N, d_model]
        :param angle: [N, 2]
        '''
        # 最天真的做法：直接在最后一个维度 Concatenate
        # x shape: [N, d_model + 2]
        x = torch.cat([seq_embedding, angle], dim=-1)

        # 强行通过 MLP 混合特征
        geo_embedding = self.proj(x)
        return geo_embedding

class BaselineDecodeLayer(nn.Module):
    def __init__(self, d_model):
        super(BaselineDecodeLayer, self).__init__()
        self.d_model = d_model

        # --- 1. Sequence Head ---
        # 没有 Magnitude 的分离，直接从完整 d_model 解码
        self.seq_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 4)
        )

        # --- 2. Angle Head ---
        # 没有 Phase 的概念，直接从完整 d_model 解码
        self.angle_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
        )

    def forward(self, z):
        # z shape: [..., d_model]
        # 直接由两个毫无隔离的 MLP 预测
        seq_logits = self.seq_mlp(z)
        raw_vec = self.angle_mlp(z)

        # 归一化约束 sin^2 + cos^2 = 1
        pred_angles = F.normalize(raw_vec, p=2, dim=-1, eps=1e-6)
        return seq_logits, pred_angles


## 新增的对比实验需要的decode，为了整体看torsion angle的作用
class SequenceOnlyDecodeLayer(nn.Module):
    """Decode nucleotide identity only, without torsion prediction."""

    def __init__(self, d_model: int):
        super().__init__()
        self.seq_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )

    def forward(self, z):
        seq_logits = self.seq_mlp(z)
        pred_angles = None
        return seq_logits, pred_angles

class FourierAngleEmbedding(nn.Module):
    """
    Convert an angle represented as [sin(chi), cos(chi)] into
    [sin(chi), cos(chi), ..., sin(K*chi), cos(K*chi)].

    This implementation avoids atan2 and its branch discontinuity.
    """

    def __init__(self, num_frequencies: int = 4):
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be >= 1")
        self.num_frequencies = num_frequencies

    def forward(self, angle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            angle: [..., 2], ordered as [sin(chi), cos(chi)]

        Returns:
            fourier_features: [..., 2 * num_frequencies]
        """
        if angle.shape[-1] != 2:
            raise ValueError(
                f"Expected angle with last dimension 2, got {angle.shape}"
            )

        # Ensure the input lies on the unit circle.
        angle = F.normalize(angle, p=2, dim=-1, eps=1e-6)

        sin_1 = angle[..., 0:1]
        cos_1 = angle[..., 1:2]

        sin_k = sin_1
        cos_k = cos_1
        features = []

        for _ in range(self.num_frequencies):
            features.extend([sin_k, cos_k])

            # Complex-number recurrence:
            # sin((k+1)chi) = sin(kchi)cos(chi) + cos(kchi)sin(chi)
            # cos((k+1)chi) = cos(kchi)cos(chi) - sin(kchi)sin(chi)
            next_sin = sin_k * cos_1 + cos_k * sin_1
            next_cos = cos_k * cos_1 - sin_k * sin_1
            sin_k, cos_k = next_sin, next_cos

        return torch.cat(features, dim=-1)


class AngleConditionedFiLM(nn.Module):
    """
    Residual FiLM conditioned on a periodic torsion representation:

        h_out = (1 + delta_gamma) * h + beta

    Zero initialization makes the module initially close to identity.
    """

    def __init__(self, d_model: int, num_frequencies: int = 4):
        super().__init__()
        self.d_model = d_model
        self.angle_embedding = FourierAngleEmbedding(num_frequencies)

        fourier_dim = 2 * num_frequencies
        self.to_gamma_beta = nn.Linear(fourier_dim, 2 * d_model)

        # Start from identity modulation:
        # delta_gamma = 0, beta = 0.
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(
        self,
        features: torch.Tensor,
        angle: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: [..., d_model]
            angle:    [..., 2], ordered as [sin(chi), cos(chi)]
        """
        if features.shape[:-1] != angle.shape[:-1]:
            raise ValueError(
                "Feature and angle leading dimensions must match: "
                f"{features.shape} vs. {angle.shape}"
            )

        angle_features = self.angle_embedding(angle)
        delta_gamma, beta = self.to_gamma_beta(angle_features).chunk(2, dim=-1)

        return (1.0 + delta_gamma) * features + beta


class FourierFiLMEncodeLayer(nn.Module):
    """
    Fuse nucleotide embedding and torsion using Fourier-conditioned FiLM.
    """

    def __init__(self, d_model: int, num_frequencies: int = 4):
        super().__init__()
        self.film = AngleConditionedFiLM(
            d_model=d_model,
            num_frequencies=num_frequencies,
        )

    def forward(
        self,
        seq_embedding: torch.Tensor,
        angle: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seq_embedding: [N, d_model]
            angle:         [N, 2], ordered as [sin(chi), cos(chi)]
        """
        return self.film(seq_embedding, angle)


class FourierFiLMDecodeLayer(nn.Module):
    """
    Decode torsion first, then use the predicted torsion to modulate
    latent features before sequence prediction.

    No ground-truth angle is used in the decoder.
    """

    def __init__(self, d_model: int, num_frequencies: int = 4):
        super().__init__()
        self.d_model = d_model

        # Predict torsion directly from the decoder feature.
        self.angle_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

        # Use predicted torsion to condition sequence features.
        self.film = AngleConditionedFiLM(
            d_model=d_model,
            num_frequencies=num_frequencies,
        )

        self.seq_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [..., d_model]

        Returns:
            seq_logits:  [..., 4]
            pred_angles: [..., 2], ordered as [sin(chi), cos(chi)]
        """
        raw_angle = self.angle_mlp(z)
        pred_angles = F.normalize(
            raw_angle,
            p=2,
            dim=-1,
            eps=1e-6,
        )

        # Do not detach pred_angles: sequence supervision should be able
        # to propagate through the torsion-conditioned fusion.
        seq_features = self.film(z, pred_angles)
        seq_logits = self.seq_mlp(seq_features)

        return seq_logits, pred_angles