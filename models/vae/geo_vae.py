import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from .layer import (
    RotationAwareEncodeLayer,
    RotationAwareDecodeLayer,
    LatentProjectionLayer,
    BaselineEncodeLayer,
    BaselineDecodeLayer,
    SequenceOnlyDecodeLayer,
    FourierFiLMEncodeLayer,
    FourierFiLMDecodeLayer,
)
from models.structure_builder import StructureBuilder

class GeoVAE(nn.Module):
    def __init__(self, model_config):
        super(GeoVAE, self).__init__()
        self.d_model = model_config["d_model"]
        self.latent_dim = model_config["latent_dim"]
        self.llm_embedding_dim = model_config["llm_embedding_dim"]
        self.num_head = model_config["num_head"]
        self.dropout = model_config["dropout"]
        self.num_layers = model_config["num_layers"]

        self.seq_emb = nn.Embedding(model_config['vocab_size'], self.d_model)

        self.representation_mode = model_config.get("representation_mode", "so2")

        if self.representation_mode == "so2":
            print(">>> Representation mode: SO(2) sequence-torsion <<<")
            self.rotation_encoder_layer = RotationAwareEncodeLayer(self.d_model)
            self.rotation_decoder_head = RotationAwareDecodeLayer(self.d_model)

        elif self.representation_mode == "concat":
            print(">>> Representation mode: Trig-Concat <<<")
            self.rotation_encoder_layer = BaselineEncodeLayer(self.d_model)
            self.rotation_decoder_head = BaselineDecodeLayer(self.d_model)

        elif self.representation_mode == "sequence_only":
            print(">>> Representation mode: Sequence-only <<<")
            self.rotation_encoder_layer = None
            self.rotation_decoder_head = SequenceOnlyDecodeLayer(self.d_model)

        elif self.representation_mode == "fourier_film":
            print(">>> Representation mode: Fourier-conditioned FiLM <<<")

            num_frequencies = model_config.get(
                "film_num_frequencies",
                4,
            )

            self.rotation_encoder_layer = FourierFiLMEncodeLayer(
                d_model=self.d_model,
                num_frequencies=num_frequencies,
            )
            self.rotation_decoder_head = FourierFiLMDecodeLayer(
                d_model=self.d_model,
                num_frequencies=num_frequencies,
            )

        else:
            raise ValueError(
                f"Unknown representation_mode: {self.representation_mode}"
            )

        self.repa_layer = nn.Sequential(
            nn.Linear(self.llm_embedding_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.latent_dim),
            nn.LayerNorm(self.latent_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_head,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # 从embedding空间到latent
        self.encoder_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU()
        )

        self.fc_mu = nn.Linear(self.d_model, self.latent_dim)
        self.fc_var = nn.Linear(self.d_model, self.latent_dim)

        #从latent空间到embedding空间
        self.z_to_d = LatentProjectionLayer(self.latent_dim, self.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.num_head,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_layers
        )


        ## 进行repa 对齐
        self.repa_proj = nn.Sequential(
            nn.Linear(model_config['llm_embedding_dim'], self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.latent_dim),
            nn.LayerNorm(self.latent_dim)  # 归一化输出，让 Teacher 信号更稳定
        )

        ## Physics Engine (Differentiable)
        self.structure_builder = StructureBuilder(model_config['templates_path'])
        # 冻结 StructureBuilder，确保其只作为不可学习的物理规则
        for p in self.structure_builder.parameters():
            p.requires_grad = False

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def encode(self, batch):
        # 1. Feature Extraction (Flat [N_total, D])
        h_seq = self.seq_emb(batch.seq_idx)

        ## 不加torsion angle的 abalation
        if self.representation_mode == "sequence_only":
            h_geo = h_seq
        else:
        ## 加torsion angle，但是融合模块不相同的模块
            # 物理纠缠：将序列与几何角度融合
            if self.training:
                noise = torch.randn_like(batch.angles_gt) * 0.1
                noisy_angles = batch.angles_gt + noise
                # 重新归一化到圆上
                noisy_angles = F.normalize(noisy_angles, p=2, dim=-1)
                h_geo = self.rotation_encoder_layer(h_seq, noisy_angles)
            else:
                h_geo = self.rotation_encoder_layer(h_seq, batch.angles_gt)

        # 2. To Dense Batch [B, L_max, D]
        x, mask = to_dense_batch(h_geo, batch.batch)

        # 3. Transformer Encoding
        # src_key_padding_mask: True 表示 padding (需要被忽略)
        padding_mask = ~mask

        # Transformer 输出 [B, L_max, D]
        memory = self.encoder(x, src_key_padding_mask=padding_mask)

        # 4. Project to Latent
        mu = self.fc_mu(memory)  # [B, L, latent_dim]
        logvar = self.fc_var(memory)  # [B, L, latent_dim]

        # 返回分布参数和 mask，供 Stage 2 Dataloader 或 Loss 使用
        return mu, logvar, mask

    def decode(self, z, mask, coords_coarse_flat=None):
        # 1. Expand z to d_model
        z_projected = self.z_to_d(z)  # [B, L, D]

        padding_mask = ~mask

        out_dense = self.decoder(
            z_projected,
            z_projected,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask
        )

        # 只取出 mask 为 True 的有效部分，以便通过 Linear 层和 StructureBuilder
        out_flat = out_dense[mask]

        seq_logits, pred_angles = self.rotation_decoder_head(out_flat)

        # 5. Physics Construction (StructureBuilder)
        pred_coords = None
        if (pred_angles is not None and coords_coarse_flat is not None):
            # 必须保证 coords_coarse_flat 的长度等于 mask.sum()
            # 使用 Softmax 保证梯度回传
            seq_soft = F.softmax(seq_logits, dim=-1)
            pred_coords, _ = self.structure_builder(seq_soft, pred_angles, coords_coarse_flat)

        return seq_logits, pred_angles, pred_coords

    def forward(self, batch):
        # --- A. Encode (Student Stream) ---
        mu, logvar, mask = self.encode(batch)

        # Reparameterize (Dense)
        z = self.reparameterize(mu, logvar)

        # 直接传入 flat coords 用于计算物理 Loss
        seq_logits, pred_angles, pred_coords = self.decode(z, mask, batch.coords_coarse)

        # 1. 获取 Teacher 特征 (Flat) 并 Detach
        with torch.no_grad():
            teacher_feat_flat = batch.llm_opt.detach()  # [N_total, embedding_dim]

        # 2. 转为 Dense 以便和 z (Dense) 对齐计算 Loss
        teacher_feat_dense, _ = to_dense_batch(teacher_feat_flat, batch.batch)
        #print("teacher_feat_dense: ", teacher_feat_dense.shape)

        # 3. 投影到 Latent 空间
        z_teacher = self.repa_proj(teacher_feat_dense)  # [B, L, latent]

        return {
            "seq_logits": seq_logits,  # Flat [N_total, 4]
            "pred_angles": pred_angles,  # Flat [N_total, 2]
            "pred_coords": pred_coords,  # Flat [N_total, 14, 3]

            "mu": mu,  # Dense [B, L, latent]
            "logvar": logvar,  # Dense [B, L, latent]
            "mask": mask,  # Dense [B, L]

            "z_student": mu,  # Dense (用 mu 对齐比用 z 更稳)
            "z_teacher": z_teacher  # Dense
        }