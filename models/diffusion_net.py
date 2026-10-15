import torch
import torch.nn as nn
from models.dit import SinusoidalPositionalEmbedding, TransformerEncoderCondLayer
from models.gvp_pytorch import GVP, GVPConvLayer
from models.layers import LearnedSinusoidalPosEmb
from models.utils import LayerNorm


class DiffusionDiT(nn.Module):
    """
    Standard DiT implementation for Inverse Folding.
    Input: Noisy Latent Z_t
    Condition: Backbone Structure (via GVP) + Time t
    Interaction: Adaptive Layer Norm (Modulation)
    """

    def __init__(self, config):
        super().__init__()
        # === Configs ===
        self.use_ss_feature = config.get('use_ss_feature', False)
        original_edge_in_dim = list(config["edge_in_dim"])  # 例如 [32, 1]
        if self.use_ss_feature:
            original_edge_in_dim[0] += 1  # 增加 1 维 (is_ss 标识符)，变成 [33, 1]
            print("🚀 [Ablation Mode] SS feature injection is ENABLED. Edge dim +1.")
        else:
            print("🛑 [Standard Mode] SS feature injection is DISABLED.")

        self.edge_in_dim = tuple(original_edge_in_dim)  # [32, 1]
        self.edge_h_dim = config["edge_h_dim"]  # [32, 1]
        self.node_in_dim = config["node_in_dim"]  # [15, 3]
        self.node_h_dim = config["node_h_dim"]  # [128, 16] (Scalar, Vector)

        self.latent_dim = config["latent_dim"]  # 64
        self.hidden_dim = self.node_h_dim[0]  # 128 (Transformer Hidden Size)

        # 确保 Transformer 维度和 GVP Scalar 输出维度一致，方便相加
        assert config["trans"]["encoder_embed_dim"] == self.hidden_dim, \
            f"Transformer dim {config['trans']['encoder_embed_dim']} must match GVP scalar dim {self.hidden_dim}"

        # =====================================================================
        # 1. Condition Encoder: GVP (只看 Backbone)
        # =====================================================================
        self.W_v = nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim, activations=(None, None), vector_gate=True)
        )
        self.W_e = nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim, activations=(None, None), vector_gate=True)
        )

        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(self.node_h_dim, self.edge_h_dim, drop_rate=config["drop_rate"])
            for _ in range(config["num_layers"])
        )

        # GVP 输出层：只输出 Scalar 特征作为 Condition Embedding
        # Output: [N, hidden_dim]
        self.W_out = GVP(self.node_h_dim, (self.hidden_dim, 0), activations=(None, None))

        # =====================================================================
        # 2. Time Encoder
        # =====================================================================
        # DiT style: Frequency Embedding -> MLP
        self.time_mlp = nn.Sequential(
            LearnedSinusoidalPosEmb(16),  # Learnable freq
            nn.Linear(17, self.hidden_dim),  # Proj
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)  # Output matches hidden_dim
        )

        # =====================================================================
        # 3. Input Projection (Latent -> Hidden)
        # =====================================================================
        # 负责把 noisy z_t (64) 映射到 Transformer 空间 (128)
        self.input_proj = nn.Linear(self.latent_dim, self.hidden_dim)

        # Positional Embedding (standard sinusoidal)
        self.embed_positions = SinusoidalPositionalEmbedding(self.hidden_dim, -1)

        # =====================================================================
        # 4. DiT Backbone (Modulated Transformer)
        # =====================================================================
        # 这里的 TransformerLayer 必须支持 'cond' 输入并执行 modulation
        self.trans_layers = nn.ModuleList(
            TransformerEncoderCondLayer(config["trans"])
            for _ in range(config["num_trans_layer"])
        )

        # =====================================================================
        # 5. Final Output Head (AdaLN-Zero style final norm + proj)
        # =====================================================================
        # 标准 DiT 在最后输出前也有一个 modulation
        self.final_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_cond_proj = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)  # shift, scale
        self.final_linear = nn.Linear(self.hidden_dim, self.latent_dim)  # Predict noise

        # 对于 z_0 预测，使用标准初始化比零初始化更好，因为 z_0 不接近 0
        nn.init.xavier_uniform_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def forward(self, data, t):
        """
        data: PyG Batch
          - data.x: Noisy Latent Z_t [N_total, 64]
          - data.node_v, extra_x: Backbone Features
          - data.batch: Batch index
        t: Timesteps [Batch, 1]
        """
        # ==========================================
        # Step 1: Prepare Inputs & Shapes
        # ==========================================
        batch_size = t.shape[0]
        # 假设 Batch 内图大小一致或已处理 (Standard DiT 假设 tensor 形式 [B, L, D])
        # 这里我们需要处理 PyG 的 Flat Batch [N_total, D] -> [B, L, D]
        # 为了严谨，我们先计算 max_len 并 pad，或者简化假设每个图长度一样
        # 简化假设：所有 RNA 长度一样，或者我们在此处 reshape
        max_len = data.x.shape[0] // batch_size

        z_t_flat = data.x

        # ==========================================
        # Step 2: Encode Condition (GVP) -> Structure S
        # ==========================================
        # GVP 只看几何信息 (extra_x, node_v)，绝对不看 z_t
        h_V = (data.extra_x, data.node_v)
        h_E = (data.edge_attr, data.edge_v)

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.encoder_layers:
            h_V = layer(h_V, data.edge_index, h_E)

        # struct_embed: [N_total, hidden_dim]
        # 这是纯净的骨架特征
        struct_embed_flat = self.W_out(h_V)

        #print("struct_embed_flat: ", struct_embed_flat.shape)

        # ==========================================
        # Step 3: Encode Time -> T
        # ==========================================
        # t_emb: [Batch, hidden_dim]
        t_emb = self.time_mlp(t.squeeze(1))

        # Expand Time to [N_total, hidden_dim] (align with flat nodes)
        # 也就是把全局时间 t 广播给每个残基
        t_emb_expanded = t_emb[data.batch]

        # ==========================================
        # Step 4: Construct Modulation Condition
        # ==========================================
        # DiT 的核心：Condition = Structure + Time
        # 我们把结构特征和时间特征相加，作为一个统一的 conditioning vector
        # cond_flat: [N_total, hidden_dim]
        cond_flat = struct_embed_flat + t_emb_expanded

        # ==========================================
        # Step 5: Input Processing (Main Stream)
        # ==========================================
        # 将 Latent Z_t 映射到 Transformer 空间
        x_flat = self.input_proj(z_t_flat)  # [N_total, 128]

        # Reshape to [Batch, Length, Hidden] for Transformer
        # 注意：这里隐式假设数据是规整的。如果是变长，需要 padding mask
        x = x_flat.reshape(batch_size, max_len, -1)
        cond = cond_flat.reshape(batch_size, max_len, -1)

        pos_emb_input = torch.zeros((batch_size, max_len), dtype=torch.long, device=x.device)

        # 2. 获取位置编码
        pos_emb = self.embed_positions(pos_emb_input)
        x = x + pos_emb

        # ==========================================
        # Step 6: DiT Blocks (Modulation)
        # ==========================================
        # x: Main Stream
        # cond: Condition Stream (用来生成 shift, scale)

        # Transformer 期望输入: [Length, Batch, Hidden]
        x = x.transpose(0, 1)
        cond = cond.transpose(0, 1)

        for layer in self.trans_layers:
            # [关键] 调用 DiT Block
            # 这里对应 dit.py 里的 TransformerEncoderCondLayer
            # 它内部会做: shift, scale = mlp(cond); x = x * (1+scale) + shift
            x = layer(x, encoder_padding_mask=None, cond=cond)

        x = x.transpose(0, 1)  # [Batch, Length, Hidden]
        cond = cond.transpose(0, 1)

        # ==========================================
        # Step 7: Final Modulation & Output
        # ==========================================
        # DiT 最后的标准操作：AdaLN-Zero + Linear
        # cond 依然是 (Structure + Time)

        # 1. 计算最后的 shift, scale
        shift, scale = self.final_cond_proj(cond).chunk(2, dim=-1)

        # 2. Modulate
        x = self.final_norm(x)
        x = x * (1 + scale) + shift

        # 3. Predict Noise
        pred = self.final_linear(x)  # [Batch, Length, Latent_Dim]

        # Flatten back to [N_total, Latent_Dim]
        return pred.reshape(-1, self.latent_dim)