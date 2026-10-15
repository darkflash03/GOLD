import torch
import torch.nn as nn
from models.diffusion.latent_transition import LatentTransition
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from utils.guide import compute_semantic_guidance, compute_clash_guidance

class DiffusionModel(nn.Module):
    def __init__(self, model, latent_transition_config):
        super(DiffusionModel, self).__init__()
        self.model = model  # DiffusionDiT
        self.latent_transition = LatentTransition(**latent_transition_config)
        self.num_steps = self.latent_transition.num_steps

    def forward(self, data, t=None):
        """
        Training Forward Pass
        Input: data (PyG Batch with clean latent in data.x)
        Output: pred_latent (predicted x_0), t, mask(None)
        """
        # 1. 随机采样时间步 t
        # data.num_graphs 获取 batch size
        batch_size = data.num_graphs
        if t is None:
            t = torch.randint(0, self.num_steps, (batch_size, 1), device=data.x.device)

        # 2. 扩展 t 到每个节点 (Flat Batch 适配)
        # data.batch 是 [N_nodes] 的索引，例如 [0,0,0, 1,1,1...]
        t_expand = t[data.batch].squeeze(-1)  # [N_nodes]

        # 3. 加噪 (x_0 -> x_t)
        clean_latent = data.x
        noisy_x = self.latent_transition.add_noise(clean_latent, t_expand)

        # 4. 构造 Noisy Data
        noisy_data = data.clone()
        noisy_data.x = noisy_x

        # 5. 模型预测 (Predict x_0)
        # 注意：model forward 需要接受 (data, t)
        pred_latent = self.model(noisy_data, t)

        return pred_latent, t

    @torch.no_grad()
    def ddpm_sample(self, data):
        """
        DDPM Inference
        data: 包含骨架信息的 Batch (data.x 可以是随机噪声或占位符)
        """
        # 初始化纯噪声
        N_nodes = data.x.shape[0]
        latent_dim = self.model.latent_dim

        num_graphs = data.batch[-1] + 1
        latent_t = torch.randn((N_nodes, latent_dim), device=data.x.device, dtype=torch.float32)

        # 反向去噪循环
        for t in range(self.num_steps - 1, -1, -1):
            # 构造 Batch 时间步
            sample_t = torch.full((num_graphs, 1), t, dtype=torch.long, device=data.x.device)

            # 构造模型输入
            noisy_data = data.clone()
            noisy_data.x = latent_t

            # 预测 x_0
            pred_latent_t0 = self.model(noisy_data, sample_t)

            # 状态更新 x_t -> x_{t-1}
            latent_t = self.latent_transition.ddpm_step(latent_t, pred_latent_t0, t)

        return latent_t

    @torch.no_grad()
    def ddim_sample(self, data, num_steps=None, eta=0.0):
        if num_steps is None:
            num_steps = self.num_steps

        # 训练时 t \in [0, num_steps-1]
        times = torch.linspace(self.num_steps - 1, 0, num_steps).long().to(data.x.device)

        # 初始化噪声
        N_nodes = data.x.shape[0]
        latent_dim = self.model.latent_dim
        latent_t = torch.randn((N_nodes, latent_dim), device=data.x.device, dtype=torch.float32)

        # 获取 Batch Size (图的数量)
        if hasattr(data, 'num_graphs'):
            num_graphs = data.num_graphs
        else:
            num_graphs = data.batch.max().item() + 1

        # DDIM 逆过程
        for i in range(len(times)):
            t = times[i]  # 当前步，例如 49

            # 计算下一步 s
            if i == len(times) - 1:
                s = -1  # 结束标记
            else:
                s = times[i + 1]  # 下一步，例如 48

            sample_t = torch.full((num_graphs, 1), t.item(), dtype=torch.long, device=data.x.device)

            # 构造模型输入
            noisy_data = data.clone()
            noisy_data.x = latent_t

            # 预测 x_0
            pred_latent_t0 = self.model(noisy_data, sample_t)

            # 避免最后再加回一次 sigma 噪声，提高结构准确度
            if t == 0:
                return pred_latent_t0

            # DDIM 更新 (t -> s)
            latent_t = self.latent_transition.ddim_step(latent_t, pred_latent_t0, t.item(), s.item(), eta)

        return latent_t

    @torch.no_grad()
    def guide_ddim_sample(self, data, num_steps=None, eta=0.0, vae=None, latent_mean=None, latent_std=None, ss_pairs=None, guidance_config=None):
        if num_steps is None:
            num_steps = self.num_steps

        if guidance_config is None:
            guidance_config = {}
        w_secondary_structure = guidance_config.get('w_secondary_structure', 0.0)
        w_clash = guidance_config.get('w_clash', 0.0)

        need_ss_guidance = (
                w_secondary_structure > 0
                and ss_pairs is not None
                and len(ss_pairs) > 0
        )
        need_clash_guidance = w_clash > 0

        vae_mode = getattr(vae, "representation_mode", "so2")

        if vae_mode == "sequence_only" and need_clash_guidance:
            raise ValueError(
                "sequence_only does not support clash guidance "
                "because it does not generate atomic coordinates."
            )

        if not need_ss_guidance and not need_clash_guidance:
            return self.ddim_sample(
                data,
                num_steps=num_steps,
                eta=eta
            )

        times = torch.linspace(self.num_steps - 1, 0, num_steps).long().to(data.x.device)

        N_nodes = data.x.shape[0]
        latent_dim = self.model.latent_dim
        latent_t = torch.randn((N_nodes, latent_dim), device=data.x.device, dtype=torch.float32)

        num_graphs = data.batch.max().item() + 1

        for i in range(len(times)):
            t = times[i]
            s = -1 if i == len(times) - 1 else times[i + 1]
            sample_t = torch.full((num_graphs, 1), t.item(), dtype=torch.long, device=data.x.device)

            with torch.enable_grad():
                latent_t = latent_t.detach().requires_grad_(True)
                noisy_data = data.clone()
                noisy_data.x = latent_t

                # 预测 x0
                pred_latent_t0 = self.model(noisy_data, sample_t)
                energy = torch.tensor(0.0, device=data.x.device)

                if t.item() > 0:
                    safe_std = torch.where(latent_std < 0.1, torch.tensor(1.0, device=data.x.device), latent_std)
                    z_denorm = pred_latent_t0 * safe_std + latent_mean
                    z_denorm = torch.clamp(z_denorm, min=-10.0, max=10.0)

                    # VAE 解码 (BS=1, mask_dense 就是全 True 或者直接通过)
                    z_dense, mask_dense = to_dense_batch(z_denorm, data.batch)
                    coords_for_decode = (
                        data.coords_coarse
                        if need_clash_guidance
                        else None
                    )

                    seq_logits, pred_angles, pred_coords = vae.decode(z_dense, mask_dense, coords_coarse_flat=coords_for_decode)

                    seq_logits_single = seq_logits  # [L, 4]

                    ##二级结构引导
                    if w_secondary_structure > 0:
                        seq_probs = F.softmax(seq_logits_single, dim=-1)
                        energy += w_secondary_structure * compute_semantic_guidance(seq_probs, ss_pairs)

                    #碰撞引导
                    if w_clash > 0:
                        if need_clash_guidance:
                            if pred_coords is None:
                                raise RuntimeError(
                                    f"VAE mode '{vae_mode}' does not produce coordinates, "
                                    "but clash guidance was requested."
                                )

                            mask_atoms_single = data.mask_gt.bool()
                            clash_energy = compute_clash_guidance(
                                pred_coords,
                                mask_atoms_single
                            )

                            energy = energy + w_clash * clash_energy

                    ##作用在latent 的梯度上进行引导
                    if energy > 0:
                        grad = torch.autograd.grad(outputs=energy, inputs=latent_t)[0]
                        scale = 1.0 * (t.item() / self.num_steps)  # 动态退火步长
                        latent_t_guided = latent_t - scale * grad
                    else:
                        latent_t_guided = latent_t

                else:
                    latent_t_guided = latent_t

            if t == 0:
                return pred_latent_t0.detach()

            latent_t = self.latent_transition.ddim_step(latent_t_guided.detach(), pred_latent_t0.detach(), t.item(), s.item(), eta)

        return latent_t