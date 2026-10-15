import torch
import torch.nn as nn
from models.diffusion.scheduler import NoiseScheduler

class LatentTransition(nn.Module):
    def __init__(self, num_steps):
        super(LatentTransition, self).__init__()
        self.noise_scheduler = NoiseScheduler(num_steps)
        self.num_steps = num_steps

    @torch.no_grad()
    def add_noise(self, latent_t0, t):
        """
        给 x_0 加噪得到 x_t
        latent_t0: [N, D] (Clean)
        t: [N] (每个节点对应的时间步)
        """
        # 获取 alpha_bar 并调整形状以适配 [N, D]
        # t 应该是 [N] 或者 [Batch] 扩展后的
        alpha_bar = self.noise_scheduler.alpha_bars.to(latent_t0.device)[t]

        alpha_bar = alpha_bar.to(dtype=latent_t0.dtype)

        # 确保 alpha_bar 是 [N, 1] 以便广播
        if alpha_bar.dim() == 1:
            alpha_bar = alpha_bar.unsqueeze(-1)

        c0 = torch.sqrt(alpha_bar)
        c1 = torch.sqrt(1 - alpha_bar)

        e_rand = torch.randn_like(latent_t0)
        latent_noisy = c0 * latent_t0 + c1 * e_rand
        return latent_noisy

    @torch.no_grad()
    def ddpm_step(self, latent_t, pred_latent_t0, t):
        """
        DDPM 逆向采样步 (基于预测的 x_0)
        Formula: posterior mean mu_tilde(x_t, x_0)
        """
        # 获取系数
        alpha_t = self.noise_scheduler.alphas[t].clamp_min(
            self.noise_scheduler.alphas[-2]
        )
        alpha_bar_t = self.noise_scheduler.alpha_bars[t]
        alpha_bar_s = self.noise_scheduler.alpha_bars[t - 1]
        sigma = self.noise_scheduler.sigmas[t]  # Scalar usually

        z = torch.randn_like(latent_t)

        latent_s = (torch.sqrt(alpha_t) * (1 - alpha_bar_s) * latent_t +
                    torch.sqrt(alpha_bar_s) * (1 - alpha_t) * pred_latent_t0) / \
                   (1 - alpha_bar_t + 1e-8) + sigma * z
        return latent_s

    @torch.no_grad()
    def ddim_step(self, latent_t, pred_latent_t0, t, s, eta):
        """
        DDIM 单步去噪。
        Args:
            latent_t: 当前时间步的隐变量 x_t
            pred_latent_t0: 模型预测的 x_0
            t: 当前时间步
            s: 下一个时间步
            eta: 控制随机性的参数，eta=1的随机分量等同于ddpm，eta=0则是完全确定的路径
        Returns:
            latent_s: 下一个时间步的隐变量 x_s
        """
        alpha_bar_t = self.noise_scheduler.alpha_bars[t]
        alpha_bar_s = self.noise_scheduler.alpha_bars[s]
        sigma_t = eta * torch.sqrt((1 - alpha_bar_s) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_s))

        # DDIM 更新公式
        pred_noise = (latent_t - torch.sqrt(alpha_bar_t) * pred_latent_t0) / torch.sqrt(1 - alpha_bar_t + 1e-8)
        latent_s = (torch.sqrt(alpha_bar_s) * pred_latent_t0 +
                    torch.sqrt(1 - alpha_bar_s - sigma_t ** 2) * pred_noise +
                    sigma_t * torch.randn_like(latent_t))

        return latent_s