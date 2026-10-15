import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from ema_pytorch import EMA
from torch_geometric.utils import to_dense_batch

def calc_gt_structure_metric(dataset, device, builder):
    batch_mses = []
    total_sq_diff = 0.0
    total_atoms = 0
    print("....... eval gt structure error ........")
    builder = builder.to(device)
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            example = dataset[i]['graph']
            example = example.to(device)
            seq_idx = example.seq_idx  # [N]
            angles_gt = example.angles_gt  # [N, 2]
            coords_coarse = example.coords_coarse  # [N, 3, 3]
            pred_coords, pred_mask = builder(seq_idx, angles_gt, coords_coarse)
            gt_coords = example.coords_gt  # [N, 14, 3]
            gt_mask = example.mask_gt.bool()  # [N, 14]

            # 只有当 (PDB里有这个原子) AND (模板里定义了这个原子) 时，才计算 Loss
            valid_mask = gt_mask & pred_mask.bool()
            if valid_mask.sum() == 0:
                continue

            # E. 计算误差
            # 提取有效坐标
            p_pred = pred_coords[valid_mask]
            p_gt = gt_coords[valid_mask]

            # MSE Loss (per atom)
            mse = F.mse_loss(p_pred, p_gt, reduction='mean')
            batch_mses.append(mse.item())

            # 累积用于计算全局 RMSD
            sq_diff = ((p_pred - p_gt) ** 2).sum().item()
            n_atoms = valid_mask.sum().item()

            total_sq_diff += sq_diff
            total_atoms += n_atoms

    global_mse = total_sq_diff / total_atoms
    global_rmsd = np.sqrt(global_mse)
    return global_mse, global_rmsd

def validate_vae(model, device, loader, loss_fn, dataset_name, epoch):
    model.eval()

    # --- 累积器 ---
    total_loss = 0.0
    loss_accum = {
        "l_seq": 0.0,
        "l_angle": 0.0,
        "l_coord": 0.0,
        "l_repa": 0.0,
        "l_kl": 0.0
    }

    # 序列准确率相关
    total_seq_correct = 0
    total_seq_len = 0

    # RMSD 相关
    total_sq_diff_coord = 0.0
    total_valid_atoms = 0

    num_batches = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Evaluating {dataset_name}", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            outputs = model(batch)

            # 2. Calculate VAE Losses
            loss, loss_dict = loss_fn(outputs, batch)

            # 累积 Loss
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k in loss_accum:
                    loss_accum[k] += v

            # 3. Calculate Sequence Recovery Rate
            # seq_logits: [N_total, 4] -> argmax -> [N_total]
            pred_seq = outputs['seq_logits'].argmax(dim=-1)
            gt_seq = batch.seq_idx

            # 简单准确率
            correct = (pred_seq == gt_seq).sum().item()
            total_seq_correct += correct
            total_seq_len += gt_seq.size(0)

            # 4. Calculate RMSD (Root Mean Square Deviation)
            # 只计算 mask_gt 为 True 的原子
            pred_coords_all = outputs.get("pred_coords", None)
            if pred_coords_all is not None:
                mask = batch.mask_gt.bool()
                if mask.any():
                    pred_coords = pred_coords_all[mask]  # [N_valid, 3]
                    gt_coords = batch.coords_gt[mask]  # [N_valid, 3]

                    # Sum of Squared Errors
                    sq_diff = (pred_coords - gt_coords).pow(2).sum().item()
                    total_sq_diff_coord += sq_diff
                    total_valid_atoms += mask.sum().item()

            num_batches += 1

            cur_loss = total_loss / num_batches
            pbar.set_postfix(loss=cur_loss)

    # --- 聚合指标 ---
    avg_loss = total_loss / num_batches
    avg_loss_dict = {k: v / num_batches for k, v in loss_accum.items()}

    # 序列恢复率
    seq_recovery = total_seq_correct / total_seq_len if total_seq_len > 0 else 0.0

    # 全局 RMSD = sqrt( sum(diff^2) / N_atoms )
    # 注意：这是所有原子的 RMSD，不是基于叠合(Superimposition)后的 RMSD。
    # 对于生成模型 Loss 监控，这个 "Un-aligned RMSD" 正是 $L_{coord}$ 的物理意义。
    global_rmsd = np.sqrt(total_sq_diff_coord / total_valid_atoms) if total_valid_atoms > 0 else None

    # --- 组装结果 ---
    metrics = {
        "loss_total": avg_loss,
        **avg_loss_dict,  # 展开 l_seq, l_angle...
        "seq_recovery": seq_recovery,
    }

    if global_rmsd is not None:
        metrics["rmsd"] = global_rmsd


    if global_rmsd is not None:
        geometry_msg = (
            f"Coord: {metrics['l_coord']:.4f} "
            f"(Native RMSD: {global_rmsd:.4f}A)"
        )
    else:
        geometry_msg = "Coord: N/A | Native RMSD: N/A"

    print(
        f"[{dataset_name} Epoch {epoch}] "
        f"Loss: {avg_loss:.4f} | "
        f"{geometry_msg} | "
        f"Seq Acc: {seq_recovery:.4%} | "
        f"KL: {metrics['l_kl']:.4f}"
    )
    return metrics


def validate_latent_diffusion(diffusion_model, vae, loader, device, latent_mean, latent_std, epoch, num_steps=50):
    """
    Stage 2 Validation: Sampling -> Decoding -> RMSD
    """
    diffusion_model.eval()
    vae.eval()

    vae_mode = getattr(vae, "representation_mode", "so2")
    supports_native_geometry = vae_mode in {"so2", "concat"}

    print(
        f"Validating GeoVAE mode={vae_mode}, "
        f"native_geometry={supports_native_geometry}"
    )

    total_samples = 0
    total_seq_correct = 0
    total_sq_diff_coord = 0.0
    total_valid_atoms = 0

    if isinstance(diffusion_model, EMA):
        model_to_eval = diffusion_model.ema_model
    else:
        model_to_eval = diffusion_model

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Validating Epoch {epoch}", leave=False):
            if batch is None: continue
            batch = batch.to(device)

            # 1. DDIM Sampling (得到 Scaled Latent)
            # batch.x 此时是 dummy noise，sample 内部会重新初始化
            z_sampled_norm = model_to_eval.ddim_sample(batch, num_steps=num_steps, eta=0.0)

            safe_std = torch.where(latent_std < 0.1, torch.tensor(1.0, device=device), latent_std)

            # 2. Reverse Scale Factor
            z_recon = z_sampled_norm * safe_std + latent_mean

            # 3. VAE Decode
            # Flat -> Dense for VAE
            z_dense, mask_dense = to_dense_batch(z_recon, batch.batch)

            #对 sequence only 的 geovae不进行坐标的解码
            coords_for_decode = (
                batch.coords_coarse
                if supports_native_geometry
                else None
            )

            seq_logits, pred_angles, pred_coords = vae.decode(
                z_dense, mask_dense, coords_coarse_flat=coords_for_decode
            )


            # 4. Metrics
            # Seq Accuracy
            pred_seq = seq_logits.argmax(dim=-1)
            correct = (pred_seq == batch.seq_idx).sum().item()
            total_seq_correct += correct
            total_samples += batch.seq_idx.size(0)

            # RMSD (Only valid atoms)
            if pred_coords is not None:
                mask = batch.mask_gt.bool()
                if mask.sum() > 0:
                    p_pred = pred_coords[mask]
                    p_gt = batch.coords_gt[mask]
                    sq_diff = (p_pred - p_gt).pow(2).sum().item()
                    total_sq_diff_coord += sq_diff
                    total_valid_atoms += mask.sum().item()

    avg_seq_acc = total_seq_correct / total_samples if total_samples > 0 else 0.0
    rmsd = (
        np.sqrt(total_sq_diff_coord / total_valid_atoms)
        if total_valid_atoms > 0
        else None
    )

    return avg_seq_acc, rmsd