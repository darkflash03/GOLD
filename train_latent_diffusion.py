import warnings
warnings.filterwarnings("ignore")

from utils import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dataset import *
from torch.utils.data import DataLoader
from models.vae import *
from models import DiffusionDiT, DiffusionModel
from ema_pytorch import EMA
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

def train_latent_diffusion(config):
    print("Training Latent-Diffusion model...")
    torch.multiprocessing.set_start_method('spawn')

    device = config['device']
    data_config = config['data_config']
    train_config = config['train_config']
    seed = config['seed']
    seeding(seed)

    vae_config = config['vae_config']
    latent_dim = vae_config['model_config']['latent_dim']
    result_folder = os.path.join(config['train_config']['output_dir'], f'latent_dim_{latent_dim}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_config(os.path.join(result_folder, 'config.yaml'), config)

    # Initialize TensorBoard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(result_folder, "runs", current_time)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    train_dataset = GraphDataset(
        data_path=data_config['train_data_path'],
        data_config=data_config
    )

    valid_dataset = GraphDataset(
        data_path=data_config['valid_data_path'],
        data_config=data_config
    )

    if train_config.get("sampler"):
        train_sampler = TrainSampler(
            train_dataset,
            batch_size=train_config['batch_size'],
            sample_mode=train_config["sample_mode"],
            max_squared_res=train_config["max_squared_res"],
        )
    else:
        train_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=8,
        collate_fn=train_dataset.collate_fn,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=valid_dataset.collate_fn,
    )

    # A. VAE (Frozen)
    print("Loading VAE...")
    vae = GeoVAE(config['vae_config']['model_config']).to(device)
    vae_ckpt = torch.load(config['vae_config']['ckpt_path'], map_location=device)
    if 'model_state_dict' in vae_ckpt:
        vae.load_state_dict(vae_ckpt['model_state_dict'])
    else:
        vae.load_state_dict(vae_ckpt)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    vae_mode = config['vae_config']['model_config'].get('representation_mode', 'so2')

    print(f"Loaded GeoVAE mode: {vae_mode}")
    print(f"Loaded GeoVAE checkpoint: {config['vae_config']['ckpt_path']}")

    assert (config['vae_config']['model_config']['latent_dim'] == config['model_config']['latent_dim']), "GeoVAE latent_dim and diffusion latent_dim must match."

    stats_path = os.path.join(os.path.dirname(config['vae_config']['ckpt_path']), 'vae_latent_stats.pt')
    if os.path.exists(stats_path):
        print(f"Loading latent stats from {stats_path}")
        stats = torch.load(stats_path, map_location=device)
        latent_mean = stats['mean'].to(device)
        latent_std = stats['std'].to(device)
    else:
        raise FileNotFoundError(f"Latent stats not found at {stats_path}. Please run calc_latent_stats.py first.")

    # B. Diffusion
    print("Initializing Diffusion...")

    model_config = config['model_config']
    use_ss_feature = data_config.get('use_ss_feature', False)
    model_config['use_ss_feature'] = use_ss_feature
    dit_model = DiffusionDiT(model_config).to(device)
    diffusion = DiffusionModel(dit_model, train_config['latent_transition_config']).to(device)
    print("diffusion: ", diffusion)

    ema = EMA(diffusion, beta=0.995, update_every=train_config["gradient_accumulate_every"]).to(device)
    optimizer = AdamW(diffusion.parameters(), lr=train_config["lr"], weight_decay=train_config['wd'])

    # --- Training Loop ---
    global_step = 0
    gradient_step = 0

    accum_step = 0
    accum_loss = 0
    accum_loss_latent = 0
    accum_loss_seq = 0
    accum_loss_coord = 0

    for epoch in range(train_config['epochs']):
        diffusion.train()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        epoch_loss = 0.0
        epoch_loss_latent = 0.0
        epoch_loss_seq = 0.0
        epoch_loss_coord = 0.0

        if epoch < 10:
            current_w_loss_seq = 0.0
            current_w_loss_coord = 0.0
        else:
            current_w_loss_seq = train_config['loss_config']['loss_seq']['weight']
            current_w_loss_coord = train_config['loss_config']['loss_coord']['weight']

        need_seq_loss = current_w_loss_seq > 0
        need_coord_loss = current_w_loss_coord > 0


        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            # --- A. Online VAE Encoding ---
            with torch.no_grad():
                mu, logvar, mask = vae.encode(batch)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                #z_raw = mu + eps * std  # Reparameterization
                z_raw = mu

                # Scale & Flatten for PyG
                safe_std = torch.where(latent_std < 0.1, torch.tensor(1.0, device=device), latent_std)
                z_norm_dense = (z_raw - latent_mean) / safe_std
                z_norm_dense = torch.clamp(z_norm_dense, min=-10.0, max=10.0)
                #z_norm_dense = z_raw
                # C. Flatten & Assign
                mask_bool = mask.bool()
                z_flat_norm = z_norm_dense[mask_bool]  # [N_total, D]

                batch.x = z_flat_norm.detach()

            # --- B. Diffusion Forward ---
            # pred_z0: [N_total, D], t: [Batch, 1]
            pred_z0, t = diffusion(batch)
            loss_latent = F.mse_loss(pred_z0, batch.x)
            loss_seq = torch.zeros((), device=device)
            loss_coord = torch.zeros((), device=device)


            # Decode (使用 Frozen VAE)
            # 注意：VAE 处于 eval() 模式，参数冻结，但输入的 grad 会回传给 z_dense -> pred_z0
            # 必须传入 coords_coarse 才能计算 structure loss
            if need_seq_loss or need_coord_loss:
                pred_z0_denorm = pred_z0 * safe_std + latent_mean
                z_dense, mask_dense = to_dense_batch(pred_z0_denorm, batch.batch)
                seq_logits, pred_angles, pred_coords = vae.decode(
                    z_dense, mask_dense, coords_coarse_flat=batch.coords_coarse
                )


            # --- C. Loss (Predict x0) ---
            if need_seq_loss:
                loss_seq = F.cross_entropy(seq_logits, batch.seq_idx)
            if need_coord_loss:
                if pred_coords is None:
                    raise ValueError(
                        "Coordinate loss is enabled, "
                        "but this GeoVAE does not output coordinates."
                    )
                mask_atoms = batch.mask_gt.bool()
                loss_coord = F.mse_loss(pred_coords[mask_atoms], batch.coords_gt[mask_atoms])

            loss = (train_config['loss_config']['loss_latent']['weight'] * loss_latent +
                    current_w_loss_seq * loss_seq +
                    current_w_loss_coord * loss_coord)

            # [关键] 异常检测与熔断
            if need_coord_loss:
                if torch.isnan(loss_coord) or torch.isinf(loss_coord):
                    print(f"\n[ALARM] NaN/Inf detected in Coord Loss at Epoch {epoch} Step {global_step}!")

                    # --- 现场取证 (打印导致崩溃的中间变量) ---
                    with torch.no_grad():
                        print("  >>> Debug Snapshot <<<")
                        # 1. 检查 Latent (归一化后的)
                        print(
                            f"  pred_z0 (Norm): min={pred_z0.min().item():.4f}, max={pred_z0.max().item():.4f}, mean={pred_z0.mean().item():.4f}")
                        # 2. 检查反归一化后的 Latent (进 VAE 的)
                        print(
                            f"  z_in_vae (Denorm): min={pred_z0_denorm.min().item():.4f}, max={pred_z0_denorm.max().item():.4f}")
                        # 3. 检查 Angle (StructureBuilder 的直接输入)
                        print(f"  pred_angles: min={pred_angles.min().item():.4f}, max={pred_angles.max().item():.4f}")
                        if torch.isnan(pred_angles).any():
                            print("  !!! pred_angles contains NaN !!!")
                        # 4. 检查 Coords (StructureBuilder 的输出)
                        print(f"  pred_coords: min={pred_coords.min().item():.4f}, max={pred_coords.max().item():.4f}")
                        if torch.isnan(pred_coords).any():
                            print("  !!! pred_coords contains NaN !!!")

                        # 5. 检查 Seq Logits (看看是不是 seq 预测得太离谱)
                        print(f"  seq_logits: min={seq_logits.min().item():.4f}, max={seq_logits.max().item():.4f}")

                    # --- 紧急避险 ---
                    print("  >>> Skipping Coord Loss for this batch (Set to 0.0) <<<")
                    loss_coord = torch.tensor(0.0, device=device)  # 设为0，切断梯度

                    # 这种情况下，总 Loss 只包含 latent 和 seq
                    # 注意：这里 seq_logits 如果也有 NaN，可能还需要进一步处理，但通常是 coord 先挂
                    loss = (train_config['loss_config']['loss_latent']['weight'] * loss_latent +
                            current_w_loss_seq * loss_seq)


            accum_loss_latent += loss_latent.item()
            if need_seq_loss:
                accum_loss_seq += loss_seq.item()
            if need_coord_loss:
                accum_loss_coord += loss_coord.item()
            accum_loss += loss.item()
            epoch_loss += loss.item()
            epoch_loss_latent += loss_latent.item()
            if need_seq_loss:
                epoch_loss_seq += loss_seq.item()
            if need_coord_loss:
                epoch_loss_coord += loss_coord.item()

            global_step += 1
            accum_step += 1
            loss.backward()

            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), train_config['clip_grad'])

            if accum_step % train_config['gradient_accumulate_every'] == 0:
                writer.add_scalar(f"Train-Step/loss", accum_loss / train_config['gradient_accumulate_every'], gradient_step)
                writer.add_scalar(f"Train-Step/loss_latent", accum_loss_latent / train_config['gradient_accumulate_every'], gradient_step)
                if need_seq_loss:
                    writer.add_scalar(f"Train-Step/loss_seq", accum_loss_seq / train_config['gradient_accumulate_every'], gradient_step)
                if need_coord_loss:
                    writer.add_scalar(f"Train-Step/loss_coord", accum_loss_coord / train_config['gradient_accumulate_every'], gradient_step)

                accum_loss = 0
                accum_loss_latent = 0
                accum_loss_seq = 0
                accum_loss_coord = 0
                gradient_step += 1
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item(), loss_latent=loss_latent.item(), loss_seq=loss_seq.item(), loss_coord=loss_coord.item())

        # --- Epoch End ---
        avg_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar("Train/Epoch_Loss", avg_loss, epoch)
        avg_loss_latent = epoch_loss_latent / len(train_dataloader)
        writer.add_scalar("Train/Epoch_Loss_Latent", avg_loss_latent, epoch)
        if need_seq_loss:
            avg_loss_seq = epoch_loss_seq / len(train_dataloader)
            writer.add_scalar("Train/Epoch_Loss_Seq", avg_loss_seq, epoch)
        if need_coord_loss:
            avg_loss_coord = epoch_loss_coord / len(train_dataloader)
            writer.add_scalar("Train/Epoch_Loss_Coord", avg_loss_coord, epoch)
        if need_seq_loss and need_coord_loss:
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Loss_Latent: {avg_loss_latent:.4f} | Loss_Seq: {avg_loss_seq:.4f} | Loss_Coord: {avg_loss_coord:.4f}")
        else:
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        ##validation
        val_seq_acc, val_rmsd = validate_latent_diffusion(
            diffusion, vae, valid_dataloader, device, latent_mean, latent_std, epoch,
            num_steps=train_config['val_num_steps']
        )

        writer.add_scalar("Val/Seq_Acc", val_seq_acc, epoch)
        if val_rmsd is not None:
            writer.add_scalar("Val/RMSD", val_rmsd, epoch)
            print(f"Validation: RMSD={val_rmsd:.3f}, SeqAcc={val_seq_acc:.2%}")
        else:
            print(f"Validation: RMSD=N/A, SeqAcc={val_seq_acc:.2%}")

        torch.save(
            {
                'model': diffusion.state_dict(),
                'ema': ema.state_dict()
            },
            os.path.join(result_folder, f'model_separate_epoch_{epoch}.pt')
        )

    writer.close()
    print("training complete ......")