import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils import *
from dataset import *
from models.vae import *
from models import *
from losses import VAELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_geo_vae(config):
    print("Training Geo-VAE model...")
    torch.multiprocessing.set_start_method('spawn')

    data_config = config['data_config']
    seed = config['seed']
    seeding(seed)
    device = config['device']

    model_config = config['model_config']
    train_config = config['train_config']
    encoder_decoder_name = model_config['name']
    latent_dim = model_config['latent_dim']

    result_folder = os.path.join(config['train_config']['output_dir'], encoder_decoder_name, f'latent_dim_{latent_dim}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_config(os.path.join(result_folder, 'config.yaml'), config)

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

    # builder = StructureBuilder(template_path=model_config['templates_path'])
    # global_mse, global_rmsd = calc_gt_structure_metric(valid_dataset, device, builder)
    # print(f"Global MSE: {global_mse}")
    # print(f"Global RMSD: {global_rmsd}")

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_config['batch_size'],
                                  shuffle=True,
                                  collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=train_config['batch_size'],
                                  shuffle=False,
                                  collate_fn=valid_dataset.collate_fn)

    # 4. Model & Optimization
    print("Initializing Model...")
    model = GeoVAE(model_config).to(device)
    loss_fn = VAELoss(train_config).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # Filter frozen structure_builder
        lr=train_config['lr'],
        weight_decay=train_config['weight_decay']
    )

    # Cosine Scheduler with Warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    accum_step = 0
    gradient_step = 0
    print("Start Training...")
    for epoch in range(train_config['epochs']):
        model.train()
        total_loss = 0
        accum_loss = 0
        total_seq_loss = 0
        total_angle_loss = 0
        total_coord_loss = 0
        total_repa_loss = 0
        total_kl_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{train_config['epochs']}")
        for step, batch in enumerate(pbar):
            batch = batch.to(device)
            # Forward
            outputs = model(batch)
            # Loss Calculation
            loss, loss_dict = loss_fn(outputs, batch)
            total_loss += loss.item()
            accum_loss += loss.item()
            total_seq_loss += loss_dict['l_seq']
            total_angle_loss += loss_dict['l_angle']
            total_coord_loss += loss_dict['l_coord']
            total_repa_loss += loss_dict['l_repa']
            total_kl_loss += loss_dict['l_kl']
            accum_step += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['clip_grad'])

            pbar.set_postfix(
                loss=loss.item(),
                seq_loss=loss_dict['l_seq'],
                angle_loss=loss_dict['l_angle'],
                coord_loss=loss_dict['l_coord'],
                repa_loss=loss_dict['l_repa'],
                kl_loss=loss_dict['l_kl']
            )

            if accum_step % train_config['gradient_accumulate_every'] == 0:
                writer.add_scalar("Train-Step/loss", accum_loss / train_config['gradient_accumulate_every'], gradient_step)
                gradient_step += 1
                accum_loss = 0
                optimizer.step()
                scheduler.step()

                optimizer.zero_grad()

        train_epoch_loss = total_loss / len(train_dataloader)
        writer.add_scalar(f'Train-Epoch/loss', train_epoch_loss, epoch)
        train_epoch_seq_loss = total_seq_loss / len(train_dataloader)
        writer.add_scalar(f'Train-Epoch/seq_loss', train_epoch_seq_loss, epoch)
        train_epoch_angle_loss = total_angle_loss / len(train_dataloader)
        writer.add_scalar(f'Train-Epoch/angle_loss', train_epoch_angle_loss, epoch)
        train_epoch_coord_loss = total_coord_loss / len(train_dataloader)
        writer.add_scalar(f'Train-Epoch/coord_loss', train_epoch_coord_loss, epoch)
        train_epoch_repa_loss = total_repa_loss / len(train_dataloader)
        writer.add_scalar(f"Train-Epoch/repa_loss", train_epoch_repa_loss, epoch)
        train_epoch_kl_loss = total_kl_loss / len(train_dataloader)
        writer.add_scalar(f"Train-Epoch/kl_loss", train_epoch_kl_loss, epoch)

        ## 开始validation
        val_metrics = validate_vae(model, device, valid_dataloader, loss_fn, "GOLD_VALID", epoch)
        for metric in val_metrics.keys():
            writer.add_scalar(f'Validation/{metric}', val_metrics[metric], epoch)

        # --- Checkpointing ---
        ckpt_path = os.path.join(result_folder, f'model_separate_epoch_{epoch}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, ckpt_path)

    writer.close()
    print("training complete ......")

