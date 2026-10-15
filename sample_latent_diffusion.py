import warnings
warnings.filterwarnings("ignore")

from utils import *
from dataset import *
from torch.utils.data import DataLoader
from models.vae import *
from models import DiffusionDiT, DiffusionModel
from tqdm import tqdm
from torch_geometric.utils import to_dense_batch
import pandas as pd
import time
from sklearn.metrics import f1_score


def sample_latent_diffusion(config):
    print("Sampling Latent-Diffusion model...")
    torch.multiprocessing.set_start_method('spawn')

    device = config['device']
    data_config = config['data_config']
    sample_config = config['sample_config']
    train_config = config['train_config']
    seed = config['seed']
    seeding(seed)

    w_secondary_structure = sample_config['guidance']['w_secondary_structure']
    w_clash = sample_config['guidance']['w_clash']

    test_data_path = config['data_config']['test_data_path']
    if 'casp' in test_data_path:
        benchmark = 'casp15'
    else:
        benchmark = 'gold'

    sample_result_path = os.path.join(sample_config['output_dir'], f'benchmark_{benchmark}_ss_{w_secondary_structure}_clash_{w_clash}.csv')
    sample_pdb_out_dir = os.path.join(sample_config['output_dir'], f'benchmark_{benchmark}_ss_{w_secondary_structure}_clash_{w_clash}_sampled_pdbs')
    if not os.path.exists(sample_pdb_out_dir):
        os.makedirs(sample_pdb_out_dir)

    test_csv = pd.read_csv(config['data_config']['test_data_path'])
    pdb2ss = dict(zip(test_csv['pdb_id'], test_csv['secondary_structure']))

    test_dataset = GraphDataset(
        data_path=data_config['test_data_path'],
        data_config=data_config
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=test_dataset.collate_fn,
    )

    # 2. Load Models
    vae = GeoVAE(config['vae_config']['model_config']).to(device)
    vae_ckpt = torch.load(config['vae_config']['ckpt_path'], map_location=device)
    print("vae model loading from: ", config['vae_config']['ckpt_path'])
    if 'model_state_dict' in vae_ckpt:
        vae.load_state_dict(vae_ckpt['model_state_dict'])
    else:
        vae.load_state_dict(vae_ckpt)

    vae_mode = config['vae_config']['model_config'].get('representation_mode', 'so2')
    supports_geometry = vae_mode != "sequence_only"

    stats_path = os.path.join(os.path.dirname(config['vae_config']['ckpt_path']), 'vae_latent_stats.pt')
    if os.path.exists(stats_path):
        print(f"Loading latent stats from {stats_path}")
        stats = torch.load(stats_path, map_location=device)
        latent_mean = stats['mean'].to(device)
        latent_std = stats['std'].to(device)
    else:
        raise FileNotFoundError(f"Latent stats not found at {stats_path}. Please run calc_latent_stats.py first.")

    safe_std = torch.where(latent_std < 0.1, torch.tensor(1.0, device=device), latent_std)

    dit_model = DiffusionDiT(config['model_config']).to(device)
    diffusion = DiffusionModel(dit_model, train_config['latent_transition_config'])
    checkpoint = torch.load(sample_config['ckpt_path'], map_location=device)
    print("diffusion model loading from: ", sample_config['ckpt_path'])
    diffusion.load_state_dict(checkpoint['model'])

    vae.eval()
    diffusion.eval()

    for p in vae.parameters():
        p.requires_grad_(False)

    for p in diffusion.parameters():
        p.requires_grad_(False)

    results = []
    num2letter = {0: 'A', 1: 'U', 2: 'G', 3: 'C'}
    valid_pairs_set = {(0, 1), (1, 0), (2, 3), (3, 2), (2, 1), (1, 2)}  # A-U, G-C, G-U

    print("w_secondary_structure: ", w_secondary_structure)
    print("w_clash: ", w_clash)

    print("Starting Guided Sampling...")
    with torch.no_grad():  # 全局还是 no_grad，引导时的 grad 在 wrapper 里局部开启
        for batch in tqdm(test_dataloader):
            batch = batch.to(device)
            pdb_id = batch.name[0]

            # --- 准备二级结构先验 ---
            ss_str = pdb2ss.get(pdb_id, "")
            ss_pairs = parse_dot_bracket(ss_str) if ss_str else []

            start_time = time.time()
            z_sampled_norm = diffusion.guide_ddim_sample(
                batch,
                num_steps=sample_config['num_steps'],
                eta=0.0,
                vae=vae,
                latent_mean=latent_mean,
                latent_std=latent_std,
                ss_pairs=ss_pairs,
                guidance_config=sample_config['guidance']
            )

            z_recon = z_sampled_norm * safe_std + latent_mean
            z_dense, mask_dense = to_dense_batch(z_recon, batch.batch)

            seq_logits, pred_angles, pred_coords = vae.decode(z_dense, mask_dense, coords_coarse_flat=(batch.coords_coarse
                                                                                                       if supports_geometry else None))

            end_time = time.time()
            print("decoding time: ", end_time - start_time)


            pred_seq = seq_logits.argmax(dim=-1)  # [L]
            gt_seq = batch.seq_idx[0] if batch.seq_idx.dim() > 1 else batch.seq_idx  # 兼容处理

            # 1. 序列 Recovery
            recovery = (pred_seq == gt_seq).float().mean().item()
            f1 = f1_score(gt_seq.cpu().numpy(), pred_seq.cpu().numpy(), labels=[0, 1, 2, 3], average='macro', zero_division=0)

            # 2. 二级结构配对合法率 (SS Validity)   NUCLEOTIDES = ['A', 'U', 'G', 'C']
            valid_pairs_count = 0
            for (i, j) in ss_pairs:
                b_i, b_j = pred_seq[i].item(), pred_seq[j].item()
                if (b_i, b_j) in valid_pairs_set:
                    valid_pairs_count += 1
            ss_validity = valid_pairs_count / max(len(ss_pairs), 1) if ss_pairs else 1.0

            # 3. 碰撞率 (Clash Score per 1000 atoms)
            if pred_coords is not None:
                mask_atoms_single = batch.mask_gt.bool()
                clash_score, clashing_pairs = evaluate_clash_score(pred_coords, mask_atoms_single)

                clashing_pairs_str = ", ".join([f"{i}-{j}" for i, j in clashing_pairs])

                pred_seq_str = "".join([num2letter.get(x.item()) for x in pred_seq])
                gt_seq_str = "".join([num2letter.get(x.item()) for x in gt_seq])

                # 提取单条序列对应的骨架坐标 [L, 3, 3]
                coords_coarse_single = batch.coords_coarse[0] if batch.coords_coarse.dim() == 4 else batch.coords_coarse
                # 命名带上两个关键指标，方便后续在文件夹里挑好看的 Case
                pdb_filename = f"{pdb_id}_clash_{clash_score:.1f}_ss_{ss_validity:.2f}.pdb"
                pdb_filepath = os.path.join(sample_pdb_out_dir, pdb_filename)

                export_full_rna_pdb(
                    coords_coarse=coords_coarse_single,
                    pred_coords=pred_coords,
                    seq_tensor=pred_seq,
                    out_filepath=pdb_filepath,
                    clashing_pairs=clashing_pairs  # <--- 就加这一个参数！
                )

                results.append({
                    "pdb_id": pdb_id,
                    "length": len(pred_seq),
                    "gt_seq": gt_seq_str,
                    "pred_seq": pred_seq_str,
                    "target_ss": ss_str,
                    "recovery": round(recovery, 4),
                    "f1": round(f1, 4),  # <--- 新增
                    "ss_validity": round(ss_validity, 4),
                    "clash_score": round(clash_score, 2),
                    "clash_pairs": clashing_pairs_str
                })
            else:
                results.append({
                    "pdb_id": pdb_id,
                    "length": len(pred_seq),
                    "gt_seq": gt_seq_str,
                    "pred_seq": pred_seq_str,
                    "target_ss": ss_str,
                    "recovery": round(recovery, 4),
                    "f1": round(f1, 4),  # <--- 新增
                    "ss_validity": round(ss_validity, 4)
                })

    df_results = pd.DataFrame(results)
    df_results.to_csv(sample_result_path, index=False)

    print("\n" + "=" * 50)
    print(" Sampling & Evaluation Complete!")
    print(f" Total Samples Processed: {len(df_results)}")
    print(f" Average Sequence Recovery: {df_results['recovery'].mean():.2%}")
    print(f" Average Sequence F1:       {df_results['f1'].mean():.2%}")
    if df_results['target_ss'].str.len().sum() > 0:
        print(f" Average SS Validity:       {df_results['ss_validity'].mean():.2%}")
    if 'clash_score' in df_results.columns():
        print(f" Average Clash Score:       {df_results['clash_score'].mean():.2f}")
    print("sampling complete ......")