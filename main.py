import argparse
from utils import load_config
from train_geo_vae import train_geo_vae
from train_latent_diffusion import train_latent_diffusion
from sample_latent_diffusion import sample_latent_diffusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/geo_vae_full_film_fusion.yaml')
    args = parser.parse_args()
    config = load_config(args.config)

    pipe = config['pipeline']

    if pipe == "train_geo_vae":
        train_geo_vae(config)
    elif pipe == "train_latent_diffusion":
        train_latent_diffusion(config)
    elif pipe == "sample_latent_diffusion":
        sample_latent_diffusion(config)
    else:
        raise ValueError(f'pipeline {pipe} not supported')