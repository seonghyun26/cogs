import os
import yaml
import wandb
import torch
import argparse
import numpy as np
import bgflow as bg
import mdtraj as md 

from tqdm.auto import tqdm, trange
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from model.nvp import RealNVP

from data.data_loader import *

from utils.config import *
from utils.plot import *


# Create argument parser
parser = argparse.ArgumentParser(description="Train script")
parser.add_argument("--config", type=str, default="config/basic.yaml", help="Path to config file")
parser.add_argument("--date", type=str, default="testdate", help="Date for program run")
parser.add_argument("--device", type=str, default="1", help="GPU to use")
args = parser.parse_args()

# Initialize device
device = args.device if torch.cuda.is_available() else "cpu"
ctx = torch.zeros([], device=device, dtype=torch.float32)

if __name__ == "__main__":
    # Load configs from yaml file
    configs = init_config(args)
    system, temperature, all_data, training_data, test_data, target_energy = init_dataset(configs, ctx, save_image=False)
    
    # Preprocess
    dim_cartesian, dim_bonds, dim_angles, dim_torsions = len(system.rigid_block) * 3 - 6, len(system.z_matrix), dim_bonds, dim_bonds
    coordinate_transform = bg.MixedCoordinateTransformation(
        data=training_data, 
        z_matrix=system.z_matrix,
        fixed_atoms=system.rigid_block,
        keepdims=dim_cartesian, 
        normalize_angles=True,
    ).to(ctx)
    dim_ics = dim_cartesian + dim_bonds + dim_angles + dim_torsions
    
    # Set prior distribution
    prior = set_prior(configs, dim_ics, ctx)
    # mean = torch.zeros(dim_ics).to(ctx) 
    # prior = bg.NormalDistribution(dim_ics, mean=mean)
    
    # model
    split_into_ics_flow = bg.SplitFlow(dim_bonds, dim_angles, dim_torsions, dim_cartesian)
    RealNVP(dim_ics, hidden=[128]).to(ctx).forward(prior.sample(3))[0].shape
    n_realnvp_blocks = configs["model"]["layers"]
    layers = []
    for i in range(n_realnvp_blocks):
        layers.append(RealNVP(dim_ics, hidden=[128, 128, 128]))
    layers.append(split_into_ics_flow)
    layers.append(bg.InverseFlow(coordinate_transform))
    flow = bg.SequentialFlow(layers).to(ctx)
    
    print("# Parameters:", np.sum([np.prod(p.size()) for p in flow.parameters()]))
    generator = bg.BoltzmannGenerator(
        flow=flow,
        prior=prior,
        target=target_energy
    )
    
    # train
    loss_type = configs["train"]["loss"]
    if loss_type == "nll":
        optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=1e-3
        )
        trainer = bg.KLTrainer(
            generator, 
            optim=optimizer,
            train_energy=False
        )
        trainer.train(
            n_iter=configs["train"]["iter"],
            data=training_data,
            batchsize=configs["train"]["batchsize"],
            n_print=configs["train"]["n_print"], 
            w_energy=configs["train"]["w_energy"],
            progress_bar=tqdm
        )
    elif loss_type == "mixed":
        mixed_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=1e-4
        )
        mixed_trainer = bg.KLTrainer(
            generator, 
            optim=mixed_optimizer,
            train_energy=True
        )
        mixed_trainer.train(
            n_iter=configs["train"]["iter"],
            data=training_data,
            batchsize=configs["train"]["batchsize"],
            n_print=configs["train"]["n_print"], 
            w_energy=configs["train"]["w_energy"],
            w_likelihood=configs["train"]["w_likelihood"],
            clip_forces=configs["train"]["clip_forces"],
            progress_bar=tqdm
        )
    else:
        raise ValueError(f"Invalid loss function {loss_type}")
        
    
    # Plot and save image
    samples = generator.sample(configs["sample"]["n_samples"])
    plot_distribution(configs, system, samples)

    
    wandb.finish()