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

from model import *

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
    wandb_use = True if "wandb" in configs else False
    
    # Preprocess
    dim_cartesian, dim_bonds = len(system.rigid_block) * 3 - 6, len(system.z_matrix)
    dim_angles, dim_torsions = dim_bonds, dim_bonds
    dim_ics = dim_cartesian + dim_bonds + dim_angles + dim_torsions
    dim_dict = {
        "dim_cartesian": dim_cartesian,
        "dim_bonds": dim_bonds,
        "dim_angles": dim_angles,
        "dim_torsions": dim_torsions,
        "dim_ics": dim_ics
    }
    coordinate_transform = bg.MixedCoordinateTransformation(
        data=training_data, 
        z_matrix=system.z_matrix,
        fixed_atoms=system.rigid_block,
        keepdims=dim_cartesian, 
        normalize_angles=True,
    ).to(ctx)
        
    # Set model
    prior = set_prior(configs, dim_ics, ctx)
    generator, parameters = set_model(configs, dim_dict, prior, target_energy, coordinate_transform, ctx)
    if wandb_use:
        wandb.log({"Number of parameters": parameters})
    
    # Train generator
    loss_type = configs["train"]["loss"]
    if loss_type == "nll":
        optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=1e-3
        )
        trainer = bg.KLTrainer(
            generator, 
            optim=optimizer,
            train_energy=False,
            configs=configs,
            system=system
        )
        trainer.train(
            n_iter=configs["train"]["iter"],
            data=training_data,
            batchsize=configs["train"]["batchsize"],
            n_print=configs["train"]["n_print"], 
            w_energy=configs["train"]["w_energy"],
            progress_bar=tqdm,
            wandb_use=wandb_use
        )
    elif loss_type == "kll":
        optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=1e-3
        )
        trainer = bg.KLTrainer(
            generator, 
            optim=optimizer,
            train_likelihood=False,
            train_energy=True,
            configs=configs,
            system=system
        )
        trainer.train(
            n_iter=configs["train"]["iter"],
            data=training_data,
            batchsize=configs["train"]["batchsize"],
            n_print=configs["train"]["n_print"], 
            w_energy=configs["train"]["w_energy"],
            progress_bar=tqdm,
            wandb_use=wandb_use
        )
    elif loss_type == "mixed":
        mixed_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=1e-4
        )
        mixed_trainer = bg.KLTrainer(
            generator, 
            optim=mixed_optimizer,
            train_energy=True,
            configs=configs,
            system=system
        )
        mixed_trainer.train(
            n_iter=configs["train"]["iter"],
            data=training_data,
            batchsize=configs["train"]["batchsize"],
            n_print=configs["train"]["n_print"], 
            w_energy=configs["train"]["w_energy"],
            w_likelihood=configs["train"]["w_likelihood"],
            clip_forces=configs["train"]["clip_forces"],
            progress_bar=tqdm,
            wandb_use=wandb_use
        )
    else:
        raise ValueError(f"Invalid loss function {loss_type}")
        
    
    # Plot and save image
    samples = generator.sample(configs["sample"]["n_samples"])
    plot(configs, system, samples, target_energy, idx="final")

    # Finish
    wandb.finish()