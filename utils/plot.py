import os
import wandb

import numpy as np
import mdtraj as md 

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

def plot(configs, system, samples, target_energy, idx="0"):
    
    if configs["plot"]["distribution"]:
        fig_distribution, ax = plt.subplots(figsize=(3,3))
        
        if configs["dataset"]["molecule"] == "Alanine Dipeptide":
            plot_alanine_phi_psi(configs, ax, samples, system)
        else:
            raise ValueError(f"Distribution plot not implemented for {configs['dataset']['molecule']}")
        
        image_path = f'{configs["path"]}/{configs["date"]}'
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        image_name = f'{image_path}/{configs["dataset"]["name"]}_{idx}.png'
        fig_distribution.savefig(image_name)
        print(f"Saved distribution to {image_name}")
        
        if "wandb" in configs:
            wandb.log({"Generator samples": wandb.Image(fig_distribution)})
        
        plt.close()
    
    if configs["plot"]["energy"]:
        fig_energies, ax = plt.subplots(figsize=(3,3))
        
        if configs["dataset"]["molecule"] == "Alanine Dipeptide":
            sample_energies = plot_energies(configs, ax, samples, target_energy)
        else:
            raise ValueError(f"Distribution plot not implemented for {configs['dataset']['molecule']}")
        
        image_path = f'{configs["path"]}/{configs["date"]}'
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        image_name = f'{image_path}/{configs["dataset"]["name"]}_energies_{idx}.png'
        fig_energies.savefig(image_name)
        np.save(f'{configs["path"]}/{configs["date"]}/{configs["dataset"]["name"]}_energies_{idx}.npy', sample_energies)
        print(f"Saved energy plot to {image_name}")
        
        if "wandb" in configs:
            wandb.log({"Generator energies": wandb.Image(fig_energies)})
        
        plt.close()

def plot_energies(configs, ax, samples, target_energy):
    sample_energies = target_energy.energy(samples).cpu().detach().numpy()    
    cut = max(np.percentile(sample_energies, 80), 20)
    
    ax.set_xlabel("Energy   [$k_B T$]")
    ax2 = plt.twinx(ax)
    ax.get_yaxis().set_visible(False)
    
    ax2.hist(sample_energies, range=(-50, cut), bins=configs["plot"]["hist"], density=False, label="BG")
    ax2.set_ylabel(f"Count   [#Samples / {len(samples)}]")
    ax2.legend()
    
    return sample_energies


# Molecule specific plots

def plot_alanine_phi_psi(configs, ax, trajectory, system):
    if not isinstance(trajectory, md.Trajectory):
        trajectory = md.Trajectory(
            xyz=trajectory.cpu().detach().numpy().reshape(-1, 22, 3), 
            topology=system.mdtraj_topology
        )
    phi, psi = system.compute_phi_psi(trajectory)
    
    ax.hist2d(phi, psi, configs["plot"]["hist2d"], norm=LogNorm())
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("$\phi$")
    _ = ax.set_ylabel("$\psi$")
    
    return trajectory

