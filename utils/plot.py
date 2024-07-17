import numpy as np
import mdtraj as md 

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

def plot_distribution(configs, system, samples):
    fig_distribution, ax = plt.subplots(figsize=(3,3))
    
    if configs["dataset"]["molecule"] == "Alanine Dipeptide":
        plot_alanine_phi_psi(ax, samples, system)
    else:
        raise ValueError(f"Distribution plot not implemented for {configs['dataset']['molecule']}")
    
    fig_distribution.savefig(f'{configs["path"]}/{args.date}_{configs["dataset"]}.png')
    
    if "wandb" in configs:
        wandb.log({"Generator samples": wandb.Image(fig_gen)})

def plot_energies(ax, samples, target_energy, test_data):
    sample_energies = target_energy.energy(samples).cpu().detach().numpy()
    md_energies = target_energy.energy(test_data[:len(samples)]).cpu().detach().numpy()
    cut = max(np.percentile(sample_energies, 80), 20)
    
    ax.set_xlabel("Energy   [$k_B T$]")
    # y-axis on the right
    ax2 = plt.twinx(ax)
    ax.get_yaxis().set_visible(False)
    
    ax2.hist(sample_energies, range=(-50, cut), bins=40, density=False, label="BG")
    ax2.hist(md_energies, range=(-50, cut), bins=40, density=False, label="MD")
    ax2.set_ylabel(f"Count   [#Samples / {len(samples)}]")
    ax2.legend()


# Molecule specific plots

def plot_alanine_phi_psi(ax, trajectory, system):
    if not isinstance(trajectory, md.Trajectory):
        trajectory = md.Trajectory(
            xyz=trajectory.cpu().detach().numpy().reshape(-1, 22, 3), 
            topology=system.mdtraj_topology
        )
    phi, psi = system.compute_phi_psi(trajectory)
    
    ax.hist2d(phi, psi, 50, norm=LogNorm())
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("$\phi$")
    _ = ax.set_ylabel("$\psi$")
    
    return trajectory

