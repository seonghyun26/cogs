import torch

import numpy as np
import bgflow as bg

from model.nvp import RealNVP

def set_model(configs, dim_dict, prior, target_energy, coordinate_transform, ctx):
    flow = set_flow(configs, dim_dict, prior, coordinate_transform, ctx)
    print("# Parameters in flow:", np.sum([np.prod(p.size()) for p in flow.parameters()]))
    
    generator = bg.BoltzmannGenerator(
        flow=flow,
        prior=prior,
        target=target_energy
    )
    
    return generator
    
def set_flow(configs, dim_dict, prior, coordinate_transform, ctx):
    hdim = configs["model"]["hidden_dim"]
    n_layers = configs["model"]["layers"]
    
    split_into_ics_flow = bg.SplitFlow(
        dim_dict["dim_bonds"],
        dim_dict["dim_angles"],
        dim_dict["dim_torsions"],
        dim_dict["dim_cartesian"]
    )
    
    layers = []
    for i in range(n_layers):
        layers.append(RealNVP(dim_dict["dim_ics"], hidden=hdim))
    layers.append(split_into_ics_flow)
    layers.append(bg.InverseFlow(coordinate_transform))
    
    flow = bg.SequentialFlow(layers).to(ctx)
    
    return flow