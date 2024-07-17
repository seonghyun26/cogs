import torch

import bgflow as bg

def set_prior(configs, dim, ctx):
    if configs["model"]["prior"] == "normal":
        mean = torch.zeros(dim).to(ctx)
        prior = bg.NormalDistribution(dim, mean=mean)
        
    return prior 
