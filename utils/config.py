import yaml
import wandb

def init_config(args):
    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)
    
    args_dict = vars(args)
    configs.update(args_dict)
        
    wandb.init(
        project=configs["wandb"]["project"],
        entity=configs["wandb"]["entity"],
        config=configs
    )
    
    return configs