from omegaconf import OmegaConf

def load_config(config_path):
    return OmegaConf.load(config_path)