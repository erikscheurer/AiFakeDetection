from omegaconf import OmegaConf
import argparse

def load_config(config_path):
    """
    Load configuration file and merge with extra options from --extra_opts
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--extra_opts', type=str, default='', help='Extra options to merge with config, e.g. "train.batch_size=32,train.niter=100" overwrites the config file')
    args = parser.parse_args()

    config = OmegaConf.load(config_path)
    if args.extra_opts:
        extra_opts = OmegaConf.from_dotlist(args.extra_opts.split(';'))
        config = OmegaConf.merge(config, extra_opts)

    return config

if __name__ == '__main__':
    config = load_config('models/CNNDetection/train.yaml')
    print(OmegaConf.to_yaml(config))
