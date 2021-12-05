import argparse
import logging
import os
from collections import OrderedDict

import torch
from torch import nn

from configs.utils import get_config_wandb
from echovpr.datasets.utils import load_np_file, save_np_file
from echovpr.trainer.utils.simple_processor import process_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

env_torch_device = os.environ.get("TORCH_DEVICE")
if env_torch_device is not None:
    device = torch.device(env_torch_device)
    log.info(f'Setting device set by environment to {env_torch_device}')
else:
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    log.info('Setting default available device')

os.environ["WANDB_SILENT"] = "true"

parser = argparse.ArgumentParser(description='echovpr')
parser.add_argument('--config_file', type=str, required=True,
                    help='Location of config file to load defaults from')
parser.add_argument('--project', type=str, default=None, help='Wandb Project')
parser.add_argument('--entity', type=str, default=None, help='Wandb Entity')
parser.add_argument('--artifact_name', type=str, required=True,
                    help='Artifact name in WandB to export hidden representations from')
parser.add_argument('--model_name', type=str, default='model.pt',
                    help='Model name located in the artifact')


def main(options):

    # Setup config
    run, config = get_config_wandb(
        options.config_file, project=options.project, entity=options.entity, logger=log, log=False)

    model_tensor = load_model(run, options.artifact_name, options.model_name)

    # Setup ESN
    in_features = int(config['model_in_features'])
    hidden_features = int(config['model_hidden_features'])
    out_features = int(config['model_out_features'])

    layers = []

    if hidden_features > 0:
        layers.append(('hl', nn.Linear(in_features=in_features,
                      out_features=hidden_features, bias=True)))
        out_layer_in_features = hidden_features
    else:
        out_layer_in_features = in_features

    layers.append(('out', nn.Linear(
        in_features=out_layer_in_features, out_features=out_features, bias=True)))

    model = nn.Sequential(OrderedDict(layers))

    model.load_state_dict(model_tensor)

    # Move to device
    model.eval().to(device)

    # Prepare datasets
    datasets = get_datasets_to_process(config)

    encoder = model.get_submodule('hl')

    for dataset_name, dataset, destination in datasets:
        hidden_repr = process_dataset(dataset_name, dataset, encoder, config, device, logger=log).numpy()
        save_np_file(hidden_repr, destination)

    run.finish()


def load_model(run, artifact_name: str, model_name: str) -> str:
    model_artifact = run.use_artifact(artifact_name, type='model')
    model_dir = model_artifact.download()
    return torch.load(os.path.join(model_dir, model_name))


def get_datasets_to_process(config):
    if config['dataset'] == 'nordland':
        return get_nordland_datasets(config)
    elif config['dataset'] == 'oxford':
        return get_oxford_datasets(config)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")


def get_nordland_datasets(config):
    spring_netvlad_repr = torch.from_numpy(load_np_file(
        config['dataset_nordland_spring_netvlad_repr_file_path']))
    summer_test_netvald_repr = torch.from_numpy(load_np_file(
        config['dataset_nordland_summer_netvlad_repr_file_path']))
    fall_test_netvald_repr = torch.from_numpy(load_np_file(
        config['dataset_nordland_fall_netvlad_repr_file_path']))
    winter_test_netvald_repr = torch.from_numpy(load_np_file(
        config['dataset_nordland_winter_netvlad_repr_file_path']))
    
    return [
        ("Spring", spring_netvlad_repr,
         config['dataset_nordland_spring_hidden_repr_file_path']),
        ("Summer", summer_test_netvald_repr,
         config['dataset_nordland_summer_hidden_repr_file_path']),
        ("Fall", fall_test_netvald_repr,
         config['dataset_nordland_fall_hidden_repr_file_path']),
        ("Winter", winter_test_netvald_repr,
         config['dataset_nordland_summer_hidden_repr_file_path'])
    ]


def get_oxford_datasets(config):
    day_netvlad_repr = torch.from_numpy(load_np_file(
        config['dataset_oxford_day_netvlad_repr_file_path']))
    night_netvald_repr = torch.from_numpy(load_np_file(
        config['dataset_oxford_night_netvlad_repr_file_path']))

    return [
        ("Day", day_netvlad_repr,
         config['dataset_oxford_day_hidden_repr_file_path']),
        ("Night", night_netvald_repr,
         config['dataset_oxford_night_hidden_repr_file_path'])
    ]


if __name__ == '__main__':
    options, unknowns = parser.parse_known_args()
    main(options)
