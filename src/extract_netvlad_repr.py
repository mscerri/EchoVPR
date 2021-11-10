import argparse
import logging
import os

import torch

from configs.utils import get_config_wandb, get_int_from_config
from echovpr.datasets.image_ds import ImageDataset
from echovpr.datasets.oxford_image_ds import OxfordImageDataset
from echovpr.datasets.utils import load_np_file, save_np_file
from echovpr.models.netvlad_encoder import NetVLADEncorder
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


def main(options):

    # Setup config
    run, config = get_config_wandb(
        options.config_file, project=options.project, entity=options.entity, logger=log, log=False)
    
    # Prepare datasets
    datasets = get_datasets_to_process(config)

    encoder = NetVLADEncorder(config).eval().to(device)

    batchsize = get_int_from_config(config, 'dataset_netvlad_processing_batchsize', None)
    dataLoader_threads = get_int_from_config(config, 'dataset_netvlad_processing_threads', None)

    for dataset_name, dataset, destination in datasets:
        hidden_repr = process_dataset(
            dataset_name, dataset, encoder, config, device, logger=log, batchsize=batchsize, dataLoader_threads=dataLoader_threads).numpy()
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
    summer_ds = load_np_file(
        config['dataset_nordland_summer_dataset_file_path'])
    winter_ds = load_np_file(
        config['dataset_nordland_winter_dataset_file_path'])
    spring_ds = load_np_file(
        config['dataset_nordland_spring_dataset_file_path'])
    fall_ds = load_np_file(config['dataset_nordland_fall_dataset_file_path'])

    return [
        ("Spring", ImageDataset(spring_ds['image_names'], config['dataset_root_dir'], config),
         config['dataset_nordland_spring_netvlad_repr_file_path']),
        ("Summer", ImageDataset(summer_ds['image_names'], config['dataset_root_dir'],
         config), config['dataset_nordland_summer_netvlad_repr_file_path']),
        ("Fall", ImageDataset(fall_ds['image_names'], config['dataset_root_dir'], config),
         config['dataset_nordland_fall_netvlad_repr_file_path']),
        ("Winter", ImageDataset(winter_ds['image_names'], config['dataset_root_dir'], config),
         config['dataset_nordland_winter_netvlad_repr_file_path'])
    ]


def get_oxford_datasets(config):
    day_ds = load_np_file(config['dataset_oxford_day_dataset_file_path'])
    night_ds = load_np_file(config['dataset_oxford_night_dataset_file_path'])

    return [
        ("Day", OxfordImageDataset(day_ds['image_names'], config['dataset_root_dir'], config),
         config['dataset_oxford_day_netvlad_repr_file_path']),
        ("Night", OxfordImageDataset(night_ds['image_names'], config['dataset_root_dir'], config),
         config['dataset_oxford_night_netvlad_repr_file_path'])
    ]


if __name__ == '__main__':
    options, unknowns = parser.parse_known_args()
    main(options)
