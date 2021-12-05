from typing import Dict
import torch

from echovpr.datasets.utils import load_np_file
from echovpr.trainer.utils.simple_processor import process_dataset

def prepare_esn_datasets(model_esn, config, device, log, eval_only = False) -> Dict[str, torch.Tensor]:
    if config['dataset'] == 'nordland':
        return get_esn_repr_for_nordland(model_esn, config, device, log, eval_only)
    if config['dataset'] == 'nordland_spr_fall':
        return get_esn_repr_for_nordland(model_esn, config, device, log, eval_only, with_spr_fall=True)
    elif config['dataset'] == 'oxford':
        return get_esn_repr_for_oxford(model_esn, config, device, log, eval_only)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")

def get_esn_repr_for_nordland(model_esn, config, device, log, eval_only, with_spr_fall = False) -> Dict[str, torch.Tensor]:
    summer_hidden_repr = torch.from_numpy(load_np_file(config['dataset_nordland_summer_hidden_repr_file_path']))
    winter_hidden_repr = torch.from_numpy(load_np_file(config['dataset_nordland_winter_hidden_repr_file_path']))

    # Normalize dataset
    max_n = summer_hidden_repr.max()
    
    datasets = {}

    if not eval_only:
        _ = summer_hidden_repr.divide_(max_n)
        esn_summer_repr  = process_dataset("Summer", summer_hidden_repr, model_esn, config, device, log)
        
        datasets['summer'] = esn_summer_repr

    del summer_hidden_repr

    _ = winter_hidden_repr.divide_(max_n)
    esn_winter_repr  = process_dataset("Winter", winter_hidden_repr, model_esn, config, device, log)

    del winter_hidden_repr
    
    datasets['winter'] = esn_winter_repr

    if with_spr_fall:
        spring_hidden_repr = torch.from_numpy(load_np_file(config['dataset_nordland_spring_hidden_repr_file_path']))
        _ = spring_hidden_repr.divide_(max_n)
        esn_spring_repr  = process_dataset("Spring", spring_hidden_repr, model_esn, config, device, log)
        
        del spring_hidden_repr

        datasets['spring'] = esn_spring_repr

        fall_hidden_repr = torch.from_numpy(load_np_file(config['dataset_nordland_fall_hidden_repr_file_path']))
        _ = fall_hidden_repr.divide_(max_n)
        esn_fall_repr  = process_dataset("Fall", fall_hidden_repr, model_esn, config, device, log)

        del fall_hidden_repr

        datasets['fall'] = esn_fall_repr

    return datasets

def get_esn_repr_for_oxford(model_esn, config, device, log, eval_only) -> Dict[str, torch.Tensor]:
    day_hidden_repr = torch.from_numpy(load_np_file(config['dataset_oxford_day_hidden_repr_file_path']))
    night_hidden_repr = torch.from_numpy(load_np_file(config['dataset_oxford_night_hidden_repr_file_path']))

    # Normalize dataset
    max_n = day_hidden_repr.max()
    
    datasets = {}

    if not eval_only:
        _ = day_hidden_repr.divide_(max_n)
        esn_day_repr  = process_dataset("Day", day_hidden_repr, model_esn, config, device, log)

        datasets['day'] = esn_day_repr
    
    del day_hidden_repr

    _ = night_hidden_repr.divide_(max_n)
    esn_night_repr  = process_dataset("Night", night_hidden_repr, model_esn, config, device, log)

    del night_hidden_repr
    
    datasets['night'] = esn_night_repr

    return datasets
