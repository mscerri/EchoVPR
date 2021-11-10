from typing import Dict

import numpy as np
import torch
from echovpr.datasets.utils import get_1_hot_encode, load_np_file
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset, Subset, TensorDataset

def prepare_final_datasets(esn_descriptors: Dict[str, torch.Tensor], config, eval_only = False):
    train_dataset, val_dataset, test_dataset, train_gt, eval_gt = get_datasets(esn_descriptors, config, eval_only)

    train_dataLoader = None
    
    if not eval_only:
        train_dataLoader = DataLoader(train_dataset, num_workers=int(config['dataloader_threads']), batch_size=int(config['train_batchsize']), shuffle=True)

    val_dataLoader = DataLoader(val_dataset, num_workers=int(config['dataloader_threads']), batch_size=int(config['train_batchsize']), shuffle=False)

    test_dataLoader = DataLoader(test_dataset, num_workers=int(config['dataloader_threads']), batch_size=int(config['train_batchsize']), shuffle=False)

    return train_dataset, train_dataLoader, val_dataset, val_dataLoader, test_dataLoader, train_gt, eval_gt

def get_datasets(esn_descriptors: Dict[str, torch.Tensor], config, eval_only: bool):
    if config['dataset'] == 'nordland':
        return get_datasets_for_nordland(esn_descriptors, config, eval_only)
    if config['dataset'] == 'nordland_spr_fall':
        return get_datasets_for_nordland_spring_vs_fall(esn_descriptors, config, eval_only)
    elif config['dataset'] == 'oxford':
        return get_datasets_for_oxford(esn_descriptors, config, eval_only)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")

def get_dataset_infos(config):
    if config['dataset'] == 'nordland':
        summer_dataset_info = load_np_file(config['dataset_nordland_summer_dataset_file_path'])
        winter_dataset_info = load_np_file(config['dataset_nordland_winter_dataset_file_path'])
        return summer_dataset_info, winter_dataset_info
    # if config['dataset'] == 'nordland_spr_fall':
    #     return get_datasets_for_nordland_spring_vs_fall(esn_descriptors, config, eval_only)
    elif config['dataset'] == 'oxford':
        day_dataset_info = load_np_file(config['dataset_oxford_day_dataset_file_path'])
        night_dataset_info = load_np_file(config['dataset_oxford_night_dataset_file_path'])
    
        return day_dataset_info, night_dataset_info
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")

def get_datasets_for_nordland(esn_descriptors: Dict[str, torch.Tensor], config, eval_only = False):
    summer_dataset_info = load_np_file(config['dataset_nordland_summer_dataset_file_path'])
    winter_dataset_info = load_np_file(config['dataset_nordland_winter_dataset_file_path'])
    val_test_splits = load_np_file(config['dataset_nordland_winter_val_test_splits_indices_file_path'])

    train_dataset = None
    train_gt = None

    if not eval_only:
        train_gt = summer_dataset_info['ground_truth_indices']

        esn_descriptor_summer = esn_descriptors['summer']
        summer_image_idx = torch.from_numpy(summer_dataset_info['image_indices'])
        summer_image_1_hot = torch.from_numpy(get_1_hot_encode(summer_dataset_info['image_indices'], len(summer_dataset_info['image_indices']))).type(torch.float)
        
        train_dataset = TensorDataset(esn_descriptor_summer, summer_image_1_hot, summer_image_idx)
        print(f"Train dataset size: {len(train_dataset)}")

    eval_gt = winter_dataset_info['ground_truth_indices']

    esn_descriptor_winter = esn_descriptors['winter']
    winter_image_idx = torch.from_numpy(winter_dataset_info['image_indices'])
    winter_dataset = TensorDataset(esn_descriptor_winter, winter_image_idx)

    val_dataset = Subset(winter_dataset, val_test_splits['val_indices'])
    print(f"Val dataset size: {len(val_dataset)}")

    test_dataset = Subset(winter_dataset, val_test_splits['test_indices'])
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, train_gt, eval_gt

def get_datasets_for_nordland_spring_vs_fall(esn_descriptors: Dict[str, torch.Tensor], config, eval_only = False):
    desired_train_size = int(config['desired_train_size'])
    desired_val_size = int(config['desired_val_size'])
    desired_test_size = int(config['desired_test_size'])
    continuity = int(config['continuity'])

    train_indices, val_indices, test_indices = generate_indices_splits(continuity, desired_train_size, desired_val_size, desired_test_size)

    summer_dataset_info = load_np_file(config['dataset_nordland_summer_dataset_file_path'])
    esn_descriptor_summer = esn_descriptors['summer']
    summer_image_idx = torch.from_numpy(summer_dataset_info['image_indices'])

    winter_dataset_info = load_np_file(config['dataset_nordland_winter_dataset_file_path'])
    esn_descriptor_winter = esn_descriptors['winter']
    winter_image_idx = torch.from_numpy(winter_dataset_info['image_indices'])

    train_gt = None
    train_dataset = None
    
    if not eval_only:
        train_gt = summer_dataset_info['ground_truth_indices']

        summer_image_1_hot = torch.from_numpy(get_1_hot_encode(summer_dataset_info['image_indices'], len(summer_dataset_info['image_indices']))).type(torch.float)
        summer_train_dataset = TensorDataset(esn_descriptor_summer, summer_image_1_hot, summer_image_idx)

        winter_image_1_hot = torch.from_numpy(get_1_hot_encode(winter_dataset_info['image_indices'], len(winter_dataset_info['image_indices']))).type(torch.float)
        winter_train_dataset = TensorDataset(esn_descriptor_winter, winter_image_1_hot, winter_image_idx)

        train_dataset = ConcatDataset([Subset(summer_train_dataset, train_indices), Subset(winter_train_dataset, train_indices)])
        print(f"Train dataset size: {len(train_dataset)}")
    
    summer_val_dataset = TensorDataset(esn_descriptor_summer, summer_image_idx)
    winter_val_dataset = TensorDataset(esn_descriptor_winter, winter_image_idx)
    val_dataset = ConcatDataset([Subset(summer_val_dataset, val_indices), Subset(winter_val_dataset, val_indices)])
    print(f"Val dataset size: {len(val_dataset)}")

    spring_dataset_info = load_np_file(config['dataset_nordland_spring_dataset_file_path'])
    
    esn_descriptor_spring = esn_descriptors['spring']
    spring_image_idx = torch.from_numpy(spring_dataset_info['image_indices'])
    spring_dataset = TensorDataset(esn_descriptor_spring, spring_image_idx)

    fall_dataset_info = load_np_file(config['dataset_nordland_fall_dataset_file_path'])

    esn_descriptor_fall = esn_descriptors['fall']
    fall_image_idx = torch.from_numpy(fall_dataset_info['image_indices'])
    fall_dataset = TensorDataset(esn_descriptor_fall, fall_image_idx)

    test_dataset = ConcatDataset([Subset(spring_dataset, test_indices), Subset(fall_dataset, test_indices)])
    print(f"Test dataset size: {len(test_dataset)}")

    eval_gt = winter_dataset_info['ground_truth_indices']

    return train_dataset, val_dataset, test_dataset, train_gt, eval_gt

def get_datasets_for_oxford(esn_descriptors: Dict[str, torch.Tensor], config, eval_only = False):
    day_dataset_info = load_np_file(config['dataset_oxford_day_dataset_file_path'])
    night_dataset_info = load_np_file(config['dataset_oxford_night_dataset_file_path'])
    val_test_splits = load_np_file(config['dataset_oxford_night_val_test_splits_indices_file_path'])

    train_dataset = None
    train_gt = None

    if not eval_only:
        train_gt = day_dataset_info['ground_truth_indices']

        esn_descriptor_day = esn_descriptors['day']
        day_image_idx = torch.from_numpy(day_dataset_info['image_indices'])
        day_image_1_hot = torch.from_numpy(get_1_hot_encode(day_dataset_info['image_indices'], len(day_dataset_info['image_indices']))).type(torch.float)
        
        train_dataset = TensorDataset(esn_descriptor_day, day_image_1_hot, day_image_idx)
        print(f"Train dataset size: {len(train_dataset)}")

    eval_gt = night_dataset_info['ground_truth_indices']

    esn_descriptor_night = esn_descriptors['night']
    night_image_idx = torch.from_numpy(night_dataset_info['image_indices'])
    night_dataset = TensorDataset(esn_descriptor_night, night_image_idx)

    val_dataset = Subset(night_dataset, val_test_splits['val_indices'])
    print(f"Val dataset size: {len(val_dataset)}")

    test_dataset = Subset(night_dataset, val_test_splits['test_indices'])
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, train_gt, eval_gt 

def generate_indices_splits(continuity, desired_train_size, desired_val_size, desired_test_size):
    n_train_splits = int(np.round(desired_train_size/continuity))
    n_val_splits = int(np.round(desired_val_size/continuity))
    n_test_splits = int(np.round(desired_test_size/continuity))
    
    train_size = int(n_train_splits * continuity)
    val_size = int(n_val_splits * continuity)
    test_size = int(n_test_splits * continuity)

    total_size = train_size + val_size + test_size
    indices = (np.arange(0,total_size)).astype(int)
    
    split_buckets = np.reshape(indices, [int(total_size/continuity), continuity])
    
    split_bucket_indices = np.shape(split_buckets)[0]
    ind_perm = np.random.permutation(split_bucket_indices)
    
    val_indices = split_buckets[ind_perm[0:n_val_splits], :]
    test_indices = split_buckets[ind_perm[n_val_splits:n_val_splits+n_test_splits], :]
    train_indices = split_buckets[ind_perm[n_val_splits+n_test_splits:], :]
    
    val_indices = np.sort(np.reshape(val_indices, [-1]))
    test_indices = np.sort(np.reshape(test_indices, [-1]))
    train_indices = np.sort(np.reshape(train_indices, [-1]))

    return train_indices, val_indices, test_indices
