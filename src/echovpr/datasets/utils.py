import os
import pickle
from os.path import isfile, join

import numpy as np
import torch
import torchvision.transforms as transforms
from configs import ROOT_DIR
from torch.utils.data.dataset import Subset, TensorDataset


def input_transform(resize=(480, 640)):
    if resize[0] > 0 and resize[1] > 0:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

def parse_text_file(textfile):
    print('Parsing dataset...')

    with open(textfile, 'r') as f:
        image_list = f.read().splitlines()

    if 'robotcar' in image_list[0].lower():
        image_list = [os.path.splitext('/'.join(q_im.split('/')[-3:]))[0] for q_im in image_list]

    num_images = len(image_list)

    print('Done! Found %d images' % num_images)

    return image_list, num_images

def get_dataset(dataset_file_path, limit_indices_file_path = None):
    dataset_file_path = join(ROOT_DIR, dataset_file_path)
    assert isfile(dataset_file_path)

    (x, y_target) = torch.load(dataset_file_path)
    main_dataset = TensorDataset(x, y_target)

    return get_subset_dataset(main_dataset, limit_indices_file_path)

def get_subset_dataset(main_dataset, limit_indices_file_path = None):
    if limit_indices_file_path == None:
        return main_dataset

    limit_indices_file_path = join(ROOT_DIR, limit_indices_file_path)
    assert isfile(limit_indices_file_path)
    
    with open(limit_indices_file_path,"rb") as file_handle:
        allowed_indices = pickle.load(file_handle)
        return Subset(main_dataset, allowed_indices)

def save_tensor(tensor, file_path):
    file_path = join(ROOT_DIR, file_path)
    print('Saving file to %s' % file_path)
    torch.save(tensor, file_path)

def save_np_file(array, file_path):
    file_path = join(ROOT_DIR, file_path)
    print('Saving file to %s' % file_path)
    np.save(file_path, array)

def load_np_file(file_path):
    file_path = os.path.join(ROOT_DIR, file_path)
    assert os.path.isfile(file_path)
    return np.load(file_path, allow_pickle=True)

def get_1_hot_encode(a, num_classes):
    b = np.zeros((a.size, num_classes))
    b[np.arange(a.size), a] = 1
    return b
