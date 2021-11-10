import os

import numpy as np
import torch
from patchnetvlad.tools.local_matcher import (
    calc_keypoint_centers_from_patches, normalise_func)
from patchnetvlad.tools.patch_matcher import PatchMatcher
from tqdm.auto import tqdm


def local_matcher(predictions, config, input_query_local_features_prefix, input_index_local_features_prefix, query_dataset, index_dataset, device):
    patch_sizes = [int(s) for s in config['main']['model_patch_sizes'].split(",")]
    strides = [int(s) for s in config['main']['model_strides'].split(",")]
    patch_weights = np.array(config['feature_match']['patchWeights2Use'].split(",")).astype(float)

    all_keypoints = []
    all_indices = []

    for patch_size, stride in zip(patch_sizes, strides):
        # we currently only provide support for square patches, but this can be easily modified for future works
        keypoints, indices = calc_keypoint_centers_from_patches(config['feature_match'], patch_size, patch_size, stride, stride)
        all_keypoints.append(keypoints)
        all_indices.append(indices)

    reordered_preds = []

    matcher = PatchMatcher(config['feature_match']['matcher'], patch_sizes, strides, all_keypoints, all_indices)

    for i, item in enumerate(tqdm(predictions, leave=False, desc='Patch compare pred')):
        q_idx, pred = item
        reordered_preds.append(local_matcher_loop(q_idx, pred, patch_sizes, matcher, patch_weights, input_query_local_features_prefix, input_index_local_features_prefix, query_dataset, index_dataset, device))
  
    return reordered_preds

def local_matcher_loop(q_idx, pred, patch_sizes, matcher, patch_weights, input_query_local_features_prefix, input_index_local_features_prefix, query_dataset, index_dataset, device):
    
    diffs = np.zeros((pred.shape[0], len(patch_sizes)))
        
    image_name_query = os.path.splitext(os.path.basename(query_dataset['image_names'][q_idx]))[0]
    qfeat = []
    for patch_size in patch_sizes:
        qfilename = input_query_local_features_prefix + '_' + 'psize{}_'.format(patch_size) + image_name_query + '.npy'
        qfeat.append(torch.transpose(torch.tensor(np.load(qfilename), device=device), 0, 1))
        # we pre-transpose here to save compute speed
        
    for k, candidate in enumerate(pred):
        image_name_index = os.path.splitext(os.path.basename(index_dataset['image_names'][candidate]))[0]
        dbfeat = []
        for patch_size in patch_sizes:
            dbfilename = input_index_local_features_prefix + '_' + 'psize{}_'.format(patch_size) + image_name_index + '.npy'
            dbfeat.append(torch.tensor(np.load(dbfilename), device=device))

        diffs[k, :], _, _ = matcher.match(qfeat, dbfeat)

    diffs = normalise_func(diffs, len(patch_sizes), patch_weights)
    cand_sorted = np.argsort(diffs)

    return (q_idx, pred[cand_sorted])
