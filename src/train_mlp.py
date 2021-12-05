import argparse
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, TensorDataset

import wandb
from configs.utils import (get_bool_from_config, get_config_wandb,
                           get_float_from_config, get_int_from_config,
                           get_value_from_namespace, update_config_wandb)
from echovpr.datasets.utils import get_1_hot_encode, load_np_file
from echovpr.trainer.eval import run_eval
from echovpr.trainer.metrics.recall import compute_recall

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

env_torch_device = os.environ.get("TORCH_DEVICE")
if env_torch_device is not None:
    device = torch.device(env_torch_device)
    log.info(f'Setting device set by environment to {env_torch_device}')
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   
    log.info('Setting default available device')

os.environ["WANDB_SILENT"] = "true"

parser = argparse.ArgumentParser(description='echovpr', argument_default=argparse.SUPPRESS)
parser.add_argument('--config_file', type=str, required=True, help='Location of config file to load defaults from')

parser.add_argument('--project', type=str, help='Wandb Project')
parser.add_argument('--entity', type=str, help='Wandb Entity')

parser.add_argument('--dataset', type=str, help='Dataset')

parser.add_argument('--train_batchsize', type=int, help='Batch size')
parser.add_argument('--train_max_epochs', type=int, help='Maximum training epochs')
parser.add_argument('--train_lr', type=float, help='Learning Rate')

def main(options: argparse.Namespace):

    # Setup config
    run, config = get_config_wandb(options.config_file, project=get_value_from_namespace(options, 'project', None), entity=get_value_from_namespace(options, 'entity', None), logger=log, log=False)
    
    config = update_config_wandb(run, options, logger=log, log=True)

    random_seed = get_float_from_config(config, 'model_random_seed', None)

    if (random_seed is not None):
        log.info(f'Setting random seed to {random_seed}')
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # Setup ESN
    in_features=int(config['model_in_features'])
    hidden_features=int(config['model_hidden_features'])
    out_features=int(config['model_out_features'])

    layers = []

    if hidden_features > 0:
        layers.append(('hl', nn.Linear(in_features=in_features, out_features=hidden_features, bias=True)))
        out_layer_in_features = hidden_features
    else:
        out_layer_in_features = in_features

    layers.append(('out', nn.Linear(in_features=out_layer_in_features, out_features=out_features, bias=True)))

    model = nn.Sequential(OrderedDict(layers))

    # Move to device

    model.to(device)
    
    # Prepare datasets
    train_dataLoader, val_dataLoader, test_dataLoader, train_gt, eval_gt = prepare_datasets(config)

    lr = float(config['train_lr'])    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(reduction='mean').to(device)

    # Watch Model
    wandb.watch(model, criterion=criterion, log="all", idx=1, log_graph=True)

    # Setup metrics
    n_values = [1, 5, 10, 20, 50, 100]
    top_k = max(n_values)

    # Training loop
    early_stopping_enabled = get_bool_from_config(config, 'early_stopping_enabled', False)
    early_stopping_patience = get_int_from_config(config, 'early_stopping_patience', 10)
    early_stopping_min_delta = get_float_from_config(config, 'early_stopping_min_delta', 0.01)

    best_model_path = os.path.join(wandb.run.dir, 'model.pt')

    max_epochs = get_int_from_config(config, 'train_max_epochs', 1)
    num_batches = len(train_dataLoader)

    steps = 0

    previous_val_recall_at_1 = 0
    best_val_recall_dic = {
        1: 0,
    }
    best_test_recall_dic = {}
    not_improved_epochs = 0

    for epoch in range(1, max_epochs + 1):
        epoch_loss = 0.0
        
        predictions = []

        for x, y_target, y_idx in train_dataLoader:
            steps += 1

            x = x.to(device)
            y_target = y_target.to(device)

            optimizer.zero_grad()

            y = model(x)

            loss = criterion(y, y_target)
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                _, predIdx = torch.topk(y, top_k)
                predictions += zip(y_idx.numpy(), predIdx.cpu().numpy())

            batch_loss = loss.item()
        
            epoch_loss += batch_loss
        
        train_recalls = compute_recall(train_gt, predictions, len(predictions), n_values)
        
        avg_loss = epoch_loss / num_batches

        with torch.no_grad():
            val_recalls, _ = run_eval(model, val_dataLoader, eval_gt, n_values, top_k, device)

            current_val_recall_at_1 = val_recalls[1]

            is_better = current_val_recall_at_1 > best_val_recall_dic[1]

            if is_better:
                best_val_recall_dic = val_recalls
                run.summary["best_val_recall@1"] = best_val_recall_dic[1]
                log.info(f"Better val_recall@1 reached: {best_val_recall_dic[1]}")
                torch.save(model.state_dict(), best_model_path)

                best_test_recall_dic, _ = run_eval(model, test_dataLoader, eval_gt, n_values, top_k, device)

                log.info(f"Epoch {epoch}/{max_epochs} - Loss: {avg_loss:.8f} - Train Recall@1: {train_recalls[1]:.4f} - Val Recall@1: {best_val_recall_dic[1]:.4f} - Test Recall@1: {best_test_recall_dic[1]:.4f}")            
            else:
                log.info(f"Epoch {epoch}/{max_epochs} - Loss: {avg_loss:.8f} - Train Recall@1: {train_recalls[1]:.4f} - Val Recall@1: {val_recalls[1]:.4f}")

            if early_stopping_enabled:
                if (current_val_recall_at_1 - previous_val_recall_at_1) > early_stopping_min_delta:
                    not_improved_epochs = 0
                else:
                    not_improved_epochs += 1

            previous_val_recall_at_1 = current_val_recall_at_1

        log_dic = {'train_loss': avg_loss, "epoch": epoch}

        for k, v in train_recalls.items():
            log_dic[f"train_recall@{k}"] = v

        for k, v in val_recalls.items():
            log_dic[f"val_recall@{k}"] = v

        for k, v in best_test_recall_dic.items():
            log_dic[f"test_recall@{k}"] = v

        wandb.log(log_dic, step=steps)

        if early_stopping_enabled and early_stopping_patience > 0 and not_improved_epochs > (early_stopping_patience / 1):
            log.info(f'Performance did not improve for {early_stopping_patience} epochs. Stopping.')
            break

    # Save artifacts
    model_artifact = wandb.Artifact(f'hl_model_{run.id}', "model", metadata=config)
    model_artifact.add_file(best_model_path)
    wandb.log_artifact(model_artifact, aliases=["best"]) 

    # Finalise Summary
    for key in best_val_recall_dic:
        best_key = f"best_val_recall@{key}"
        log.info(f"Setting summary {best_key} reached: {best_val_recall_dic[key]}")
        run.summary[best_key] = best_val_recall_dic[key]

    for key in best_test_recall_dic:
        best_key = f"best_test_recall@{key}"
        log.info(f"Setting summary {best_key} reached: {best_test_recall_dic[key]}")
        run.summary[best_key] = best_test_recall_dic[key]
    
    run.finish()

def prepare_datasets(config):
    if config['dataset'] == 'nordland':
        train_dataset_info, val_test_dataset_info, train_netvlad_repr, val_test_netvlad_repr, val_test_splits = get_infos_for_nordland(config)
    elif config['dataset'] == 'oxford':
        train_dataset_info, val_test_dataset_info, train_netvlad_repr, val_test_netvlad_repr, val_test_splits = get_infos_for_oxford(config)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    
    train_gt = train_dataset_info['ground_truth_indices']
    train_image_idx = torch.from_numpy(train_dataset_info['image_indices'])
    image_1_hot = torch.from_numpy(get_1_hot_encode(train_dataset_info['image_indices'], len(train_dataset_info['image_indices']))).type(torch.float)
    
    # Normalize dataset
    max_n = train_netvlad_repr.max()
    _ = train_netvlad_repr.divide_(max_n)
    
    train_dataset = TensorDataset(train_netvlad_repr, image_1_hot, train_image_idx)
    print(f"Train dataset size: {len(train_dataset)}")
    train_dataLoader = DataLoader(train_dataset, num_workers=int(config['dataloader_threads']), batch_size=int(config['train_batchsize']), shuffle=True)

    eval_gt = val_test_dataset_info['ground_truth_indices']
    night_image_idx = torch.from_numpy(val_test_dataset_info['image_indices'])

    _ = val_test_netvlad_repr.divide_(max_n)
    val_test_dataset = TensorDataset(val_test_netvlad_repr, night_image_idx)

    val_dataset = Subset(val_test_dataset, val_test_splits['val_indices'])
    print(f"Val dataset size: {len(val_dataset)}")
    val_dataLoader = DataLoader(val_dataset, num_workers=int(config['dataloader_threads']), batch_size=int(config['train_batchsize']), shuffle=False)

    test_dataset = Subset(val_test_dataset, val_test_splits['test_indices'])
    print(f"Test dataset size: {len(test_dataset)}")
    test_dataLoader = DataLoader(test_dataset, num_workers=int(config['dataloader_threads']), batch_size=int(config['train_batchsize']), shuffle=False)

    return train_dataLoader, val_dataLoader, test_dataLoader, train_gt, eval_gt

def get_infos_for_nordland(config):
    train_dataset_info = load_np_file(config['dataset_oxford_day_dataset_file_path'])
    val_test_dataset_info = load_np_file(config['dataset_oxford_night_dataset_file_path'])

    train_netvlad_repr = torch.from_numpy(load_np_file(config['dataset_oxford_day_netvlad_repr_file_path']))
    val_test_netvlad_repr = torch.from_numpy(load_np_file(config['dataset_oxford_night_netvlad_repr_file_path']))

    val_test_splits = load_np_file(config['dataset_oxford_night_val_test_splits_indices_file_path'])

    return train_dataset_info, val_test_dataset_info, train_netvlad_repr, val_test_netvlad_repr, val_test_splits

def get_infos_for_oxford(config):
    train_dataset_info = load_np_file(config['dataset_oxford_day_dataset_file_path'])
    val_test_dataset_info = load_np_file(config['dataset_oxford_night_dataset_file_path'])

    train_netvlad_repr = torch.from_numpy(load_np_file(config['dataset_oxford_day_netvlad_repr_file_path']))
    val_test_netvlad_repr = torch.from_numpy(load_np_file(config['dataset_oxford_night_netvlad_repr_file_path']))
 
    val_test_splits = load_np_file(config['dataset_oxford_night_val_test_splits_indices_file_path'])

    return train_dataset_info, val_test_dataset_info, train_netvlad_repr, val_test_netvlad_repr, val_test_splits

if __name__ == '__main__':
    options, unknowns = parser.parse_known_args()
    main(options)
