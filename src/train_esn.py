import argparse
import logging
import os

import numpy as np
import torch
from torch import nn

import wandb
from configs.utils import (get_bool_from_config, get_config_wandb,
                           get_float_from_config, get_int_from_config,
                           get_value_from_namespace, update_config_wandb)
from echovpr.models.single_esn import SingleESN
from echovpr.models.sparce_layer import SpaRCe
from echovpr.models.utils import get_sparsity
from echovpr.trainer.eval import run_eval
from echovpr.trainer.metrics.recall import compute_recall
from echovpr.trainer.prepare_esn_datasets import prepare_esn_datasets
from echovpr.trainer.prepare_final_datasets import prepare_final_datasets

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

parser.add_argument('--model_reservoir_size', type=int, help='EchoStateNetwork parameter: Number of neurons')
parser.add_argument('--model_esn_num_connections', type=int, help='EchoStateNetwork parameter: Number of average recurrent connections per neuron')
parser.add_argument('--model_esn_alpha', type=float, help='EchoStateNetwork parameter: Alpha')
parser.add_argument('--model_esn_gamma', type=float, help='EchoStateNetwork parameter: Gamma')
parser.add_argument('--model_esn_rho', type=float, help='EchoStateNetwork parameter: Rho')

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
    reservoir_size=int(config['model_reservoir_size'])
    out_features=int(config['model_out_features'])

    esn_alpha = float(config['model_esn_alpha'])
    esn_gamma = float(config['model_esn_gamma'])
    esn_rho = float(config['model_esn_rho'])
    esn_num_connections = int(config['model_esn_num_connections'])
    sparce_enabled = get_bool_from_config(config, 'model_sparce_enabled')

    # Init models
    esn_path = os.path.join(wandb.run.dir, 'esn_model.pt')

    model_esn = SingleESN(
        in_features, 
        reservoir_size, 
        alpha=esn_alpha, 
        gamma=esn_gamma, 
        rho=esn_rho,
        sparsity=get_sparsity(esn_num_connections, reservoir_size),
        device=device
    )
    
    # Save ESN Model

    model_artifact = wandb.Artifact(f'esn_{run.id}', "model", metadata=config)
    torch.save(model_esn.state_dict(), esn_path)
    model_artifact.add_file(esn_path)

    # Move to device

    model_esn.to(device)
    
    # Load datasets, normalize and process through ESN

    esn_descriptors = prepare_esn_datasets(model_esn, config, device, log)

    del model_esn
    
    torch.cuda.empty_cache()

    # Prepare Final Datasets
    train_dataset, train_dataLoader, val_dataset, val_dataLoader, test_dataLoader, train_gt, eval_gt = prepare_final_datasets(esn_descriptors, config)

    train_dataset_quantiles = None
    val_dataset_quantiles = None
    if sparce_enabled:
        # Calculate Training Dataset Quantiles
        quantile = float(config['model_sparce_quantile'])
        train_dataset_quantiles = torch.quantile(torch.abs(train_dataset.tensors[0]), quantile, dim=0).to(device)
        val_dataset_quantiles = torch.quantile(torch.abs(torch.vstack([t[0] for t in val_dataset])), quantile, dim=0).to(device)
    
    model = nn.ModuleDict()

    if sparce_enabled:
        model["sparce"] = SpaRCe(reservoir_size)

    model["out"] = nn.Linear(in_features=reservoir_size, out_features=out_features, bias=True)

    model.to(device)

    optimizer_params = []

    lr = float(config['train_lr'])    

    if sparce_enabled:
        lr_sparce = lr / get_int_from_config(config, 'train_lr_sparce_divide_by', 1000)
        optimizer_params.append({'params': model["sparce"].parameters(), 'lr': lr_sparce})

    optimizer_params.append({'params': model["out"].parameters()})

    optimizer = torch.optim.Adam(optimizer_params, lr=lr)
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

            if sparce_enabled:
                x = model["sparce"](x, train_dataset_quantiles)
            
            y = model["out"](x)

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
            val_recalls, _ = run_eval(model, val_dataLoader, eval_gt, n_values, top_k, device, model_forward=model_forward, sparce_enabled=sparce_enabled, dataset_quantiles=val_dataset_quantiles)

            current_val_recall_at_1 = val_recalls[1]

            is_better = current_val_recall_at_1 > best_val_recall_dic[1]

            if is_better:
                best_val_recall_dic = val_recalls
                run.summary["best_val_recall@1"] = best_val_recall_dic[1]
                log.info(f"Better val_recall@1 reached: {best_val_recall_dic[1]}")
                torch.save(model.state_dict(), best_model_path)

                best_test_recall_dic, _ = run_eval(model, test_dataLoader, eval_gt, n_values, top_k, device, model_forward=model_forward, sparce_enabled=sparce_enabled, dataset_quantiles=val_dataset_quantiles)

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

def model_forward(model, x, kwargs):
    if kwargs['sparce_enabled']:
        x = model["sparce"](x, kwargs['dataset_quantiles'])
    
    y = model["out"](x)
    return y

if __name__ == '__main__':
    options, unknowns = parser.parse_known_args()
    main(options)
