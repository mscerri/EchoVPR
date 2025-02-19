import argparse
import logging
import os

import torch
from torch import nn

from configs.utils import (get_bool_from_config, get_config, get_value_from_namespace_or_raise, update_config)
from echovpr.models.single_esn import SingleESN
from echovpr.models.sparce_layer import SpaRCe
from echovpr.models.utils import get_sparsity
from echovpr.trainer.eval import run_eval
from echovpr.trainer.metrics.recall import compute_recall
from echovpr.trainer.prepare_esn_datasets import prepare_esn_datasets
from echovpr.trainer.prepare_final_datasets import (get_dataset_infos,
                                                    prepare_final_datasets)
from echovpr.trainer.process_patchnetvlad import local_matcher

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

parser = argparse.ArgumentParser(description='echovpr', argument_default=argparse.SUPPRESS)
parser.add_argument('--config_file', type=str, required=True, help='Location of config file to load defaults from')
parser.add_argument('--checkpoint_dir', type=str, required=True, help='Checkpoint directory where the saved models could be located')

parser.add_argument('--validation', type=str, default='True', help='Whether to run validation')

parser.add_argument('--model_reservoir_size', type=int, help='EchoStateNetwork parameter: Number of neurons')
parser.add_argument('--model_esn_num_connections', type=int, help='EchoStateNetwork parameter: Number of average recurrent connections per neuron')
parser.add_argument('--model_esn_alpha', type=float, help='EchoStateNetwork parameter: Alpha')
parser.add_argument('--model_esn_gamma', type=float, help='EchoStateNetwork parameter: Gamma')
parser.add_argument('--model_esn_rho', type=float, help='EchoStateNetwork parameter: Rho')
parser.add_argument('--model_sparce_enabled', type=str, help='EchoStateNetwork parameter: If Sparce layer is enabled')
parser.add_argument('--model_sparce_quantile', type=float, help='EchoStateNetwork parameter: Sparce quantile')

parser.add_argument('--patchnetvlad_config_file', type=str, help='Optional: Evaluate with Patch-NetVLAD. Location of config file to Patch-NetVLAD defaults')
parser.add_argument('--test_input_features_dir', type=str, help='Required when evaluating with Patch-NetVLAD:  Path to load all test patch-netvlad features')
parser.add_argument('--val_input_features_dir', type=str, help='Required when evaluating with Patch-NetVLAD: Path to load all validation patch-netvlad features')
parser.add_argument('--index_input_features_dir', type=str, help='Required when evaluating with Patch-NetVLAD: Path to load all database patch-netvlad features')

def main(options: argparse.Namespace):

    # Setup config
    config = get_config(options.config_file, logger=log)['main']
    
    config = update_config(config, options.__dict__)

    # Setup ESN
    in_features=int(config['model_in_features'])
    reservoir_size=int(config['model_reservoir_size'])
    out_features=int(config['model_out_features'])

    esn_alpha = float(config['model_esn_alpha'])
    esn_gamma = float(config['model_esn_gamma'])
    esn_rho = float(config['model_esn_rho'])
    esn_num_connections = int(config['model_esn_num_connections'])
    sparce_enabled = get_bool_from_config(config, 'model_sparce_enabled')
    
    model_esn = SingleESN(
        in_features, 
        reservoir_size, 
        alpha=esn_alpha, 
        gamma=esn_gamma, 
        rho=esn_rho,
        sparsity=get_sparsity(esn_num_connections, reservoir_size),
        device=device
    )

    esn_model_tensor = load_model(options.checkpoint_dir, 'esn_model.pt')
    model_esn.load_state_dict(esn_model_tensor)

    model = nn.ModuleDict()

    if sparce_enabled:
        model["sparce"] = SpaRCe(reservoir_size)

    model["out"] = nn.Linear(in_features=reservoir_size, out_features=out_features, bias=True)

    model_tensor = load_model(options.checkpoint_dir, 'model.pt')
    model.load_state_dict(model_tensor, strict=False)

    # Move to device
    model_esn.eval().to(device)
    model.eval().to(device)

    # Load datasets, normalize and process through ESN

    esn_descriptors = prepare_esn_datasets(model_esn, config, device, log, eval_only=True)

    del model_esn
    
    torch.cuda.empty_cache()

    _, _, val_dataset, val_dataLoader, test_dataLoader, _, eval_gt = prepare_final_datasets(esn_descriptors, config, eval_only=True)

    val_dataset_quantiles = None
    if sparce_enabled:
        # Calculate Training Dataset Quantiles
        quantile = float(config['model_sparce_quantile'])
        val_dataset_quantiles = torch.quantile(torch.abs(torch.vstack([t[0] for t in val_dataset])), quantile, dim=0).to(device)
    
    n_values = [1, 5, 10, 20, 50, 100]
    top_k = max(n_values)
    
    if get_bool_from_config(config, 'validation', True):
        _, val_predictions = run_eval(model, val_dataLoader, eval_gt, n_values, top_k, device, model_forward=model_forward, sparce_enabled=sparce_enabled, dataset_quantiles=val_dataset_quantiles)
        val_recalls = compute_recall(eval_gt, val_predictions, len(val_predictions), n_values, print_recall=True, recall_str='Eval on Validation Set')
    
    _, test_predictions = run_eval(model, test_dataLoader, eval_gt, n_values, top_k, device, model_forward=model_forward, sparce_enabled=sparce_enabled, dataset_quantiles=val_dataset_quantiles)
    test_recalls = compute_recall(eval_gt, test_predictions, len(test_predictions), n_values, print_recall=True, recall_str='Eval on Test Set')

    if 'patchnetvlad_config_file' in options:
        patchnetvlad_config = get_config(options.patchnetvlad_config_file, log)
        train_dataset_info, val_test_dataset_info = get_dataset_infos(config)

        input_index_local_features_prefix = os.path.join(get_value_from_namespace_or_raise(options, 'index_input_features_dir'), 'patchfeats')

        if get_bool_from_config(config, 'validation', True):
            input_val_local_features_prefix = os.path.join(get_value_from_namespace_or_raise(options, 'val_input_features_dir'), 'patchfeats')
            
            reranked_val_predictions = local_matcher(val_predictions, patchnetvlad_config, input_val_local_features_prefix, input_index_local_features_prefix, val_test_dataset_info, train_dataset_info, device)
            val_patch_recalls = compute_recall(eval_gt, reranked_val_predictions, len(val_predictions), n_values, print_recall=True, recall_str='PatchNetVLAD Eval on Validation Set')

        input_test_local_features_prefix = os.path.join(get_value_from_namespace_or_raise(options, 'test_input_features_dir'), 'patchfeats')

        reranked_test_predictions = local_matcher(test_predictions, patchnetvlad_config, input_test_local_features_prefix, input_index_local_features_prefix, val_test_dataset_info, train_dataset_info, device)
        test_patch_recalls = compute_recall(eval_gt, reranked_test_predictions, len(test_predictions), n_values, print_recall=True, recall_str='PatchNetVLAD Eval on Test Set')


def load_model(model_dir: str, model_name: str) -> str:
    return torch.load(os.path.join(model_dir, model_name))

def model_forward(model, x, kwargs):
    if kwargs['sparce_enabled']:
        x = model["sparce"](x, kwargs['dataset_quantiles'])
    
    y = model["out"](x)
    return y

if __name__ == '__main__':
    options, unknowns = parser.parse_known_args()
    main(options)
