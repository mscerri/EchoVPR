import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


def process_dataloader(model: nn.Module, dataLoader: DataLoader, device: torch.device):
    with torch.no_grad():
        x_processed_list = []
        
        for x in dataLoader:
            
            x = x.to(device)
            
            x_processed = model(x)

            x_processed_list.append(x_processed.cpu())

        return torch.vstack(x_processed_list)

def process_dataset(ds_name: str, dataset: Dataset, model, config, device: torch.device, logger, batchsize=None, dataLoader_threads=None):
    batchsize = batchsize if batchsize is not None else int(config['train_batchsize'])
    dataLoader_threads = dataLoader_threads if dataLoader_threads is not None else int(config['dataloader_threads'])
    
    dataLoader = DataLoader(dataset, num_workers=dataLoader_threads, batch_size=batchsize, shuffle=False)

    logger.info(f"Start processing {ds_name} dataset")

    processed_x = process_dataloader(model, dataLoader, device)

    del dataLoader

    torch.cuda.empty_cache()

    logger.info(f"Finished processing {ds_name} Dataset")
    
    return processed_x
