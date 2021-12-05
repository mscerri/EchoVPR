import torch
from echovpr.trainer.metrics.recall import compute_recall


def run_eval(model, dataLoader, gt, n_values, top_k, device, model_forward = None, **kwargs):
    with torch.no_grad():
        predictions = []
        
        for x, y_idx in dataLoader:

            x = x.to(device)
            
            if model_forward is None:
                y = model(x)
            else:
                y = model_forward(model, x, kwargs)

            _, predIdx = torch.topk(y, top_k)
            predictions += zip(y_idx.numpy(), predIdx.cpu().numpy())

        return (compute_recall(gt, predictions, len(predictions), n_values), predictions)
