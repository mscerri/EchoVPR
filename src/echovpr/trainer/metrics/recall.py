import numpy as np

def compute_recall(gt, predictions, numQ, n_values, print_recall = False, recall_str = None):
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in predictions:
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / numQ
    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        if print_recall:
            print("====> {} - Recall@{}: {:.4f}".format(recall_str, n, recall_at_n[i]))
    return all_recalls