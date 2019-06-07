import numpy as np
import torch

def get_stat(w):
    # return f"min: {w.min()}, max: {w.max()}, mean: {w.mean()}, std: {w.std()}" 
    if isinstance(w, list):
        w = np.array(w)
    elif isinstance(w, torch.Tensor):
        w = w.cpu().numpy()
    return f"len: {w.shape}, min/max: {np.min(w):#3f}/{np.max(w):#3f}, mean: {np.mean(w):#3f}" 

def print_corrs(corrs, active_nodes=None, first_n=5, details=False):
    summary = ""
    for k, corr_per_layer in enumerate(corrs):
        score = []
        for kk, corr_per_node in enumerate(corr_per_layer):
            if active_nodes is None or kk in active_nodes[k]:
                score.append(corr_per_node["s_score"][0])

        summary += f"L{k}: {get_stat(score)}, "

    print(f"Corrs Summary: {summary}")

    if details:
        for k, corr_per_layer in enumerate(corrs):
            # For each layer
            print("Layer %d" % k)
            for j, corr_per_node in enumerate(corr_per_layer):
                s_score = corr_per_node["s_score"][:first_n]
                s_idx = corr_per_node["s_idx"][:first_n]

                s_score_str = ",".join(["%.4f" % v for v in s_score])
                s_idx_str = ",".join(["%2d [%s]" % (node_id, rank) for node_id, rank in s_idx])
                # import pdb
                # pdb.set_trace()

                min_rank = min([ int(rank) for node_id, rank in s_idx ])
                print("T[%d]: [init_best_s=%.4f] %s | idx: %s | min_rank: %d" % (j, corr_per_node["max_init_score"], s_score_str, s_idx_str, min_rank))
                # print("T[%d]: [init_best_s=%.4f] %s | idx: %s " % (j, corr_per_node["max_init_score"], s_score_str, s_idx_str))

