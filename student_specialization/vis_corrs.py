import numpy as np
import torch

def get_stat(w):
    # return f"min: {w.min()}, max: {w.max()}, mean: {w.mean()}, std: {w.std()}" 
    if isinstance(w, list):
        w = np.array(w)
    elif isinstance(w, torch.Tensor):
        w = w.cpu().numpy()
    return f"len: {w.shape}, min/max: {np.min(w):#.3f}/{np.max(w):#.3f}, mean: {np.mean(w):#.3f}" 

def get_corrs(corrs, active_nodes=None, first_n=5, cnt_thres=0.9, details=False):
    summary = ""
    for k, corr_per_layer in enumerate(corrs):
        score = []
        cnts = []
        for kk, corr_per_node in enumerate(corr_per_layer):
            if active_nodes is not None and kk not in active_nodes[k]:
                continue

            # Get best score for each teacher
            if isinstance(corr_per_node, list):
                s = [ c["score"] for c in corr_per_node ]
            else:
                s = corr_per_node["s_score"]
            score.append(s[0])
            if cnt_thres is not None:
                cnt = sum([ ss >= cnt_thres for ss in s ])
                cnts.append(cnt)

        summary += f"L{k}: {get_stat(score)}"
        if cnt_thres is not None:
            summary += f", MatchCnt[>={cnt_thres}]: {get_stat(cnts)}"
        summary += "\n"

    output = ""
    output += f"Corrs Summary:\n{summary}"

    if details:
        output += "\n"
        for k, corr_per_layer in enumerate(corrs):
            # For each layer
            output += "Layer %d\n" % k
            for j, corr_per_node in enumerate(corr_per_layer):
                s_score = corr_per_node["s_score"][:first_n]
                s_idx = corr_per_node["s_idx"][:first_n]

                s_score_str = ",".join(["%.4f" % v for v in s_score])
                s_idx_str = ",".join(["%2d [%s]" % (node_id, rank) for node_id, rank in s_idx])
                # import pdb
                # pdb.set_trace()

                min_rank = min([ int(rank) for node_id, rank in s_idx ])
                output += "T[%d]: [init_best_s=%.4f] %s | idx: %s | min_rank: %d\n" % (j, corr_per_node["max_init_score"], s_score_str, s_idx_str, min_rank)
                # print("T[%d]: [init_best_s=%.4f] %s | idx: %s " % (j, corr_per_node["max_init_score"], s_score_str, s_idx_str))

    return output

