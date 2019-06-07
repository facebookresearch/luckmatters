import torch

def corrMat2corrIdx(score):
    ''' Given score[N, #candidate], 
        for each sample sort the score (in descending order) 
        and output corr_table[N, #dict(s_idx, score)]
    '''
    sorted_score, sorted_indices = score.sort(1, descending=True)
    N = sorted_score.size(0)
    n_candidate = sorted_score.size(1)
    # Check the correpsonding weights.
    # print("For each teacher node, sorted corr over all student nodes at layer = %d" % i)
    corr_table = []
    for k in range(N):
        tt = []
        for j in range(n_candidate):
            # Compare the upward weights.
            s_idx = int(sorted_indices[k][j])
            score = float(sorted_score[k][j])
            tt.append(dict(s_idx=s_idx, score=score))
        corr_table.append(tt)
    return corr_table

'''
def corrIdx2pickMat(corr_indices):
    return [ item[0]["s_idx"] for item in corr_indices ]

def corrIndices2pickMats(corr_indices_list):
    return [ corrIdx2pickMat(corr_indices) for corr_indices in corr_indices_list ]
'''

def act2corrMat(src, dst):
    ''' src[:, k], with k < K1
        dst[:, k'], with k' < K2 
        output correlation score[K1, K2]
    '''
    # K_src by K_dst
    if len(src.size()) == 3 and len(dst.size()) == 3:
        src = src.permute(0, 2, 1).contiguous().view(src.size(0) * src.size(2), -1)
        dst = dst.permute(0, 2, 1).contiguous().view(dst.size(0) * dst.size(2), -1)

    # conv activations.
    elif len(src.size()) == 4 and len(dst.size()) == 4:
        src = src.permute(0, 2, 3, 1).contiguous().view(src.size(0) * src.size(2) * src.size(3), -1)
        dst = dst.permute(0, 2, 3, 1).contiguous().view(dst.size(0) * dst.size(2) * dst.size(3), -1)

    # Substract mean.
    src = src - src.mean(0, keepdim=True)
    dst = dst - dst.mean(0, keepdim=True)

    inner_prod = torch.mm(src.t(), dst)
    src_inv_norm = src.pow(2).sum(0).add_(1e-10).rsqrt().view(-1, 1)
    dst_inv_norm = dst.pow(2).sum(0).add_(1e-10).rsqrt().view(1, -1)

    return inner_prod * src_inv_norm * dst_inv_norm

def acts2corrMats(hidden_t, hidden_s):
    # Match response
    ''' Output correlation matrix for each layer '''
    corrs = []
    for t, s in zip(hidden_t, hidden_s):
        corr = act2corrMat(t.data, s.data)
        corrs.append(corr)
    return corrs

def acts2corrIndices(hidden_t, hidden_s):
    # Match response
    ''' Output correlation indices for each layer '''
    corrs = []
    for t, s in zip(hidden_t, hidden_s):
        corr = act2corrMat(t.data, s.data)
        corrs.append(corrMat2corrIdx(corr))
    return corrs

'''
w_t = getattr(teacher, "w%d" % (i + 1)).weight
w_s = getattr(student, "w%d" % (i + 1)).weight
w_teacher=w_t[:,k], w_student=w_s[:, s_idx]
'''

def compareCorrIndices(init_corrs, final_corrs):
    res = []
    for k, (init_corr, final_corr) in enumerate(zip(init_corrs, final_corrs)):
        # For each layer
        # print("Layer %d" % k)
        res_per_layer = []

        for j, (init_node, final_node) in enumerate(zip(init_corr, final_corr)):
            # For each node
            ranks = dict()
            max_init_score = -1000
            for node_rank, node_info in enumerate(init_node):
                node_id = node_info["s_idx"]
                score = node_info["score"]
                ranks[node_id] = dict(rank=node_rank, score=score)
                max_init_score = max(max_init_score, score)

            s_score = []
            s_idx = []
            for node_info in final_node:
                node_id = node_info["s_idx"]
                if node_id in ranks:
                    rank = ranks[node_id]["rank"]
                else:
                    rank = "-"
                s_score.append(node_info["score"])
                s_idx.append((node_id, str(rank)))
                # "%2d [%s]" % (node_id, str(rank)))
            # print("T[%d]: [init_student_max=%.4f] %s   | idx: %s | min_rank: %d" % (j, max_init_corr, ",".join(s_val), ", ".join(s_idx), min_rank))
            res_per_layer.append(dict(s_score=s_score, s_idx=s_idx, max_init_score=max_init_score))
        res.append(res_per_layer)

    return res

