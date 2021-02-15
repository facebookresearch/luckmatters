import numpy as np
import torch
import torch.nn as nn
import pprint
from collections import Counter, defaultdict

def accumulate(a, v):
    return a + v if a is not None else v


def get_stat(w):
    # return f"min: {w.min()}, max: {w.max()}, mean: {w.mean()}, std: {w.std()}" 
    if isinstance(w, list):
        w = np.array(w)
    elif isinstance(w, torch.Tensor):
        w = w.cpu().numpy()
    return f"len: {w.shape}, min/max: {np.min(w):#.6f}/{np.max(w):#.6f}, mean/median: {np.mean(w):#.6f}/{np.median(w):#.6f}" 


def corr_summary(corr, row_names=None, col_names=None, best_corr_last=None, verbose=False, cnt_thres=None):
    # Corrs: [row, col]
    sorted_corrs, indices = corr.sort(1, descending=True)
    num_row, num_col = corr.size()

    if row_names is None:
        row_names = [ f"row{i}" for i in range(num_row) ]

    if col_names is None:
        col_names = [ f"col{i}" for i in range(num_col) ]

    assert num_row == len(row_names), f"#row: {num_row} != #row_names {len(row_names)}"
    assert num_col == len(col_names), f"#col: {num_col} != #col_names {len(col_names)}"

    best_corr = sorted_corrs[:,0] 
    if best_corr_last is not None:
        best_corr_diff = best_corr - best_corr_last
    _, row_orders = best_corr.sort(0, descending=True)

    # Teacher side.
    summaries = []
    for i in row_orders:
        row_name = row_names[i]
        s = sorted_corrs[i]
        best = indices[i][0]
        best_score = sorted_corrs[i][0]
        comment = f"[{i}] {row_name}: best: {col_names[best]} ({best_score:.6f})" 
        if best_corr_last is not None:
            comment += f" delta: {best_corr_diff[i]:.6f}"
        summaries.append(comment)

    return dict(summary="\n".join(summaries), best_corr=best_corr)


class StatsCorr:
    def reset(self):
        self.initialized = False

    def add(self, h_t, h_s):
        if not self.initialized:
            self.inner_prod = None
            self.sum_t = None
            self.sum_s = None
            self.sum_sqr_t = None
            self.sum_sqr_s = None
            self.counts = 0
            self.initialized = True

        # Compute correlation.
        # activation: [bs, #nodes]
        h_t = h_t.detach()
        h_s = h_s.detach()

        if h_t.dim() == 4:
            h_t = h_t.permute(0, 2, 3, 1).reshape(-1, h_t.size(1))
        if h_s.dim() == 4:
            h_s = h_s.permute(0, 2, 3, 1).reshape(-1, h_s.size(1))

        self.inner_prod = accumulate(self.inner_prod, h_t.t() @ h_s)

        self.sum_t = accumulate(self.sum_t, h_t.sum(dim=0))
        self.sum_s = accumulate(self.sum_s, h_s.sum(dim=0))

        self.sum_sqr_t = accumulate(self.sum_sqr_t, h_t.pow(2).sum(dim=0))
        self.sum_sqr_s = accumulate(self.sum_sqr_s, h_s.pow(2).sum(dim=0))

        self.counts += h_t.size(0)

    def get(self):
        assert self.initialized

        n = self.counts
        s_avg = self.sum_s / n
        t_avg = self.sum_t / n

        ts_centered = self.inner_prod / n - torch.ger(t_avg, s_avg)

        t_var = self.sum_sqr_t / n - t_avg.pow(2)
        s_var = self.sum_sqr_s / n - s_avg.pow(2)

        t_var.clamp_(0, None)
        s_var.clamp_(0, None)

        t_norm = t_var.sqrt()
        s_norm = s_var.sqrt()
        
        ts_norm = torch.ger(t_norm, s_norm)
        corr = ts_centered / ts_norm
        # Set 0/0 = 0
        # usually that means everything is constant (and super sparse), and we don't know the correlation
        zero_entry = (ts_norm < 1e-6) & (ts_centered.abs() < 1e-6)
        corr[zero_entry] = 0.0

        corr.clamp_(-1, 1)

        # corr: [#node_t, #node_s]
        return dict(corr=corr, s_norm=s_norm, t_norm=t_norm)
