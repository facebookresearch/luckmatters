import common_utils
import os
import sys
import glob
import numpy as np
# Now run parallel loop
import matplotlib.pyplot as plt
import argparse
import torch

from analyze_util import load_all, load_model_traj
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # The root directory after you run the parameter sweep
    # e.g.,  

    parser.add_argument("root", type=str)

    args = parser.parse_args()
    root = args.root

    all_cfgs = load_all(root)

    all_results = []

    # load the experimental results.
    freq_map = dict()

    q = "512"
    for (M, weight_decay), tbl in all_cfgs[all_cfgs["hidden_size"] == q].groupby(["M", "weight_decay"]):
        # compute their statistics
        print(M, weight_decay)
        freq_patterns = []
        for folder in tbl["folder"]:
            print(folder)
            try:
                # Load them
                # data = torch.load(os.path.join(folder, "data.pth"), map_location="cpu")
                As, Bs, Cs, ts, _ = load_model_traj(folder, indices=[9900])

                d = As.shape[0]
                # compute the freq pattern. 
                freq_patterns.append((As[1:(d-1)//2+1,:,-1].abs() > 0.05).sum(dim=1))
            except:
                pass
        # 
        if len(freq_patterns) > 0:
            freq_map[(M, weight_decay)] = torch.cat(freq_patterns)

    bin_edges = np.linspace(-0.5, 10.5, 12)

    Ms = ["23", "71", "127"]
    M_reverse = { M : i for i, M in enumerate(Ms) }

    wds = ["1e-05", "5e-05", "0.0001", "0.0002", "0.0005"]
    wd_reverse = { h : i for i, h in enumerate(wds) }

    plt.figure(figsize=(12,5))
    for k, v in freq_map.items():
        try:
            row_idx = M_reverse[k[0]]
            col_idx = wd_reverse[k[1]]
        except:
            continue
        
        plt.subplot(3, 5, row_idx * 5 + col_idx + 1)
        plt.hist(v, bins=bin_edges)
        plt.axvline(4, color="r", linestyle='--')
        plt.axvline(6, color="r", linestyle='--')
        plt.title(f"d={k[0]}, wd={k[1]}")
        if col_idx == 0:
            plt.ylabel("Counts")
        if row_idx != 2:
            plt.xticks([])
        else:
            plt.xlabel("Solution order")
            
    plt.tight_layout()
    # Draw figure 4 in the paper
    plt.savefig(f"solution_distri_{q}.pdf")
