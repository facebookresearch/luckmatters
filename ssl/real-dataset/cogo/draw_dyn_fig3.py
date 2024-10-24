import common_utils
import os
import sys
import argparse
import matplotlib.pyplot as plt
import torch

from analyze_util import load_model_traj

def compute_mps(As, Bs, Cs):
    # Compute r_{k1k2k}
    rkkks = torch.einsum('kjt,ljt,njt->klnt', As, Bs, Cs)
    
    d = As.shape[0]

    indices = list(range(d))
    ramkks = []
    rbmkks = []
    # Compute rotation.
    for m in range(d):
        ramkks.append(torch.einsum('kjt,njt->knt', As * As[indices,:,:].conj(), Cs))
        rbmkks.append(torch.einsum('kjt,njt->knt', Bs * Bs[indices,:,:].conj(), Cs))
        indices = indices[1:] + [indices[0]]


    # Compute r_pmk'k
    ramkks = torch.stack(ramkks, dim=0)
    rbmkks = torch.stack(rbmkks, dim=0)
    
    return dict(rkkks=rkkks, ramkks=ramkks, rbmkks=rbmkks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # The root directory of an individual experiment
    parser.add_argument("root", type=str)
    args = parser.parse_args()
    root = args.root

    cfg = common_utils.MultiRunUtil.load_cfg(root)
    cfg = { entry.split("=")[0] : entry.split("=")[1] for entry in cfg }
    print(cfg)

    As, Bs, Cs, ts, final_results = load_model_traj(root)
    d = As.shape[0]

    import pandas as pd

    df = pd.DataFrame(final_results)

    first_t = 200

    #plt.figure(figsize=(4,2))

    fig, ax1 = plt.subplots(figsize=(5,3))

    h1, = ax1.plot(df["epoch"][:first_t], df["test_loss"][:first_t], "--", label="test_loss")
    h2, = ax1.plot(df["epoch"][:first_t], df["train_loss"][:first_t], "--", label="train_loss")
    ax1.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")

    ax2 = ax1.twinx()
    h3, = ax2.plot(df["epoch"][:first_t], df["test_acc"][:first_t], label="test_acc")
    h4, = ax2.plot(df["epoch"][:first_t], df["train_acc"][:first_t], label="train_acc")
    ax2.axhline(1.0, color='k', linestyle='--', linewidth=0.5)
    ax2.set_ylabel("Accuracy")

    plt.legend(handles=[h1,h2,h3,h4], loc="center right")
    plt.title(f"Training/test loss/accuracy for d = {d}")
    plt.tight_layout()
    plt.savefig(f"m{d}-training.pdf")
    # plt.show()

    freq_patterns = (As[:(d-1)//2+1,:,-1].abs() > 0.05).sum(dim=1)
    print(freq_patterns)

    plt.figure(figsize=(5,3))

    import numpy as np
    bin_edges = np.linspace(-0.5, 10.5, 12)
    plt.hist(freq_patterns[1:][:], bins=bin_edges)
    plt.xlabel("Solution order at all frequencies")
    plt.ylabel("Count")
    plt.title(f"Distribution of Solution order at 10k epochs")
    plt.tight_layout()
    plt.savefig(f"m{d}-order-distri.pdf")

    data = compute_mps(As, Bs, Cs)
    # multiple things by 2d to make sure it fits to our setup
    rkkks = data["rkkks"] * 2 * d
    ramkks = data["ramkks"] * 2 * d
    rbmkks = data["rbmkks"] * 2 * d

    first_t = 205

    diag_dyn = 0
    for i in range(d):
        diag_dyn += rkkks[i,i,i,:first_t].real
        
    # diag_dyn /= d
        
    off_diag_dyn = 0
    for i1 in range(d):
        for i2 in range(d):
            for i3 in range(d):
                if i1 == i2 == i3:
                    continue
                off_diag_dyn += rkkks[i1,i2,i3,:first_t].real
    # off_diag_dyn /= (d**3 - d)

    plt.figure(figsize=(5,3))

    plt.plot(ts[:first_t], diag_dyn, label="Diag $r_{kkk}$")
    plt.plot(ts[:first_t], off_diag_dyn, label = "Off-diag $r_{k_1k_2k}$")
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.axhline(d - 1, color='k', linestyle='--', linewidth=0.5)

    print("Diag_dyn")
    print(diag_dyn.argmax())
    print(diag_dyn.max())
    print("")

    print("Off_Diag_dyn")
    print(off_diag_dyn.argmax())
    print(off_diag_dyn.max())
    print(off_diag_dyn.argmin())
    print(off_diag_dyn.min())

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Summed $r_{k_1k_2k}$")
    plt.title("Dynamics of diag/off-diag $r_{k_1k_2k}$")
    plt.tight_layout()
    plt.savefig(f"m{d}-dyn-r.pdf")
    # plt.show()

    count_4 = (freq_patterns == 4).sum().item() // 2
    count_6 = (freq_patterns == 6).sum().item() // 2

    plt.figure(figsize=(5,3))
    for k in range(1, (d-1) // 2 + 1):
        if freq_patterns[k] == 4:
            # plt.plot(ts, rbmkks[0,:,k].sum(dim=0).abs(), label=f"k={k}")
            plt.plot(ts, ramkks[0,k,k].abs(), label=f"k={k}")
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.xlabel("Epoch")
    symbol = "$r_{a,k,-k,k}$"

    plt.ylabel(symbol)
    plt.title(f"{symbol} increases in order-4 solution")
    plt.tight_layout()
    plt.savefig(f"m{d}-dyn-order-4.pdf")

            # plt.subplot(2, count_4, cnt + count_4)
            # plt.plot(rbmkks[0,k,k].abs())
            # plt.title(f"freq {k}")
            
    plt.figure(figsize=(5,3))
    for k in range(1, (d-1) // 2 + 1):
        if freq_patterns[k] == 6:
            plt.plot(ts, ramkks[0,k,k].abs(), label=f"k={k}")
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(symbol)
    plt.title(f"{symbol} vanishes in order-6 solution")
    plt.tight_layout()
    plt.savefig(f"m{d}-dyn-order-6.pdf")
    # plt.show()

            
            # plt.subplot(2, count_6, cnt + count_6)
            # plt.plot(rbmkks[0,k,k].abs())
            # plt.title(f"freq {k}")
    first_t = 220
    m = 1
    symbol = "$r_{a,k," + str(m) + "-k,k}$"

    plt.figure(figsize=(5,3))
    for k in range(1, (d-1) // 2 + 1):
        plt.plot(ts[:first_t], ramkks[m,k,k].abs()[:first_t], label=f"k={k}")
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    # plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(symbol)
    plt.title(f"{symbol} vanishes in both order-4 and order-6")
    plt.tight_layout()
    plt.savefig(f"m{d}-dyn-a{m}kk.pdf")


