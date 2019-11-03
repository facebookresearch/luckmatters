import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import sys
import pandas as pd

import argparse

import re
import torch
import json
import math

from utils import find_params, load_data

def figure_l_shape(data):
    # Figure 1. L-shape with 2-layer network. 
    multis = (1, 2, 5, 10)
    decays = (0, 1, 2)
    num_teacher = 10

    plt.figure(figsize=(15, 10))

    counter = 1
    for decay in decays:
        for multi in multis:
            plt.subplot(3, len(multis), counter)
            counter += 1

            d = find_params(data, dict(multi=multi, teacher_strength_decay=decay, m=num_teacher))

            # print("multi: ", d["args"]["multi"])
            # print("decay: ", d["args"]["teacher_strength_decay"])

            losses = []

            for seed, stats in d["stats"].items():
                s = stats[-1]

                corrs = s["corr_train"]

                norms = s["W2_s"].norm(dim=1)
                norms = norms[:-1]

                plt.scatter(corrs.max(dim=0)[0], norms)
                losses.append(s["eval_loss"])
                
            if decay == 2:
                plt.xlabel('Correlation to the best correlated teacher')
            
            if multi == 1:
                plt.ylabel('norm of fan-out weights')
            # plt.title(f"{multi}x, loss={sum(losses) / len(losses):#.2f}")
            
            if decay == 0: 
                plt.title(f"{multi}x")
            plt.axis([0.0, 1.1, -0.1, 2.5])
    
    plt.savefig(f"l-shape-m{num_teacher}.pdf")
    # plt.show()

def figure_success_rate(data):
    multis = (1, 2, 5, 10)
    thres = 0.95
    num_teacher = 20

    plt.figure(figsize=(12, 2.5))
    # plt.figure()

    counter = 0

    # fig, ax = plt.subplots(figsize=(6, 5))
    for decay in (0.5, 1, 1.5, 2, 2.5):
        ax = plt.subplot(1, 5, counter + 1)
        counter += 1
        for iter, style in zip((5, -1), (':', '-')):
            bars = []
            ind = torch.FloatTensor(list(range(num_teacher)))
            # width = 0.15
            colors = ['r', 'g','b','c']
            for i, multi in enumerate(multis):
                #plt.subplot(1, len(multis), counter)
                #counter += 1

                d = find_params(data, dict(multi=multi, teacher_strength_decay=decay, m=num_teacher))

                losses = []

                counts = None
                for seed, stats in d["stats"].items():
                    s = stats[iter]
                    v = (s["counts_eval"][thres] > 0).float()
                    if counts is None:
                        counts = v
                    else:
                        counts += v

                    losses.append(s["eval_loss"])

                counts /= len(d["stats"])
                plt.plot(ind.numpy(), counts.numpy(), colors[i], label=f"{multi}x" if iter == -1 else None, linestyle=style)
                # plt.scatter(ind.numpy(), counts.numpy(), color=colors[i])

            # plt.title(f"multi={multi}, loss={sum(losses) / len(losses):#.5f}")
            # plt.title(f"iter={iter}")

        plt.xlabel('Teacher idx')
        plt.title(f"$p={decay}$")
        plt.axis([-1, num_teacher, 0, 1.1])
        if counter == 1:
            plt.ylabel('Successful Recovery Rate')
            plt.legend()
        
        ticks = ind[::4].numpy()

        ax.set_xticks(ticks)
        ax.set_xticklabels([ str(int(i)) for i in ticks ])
        if counter > 1:
            ax.set_yticklabels([])

            # ax.legend(bars, [ f"{multi}x" for multi in multis ])
            
    plt.tight_layout()
        
    plt.savefig(f"rate_drop_m{num_teacher}_thres{thres}.pdf")
    # plt.show()

def figure_loss(data):
    multis = (1, 2, 5, 10)
    decays = (0, 0.5, 1, 1.5, 2, 2.5)
    num_teacher = 20

    plt.figure(figsize=(15, 7))
    # plt.figure()

    counter = 1

    # fig, ax = plt.subplots(figsize=(6, 5))
    for decay in decays:
        ax = plt.subplot(2, len(decays) / 2, counter)
        counter += 1
        for i, multi in enumerate(multis):
            d = find_params(data, dict(multi=multi, teacher_strength_decay=decay, m=num_teacher))
            losses = None
            for j, (seed, stats) in enumerate(d["stats"].items()):
                v = torch.DoubleTensor([ math.log(s["eval_loss"]) / math.log(10.0) for s in stats ])
                if losses is None:
                    losses = torch.DoubleTensor(len(stats), len(d["stats"]))
                losses[:, j] = v
                
            loss = losses.mean(dim=1)
            loss_std = losses.std(dim=1)
            p = plt.plot(loss.numpy(), label=f"{multi}x")
            plt.fill_between(list(range(loss.size(0))), (loss - loss_std).numpy(), (loss + loss_std).numpy(), color=p[0].get_color(), alpha=0.2)

        if counter >= 5:
            plt.xlabel('Epoch')
                
        if counter == 2 or counter == 5:           
            plt.ylabel('Evaluation log loss')
        else:
            ax.set_yticklabels([])
                
        plt.title(f"$p={decay}$")
        plt.axis([0, 100, -8, 0])

        if counter == 2:
            plt.legend()
        
    plt.savefig(f"convergence_m{num_teacher}.pdf")
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('root', type=str, help="root directory")

    args = parser.parse_args()

    data = load_data(args.root)
    plot_max_corr_alpha(stats, teacher_thres=0.2, student_thres=0.6)
