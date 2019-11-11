import pickle
import torch
import os
import sys
import glob
import yaml

def find_params(data, cond):
    for d in data:
        found = True
        for k, v in cond.items():
            if d["args"][k] != v:
                found = False
        if found:
            return d
    return None

def find_all_params(data, cond):
    all_d = []
    for d in data:
        found = True
        for k, v in cond.items():
            if d["args"][k] != v:
                found = False
        if found:
            all_d.append(d)
    return all_d

def load_stats(folder):
    print(f"Load stats from {folder}")
    filename = os.path.join(folder, "stats.pickle")
    if os.path.exists(filename):
        config_filename = os.path.join(folder, "config.yaml")
        if not os.path.exists(config_filename):
           config_filename = (os.path.join(folder, ".hydra/config.yaml"))
        else:
            return None
        print(f"Config file: {config_filename}")
        args = yaml.load(open(config_filename, "r"))
        stats = torch.load(filename)
        return dict(args=args,stats=stats)
    else:
        print(f"The {filename} doesn't exist")
        return None


def load_data(root):
    data = []
    total = 0
    folders = sorted(glob.glob(os.path.join(root, "*")))
    last_prefix = None

    for folder in folders:
        path, folder_name = os.path.split(folder)
        prefix, job_id = folder_name.split("_")
        if prefix == last_prefix:
            continue

        stats = load_stats(folder)
        if stats is not None:
            print(f"{len(data)}: {folder}")
            data.append(stats)
            last_prefix = prefix

    return data
