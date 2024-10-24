import torch
import math
import os
import sys
import glob
import common_utils

import multiprocessing as mp
import tqdm

import pandas as pd

# Load a series of models and check their monomial potentials
def construct_bases(d):
    # compute fourier transform in

    v = torch.ones(d, dtype=torch.cfloat)
    # unit 
    for i in range(d):
        v[i].real = math.cos(2 * math.pi * i / d)
        v[i].imag = math.sin(2 * math.pi * i / d)

    fourierBases = torch.zeros(d, d, dtype=torch.cfloat, requires_grad=False)
    for i in range(d):
        fourierBases[:,i] = v ** i
        
    return fourierBases

def load_model_traj(root, indices=None):
    fourierBases = None

    As = []
    Bs = []
    Cs = []
    ts = []
    
    if indices is None:
        filenames = glob.glob(os.path.join(root, "model*.pt"))
    else:
        filenames = [ glob.glob(os.path.join(root, f"model{idx:05}*.pt"))[0] for idx in indices ]

    # Load all models and their ABC
    for f_name in sorted(filenames):
        # print(f_name)
        data = torch.load(f_name, map_location="cpu")
        model = data["model"]
        if fourierBases is None:
            d = model["layera.weight"].shape[1]
            fourierBases = construct_bases(d)

        A = model["layera.weight"].cfloat() @ fourierBases.conj() / d
        B = model["layerb.weight"].cfloat() @ fourierBases.conj() / d
        C = model["layerc.weight"].cfloat().t() @ fourierBases / d

        As.append(A.t())
        Bs.append(B.t())
        Cs.append(C.t())
        ts.append(data["results"][-1]["epoch"])

        # Compute the loss. 

    As = torch.stack(As, dim=2)
    Bs = torch.stack(Bs, dim=2)
    Cs = torch.stack(Cs, dim=2)

    final_results = data["results"]
    
    return As, Bs, Cs, ts, final_results

def process_one(folder):    
    try:
        cfg = common_utils.MultiRunUtil.load_cfg(folder)
        cfg = { entry.split("=")[0] : entry.split("=")[1] for entry in cfg }
        cfg["folder"] = folder
        return cfg
    except:
        return None

def load_all(root):
    folders = [ folder for folder in glob.glob(os.path.join(root, "*")) if os.path.isdir(folder) ]
        
    with mp.Pool(32) as pool:
        all_cfgs = list(filter(lambda x: x is not None, tqdm.tqdm(pool.imap(process_one, folders), total=len(folders))))

    all_cfgs = pd.DataFrame(all_cfgs)
    return all_cfgs

