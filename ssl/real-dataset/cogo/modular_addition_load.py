import common_utils
import math
import os
import copy
import time
import warnings
import torch
import glob

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

def load_model_traj(root, map_location=None):
    fourierBases = None

    As = []
    Bs = []
    Cs = []
    ts = []

    filenames = glob.glob(os.path.join(root, "model*.pt"))

    # Load all models and their ABC
    for f_name in sorted(filenames):
        # print(f_name)
        data = torch.load(f_name, map_location=map_location)
        model = data["model"]
        if fourierBases is None:
            d = model["layera.weight"].shape[1]
            fourierBases = construct_bases(d)
            fourierBases = fourierBases.to(model["layera.weight"].device)

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
    print(folder)
    cfg = common_utils.MultiRunUtil.load_cfg(folder)
    cfg = { entry.split("=")[0] : entry.split("=")[1] for entry in cfg }
    
    As, Bs, Cs, ts, results = load_model_traj(folder)
    # Compute 
    try:
        mps = compute_mps(As, Bs, Cs)
    except:
        # switch to CPU and retry
        mps = compute_mps(As.cpu(), Bs.cpu(), Cs.cpu())
    
    entry = dict(As=As,Bs=Bs,Cs=Cs,results=results)
    entry.update(mps)
    entry["cfg"] = cfg
    
    return entry

