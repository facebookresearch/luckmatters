import common_utils
import os
import sys
import glob
import tqdm
import argparse
import matplotlib.pyplot as plt

from analyze_util import *

def v_annotate(expr, upper, fontsize, color='r', margin=0.1):
    expr_val = eval(expr)
    plt.axvline(expr_val, color=color, linestyle='--', linewidth=0.75)
    expr = expr.replace("math.", "\\").replace("*", "")
    plt.annotate(f"${expr}$", (expr_val + margin, upper), fontsize=fontsize)

def check_constraint(F):
    print("Rc")
    print(f"abc = {(F[0,:] * F[1,:] * F[2,:]).sum()}")
    print(f"^abc = {(F[0,:].conj() * F[1,:] * F[2,:]).sum()}")
    print(f"a^bc = {(F[0,:] * F[1,:].conj() * F[2,:]).sum()}")
    print(f"ab^c = {(F[0,:] * F[1,:] * F[2,:].conj()).sum()}")
    print("Rn")
    print(f"a^ac = {(F[0,:] * F[0,:].conj() * F[2,:]).sum()}")
    print(f"b^bc = {(F[1,:] * F[1,:].conj() * F[2,:]).sum()}")
    print("R*")
    print(f"aac = {(F[0,:] * F[0,:] * F[2,:]).sum()}")
    print(f"bbc = {(F[1,:] * F[1,:] * F[2,:]).sum()}")
    print(f"^a^ac = {(F[0,:].conj() * F[0,:].conj() * F[2,:]).sum()}")
    print(f"^b^bc = {(F[1,:].conj() * F[1,:].conj() * F[2,:]).sum()}")

def analyze_component_v2(F):
    q = F.shape[1]
    # Determine the permutation
    min_err = 1e8
    min_perm = None
    all_errs = []
    for perm in itertools.permutations(range(q)):
        tmpF = F[:, perm]

        ratio = tmpF[2, :q//2] / tmpF[2, q//2:]
        err = (ratio - ratio.mean()).norm()

        # ratio should be constant per row 
        all_errs.append(err)
        if err < min_err:
            min_err = err
            min_perm = perm
            
    all_errs = torch.stack(all_errs)
    # print(all_errs)
    # print(min_err, min_perm)
    
    # Then specify the min_perm and do decomposition
    F = F[:, min_perm]
    
    # further decompose 
    ratio_test = F[0, q//2:] / F[0, :q//2]
    
    # if there is any place that requires applying [-1, -1, 1], apply it
    ratio_test_err = (ratio_test - ratio_test[0]).norm()
    idd = ((ratio_test - ratio_test[0]).abs() > 0.2).nonzero().squeeze(1).tolist()
    
    if len(idd) > 0:
        # print(f"Correcting [-1,-1,1] with ratio_test_err = {ratio_test_err}")
        F[:2, idd] *= -1
    
    sec2 = F[:, q//2:]
    sec1 = F[:, :q//2]
        
    ratio = sec2 / sec1
    err = (ratio - ratio.mean(dim=1, keepdim=True)).norm()
    # print("Confirmed err: ", err)
    component1 = ratio.mean(dim=1)
    component1 = torch.stack([torch.ones((3,), dtype=component1.dtype), component1], dim=1)
    
    component2 = sec1 / sec1[:,0][:,None]
    
    return component1, component2, err, ratio


# Check types of component1 and component2
def check_component(A):
    #define some constant
    one = torch.ones(1, dtype=torch.cfloat)
    plus_i = torch.zeros(1, dtype=torch.cfloat)
    plus_i.imag = 1
    
    minus_i = torch.zeros(1, dtype=torch.cfloat)
    minus_i.imag = -1
    
    normalize = True
    
    if normalize:
        A = A / A.prod(dim=0,keepdim=True).pow(1.0/3)
    
    thres = 0.1
    
    if A.shape[1] == 2:
        # order-2 
        # import pdb
        # pdb.set_trace()
        if (A[0,1] * A[1,1] * A[2,1] - one).abs() > thres:
            return "order-2-unnormalized"
        if (A[2,1] - plus_i).abs() < thres or (A[2,1] - minus_i).abs() < thres:
            # [1, xi], [1, \pm i\bar xi], [1, \pm i]
            return "order-2-xi", (A[0,1].item(),)
        if (A[2,1] - one).abs() < thres and ((A[0,1] - plus_i).abs() < thres or (A[0,1] - minus_i).abs() < thres):
            return "order-2-4c", (A[0,1].item(),)
        #if (A[2,1] + one).abs() < thres and ((A[0,1] - one).abs() < thres or (A[0,1] + one).abs() < thres):
        #    return "order-2-one", (1,)
        if (A[0,1] + A[1,1]).abs() < thres:
            return "order-2-nu", (A[0,1].item(),)
        return "order-2-unknown", None
    
    elif A.shape[1] == 3:
        omega3 = torch.ones(1, dtype=torch.cfloat)
        omega3[0].real = math.cos(2*math.pi/3)
        omega3[0].imag = math.sin(2*math.pi/3)
        
        if (A[0,1] * A[1,1] * A[2,1] - one).abs() > thres:
            return "order-3-unnormalized-1", None
        if (A[0,2] * A[1,2] * A[2,2] - one).abs() > thres:
            return "order-3-unnormalized-2", None
        
        # remove -1, -1, 1 factor for each col
        for i in (1,2):
            if A[0,i].real > 0:
                A[:2,i] *= -1
                
        if A[2,1].imag < 0:
            A = A.conj()
        # check
        if (A[:,1] - omega3).norm() < thres * math.sqrt(3) and (A[:,2] - omega3.conj()).norm() < thres * math.sqrt(3):
            return "order-3-syn", (1, 1)
        
        if (A[2,1] - omega3).norm() < thres and (A[2,2] - omega3.conj()).norm() < thres:
            return "order-3-syn-ab", (A[0,1] / omega3, A[0,2] / omega3)
        
        return "order-3-unknown", None
    
    else:
        return "error", None

import itertools
from collections import Counter

# categorize one solution
# for each node, get their best freq
def analyze_solution(A, B, C, stats4, stats6, errs4, errs6):
    d = A.shape[0]
    q = A.shape[1]

    best_freqs_max, best_freqs = A[:(d-1)//2+1].abs().max(dim=0)

    freq_filter = best_freqs_max > 0.05
    best_freqs = best_freqs[freq_filter].tolist()
    best_freqs_indices = freq_filter.nonzero().squeeze(dim=1).tolist()
    
    for k in range(1, (d-1)//2 + 1):
        indices = [ idx for idx,kk in zip(best_freqs_indices, best_freqs) if k == kk ]

        Ak = A[k,indices]
        Bk = B[k,indices]
        Ck = C[k,indices]
        q = len(indices)
        
        if q not in (6, 4):
            continue

        F = torch.stack([Ak, Bk, Ck], dim=0)

        component1, component2, err, ratio = analyze_component_v2(F)

        if q == 4:
            stats = stats4
            errs = errs4
        elif q == 6:
            stats = stats6
            errs = errs6
            
        res1, param1 = check_component(component1)
        res2, param2 = check_component(component2)
        if res1 > res2:
            res1, res2 = res2, res1
            component1, component2 = component2, component1
            param1, param2 = param2, param1
            
        if res1 == "order-2-unknown" or res2 == "order-3-unknown":
            print(f"===== Warning!! res1 = {res1}, res2 = {res2}, err = {err}")
            # import pdb
            # pdb.set_trace()
        
        stats["cnt"].append(1)
        if err > math.sqrt(F.shape[0] * F.shape[1]) * 0.1:
            # not factorable
            stats["non-factorable"].append(1)
        else:
            stats[(res1, res2)].append((param1, param2))
            errs.append(err.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # The root directory after you run the parameter sweep
    # e.g.,  

    parser.add_argument("root", type=str)
    args = parser.parse_args()
    root = args.root
    all_cfgs = load_all(root)

    q = "512"
    wd = "5e-05"

    from collections import defaultdict

    all_stats = defaultdict(lambda: dict(stats4 = defaultdict(list), stats6 = defaultdict(list), errs4 = [], errs6 = []))

    for (M, weight_decay, seed), tbl in all_cfgs[all_cfgs["hidden_size"] == q].groupby(["M", "weight_decay", "seed"]):
        # compute their statistics
        if weight_decay != wd:
            continue
            
        entry = all_stats[M, seed]
            
        stats4 = entry["stats4"]
        stats6 = entry["stats6"]
        errs4 = entry["errs4"]
        errs6 = entry["errs6"]
        
        for folder in tbl["folder"]:
            print(folder)
            # try:
            # Load them
            # data = torch.load(os.path.join(folder, "data.pth"), map_location="cpu")
            As, Bs, Cs, ts, _ = load_model_traj(folder, indices=[9900])

            analyze_solution(As[:,:,-1], Bs[:,:,-1], Cs[:,:,-1], stats4, stats6, errs4, errs6)          

    for d in [23, 71, 127]:
        result = defaultdict(list)
        n = (d - 1) // 2
        
        for i in range(1,6):
            s = all_stats[(str(d), str(i))]
            result["err_4"].append(torch.Tensor(s["errs4"]).mean())
            result["err_6"].append(torch.Tensor(s["errs6"]).mean())
            
            s4 = s["stats4"]
            s6 = s["stats6"]
            
            n4 = len(s4["cnt"])
            n6 = len(s6["cnt"])
            
            result["1nonfact_4"].append(len(s4["non-factorable"]) / n4)
            result["1nonfact_6"].append(len(s6["non-factorable"]) / n6)
            
            nfact = n4 + n6 - len(s4["non-factorable"]) - len(s6["non-factorable"])
            
            n4_4cxi = len(s4[('order-2-4c', 'order-2-xi')])
            n6_nusyn = len(s6[('order-2-nu', 'order-3-syn')])
            n6_4csynab = len(s6[('order-2-4c', 'order-3-syn-ab')]) + len(s6[('order-2-4c', 'order-3-syn')])
            
            result["sol_4_4cxi"].append(n4_4cxi / nfact)
            result["sol_6_nusyn"].append(n6_nusyn / nfact)
            result["sol_6_4csynab"].append(n6_4csynab / nfact)
            result["sol_others"].append((nfact - n4_4cxi - n6_nusyn - n6_4csynab) / nfact)
            
            result["0_non_46"].append(n - n4 - n6)
             
        # print table 2 in the paper. 
        print(f"d = {d}:")
        rows = ""
        row_names = ""
        
        for k in sorted(result.keys()):
            result[k] = torch.Tensor(result[k])
            mean = result[k].mean()
            std = result[k].std() / math.sqrt(len(result[k]))
            
            # if k.startswith("err"):
            if not k.startswith("0_"):
                row = f"${mean*100:.2f}${{\\tiny$\\pm {std*100:.2f}$}}"
            else:
                row = f"${mean:.2f}${{\\tiny$\\pm {std:.2f}$}}" 
            # else:
            #    row = f"${mean:.2f} \pm {std:.2f}$"
                
            rows += "& " + row 
            row_names += "& " + k
                
        print(row_names)
        print(rows)
        print()
        print()    

    # draw histogram
    sol_type = ('order-2-4c', 'order-2-xi')

    params = []

    for d in [23, 71, 127]:
        for i in range(1,6):
            s = all_stats[(str(d), str(i))]
            s4 = s["stats4"]

            for sol in s4[sol_type]:
                xi = torch.tensor(sol[1][0])
                params.append(xi.angle())
                
    plt.figure(figsize=(5,3))
        
    plt.hist(params, bins=50)
    plt.axis([-math.pi - 0.1, math.pi + 0.1, None, None])

    label_fontsize = 14
    plt.ylabel("Count", fontsize=label_fontsize)
    plt.xlabel(r"Angle of $\xi$", fontsize=label_fontsize)
    plt.xticks(fontsize=label_fontsize)
    plt.yticks(fontsize=label_fontsize)
    plt.title(r"$\mathbf{z}_{\nu=\mathrm{i}} * \mathbf{z}_\xi$", fontsize=label_fontsize+2)

    upper = 65
    fontsize = 14

    v_annotate("math.pi/4", upper, fontsize)
    v_annotate("-math.pi/4", upper, fontsize)
    v_annotate("3*math.pi/4", upper, fontsize)
    v_annotate("-3*math.pi/4", upper, fontsize)

    plt.tight_layout()
    plt.savefig("angle_of_xi.pdf")

    # draw histogram
    sol_type = ('order-2-nu', 'order-3-syn')
    sol_type2 = ('order-2-4c', 'order-3-syn')

    params = []

    for d in [23, 71, 127]:
        for i in range(1,6):
            s = all_stats[(str(d), str(i))]
            s6 = s["stats6"]

            for sol in s6[sol_type]:
                nu = torch.tensor(sol[0][0])
                params.append(nu.angle())
                
            for sol in s6[sol_type2]:
                nu = torch.tensor(sol[0][0])
                params.append(nu.angle())

                
    plt.figure(figsize=(5,3))
        
    label_fontsize = 14
    plt.hist(params, bins=50)
    plt.axis([-math.pi - 0.1, math.pi + 0.1, None, None])
    # plt.ylabel("Count", fontsize=label_fontsize)
    plt.xlabel(r"Angle of $\nu$", fontsize=label_fontsize)
    plt.xticks(fontsize=label_fontsize)
    plt.yticks(fontsize=label_fontsize)
    plt.title(r"$\mathbf{z}_{\nu} * \mathbf{z}_{\mathrm{syn}}$", fontsize=label_fontsize+2)

    upper = 10
    fontsize = 14

    v_annotate("math.pi/2", 8, fontsize, color='g', margin=-0.2)
    v_annotate("-math.pi/2", 8, fontsize, color='g', margin=-0.4)

    v_annotate("math.pi/3", upper, fontsize, margin=-0.5)
    v_annotate("2*math.pi/3", upper, fontsize, margin=0.1)
    v_annotate("-math.pi/3", upper, fontsize, margin=0)
    v_annotate("-2*math.pi/3", upper, fontsize, margin=-1)

    plt.tight_layout()
    plt.savefig("angle_of_nu.pdf")

    # draw histogram
    sol_type = ('order-2-4c', 'order-3-syn-ab')
    sol_type2 = ('order-2-4c', 'order-3-syn')

    params_alpha = []
    params_beta = []

    for d in [23, 71, 127]:
        for i in range(1,6):
            s = all_stats[(str(d), str(i))]
            s6 = s["stats6"]

            for sol in s6[sol_type]:
                alpha_beta = torch.tensor(sol[1])
                params_alpha.append(alpha_beta[0].angle())
                params_beta.append(alpha_beta[1].angle())
                
            for sol in s6[sol_type2]:
                params_alpha.append(0)
                params_beta.append(0)

                
    plt.figure(figsize=(5,3))
        
    plt.hist(params_alpha, bins=10, color='b', label=r"$\alpha$", alpha = 0.5)
    plt.hist(params_beta, bins=10, color='g', label=r"$\beta$", alpha = 0.5)
    plt.axis([-math.pi - 0.1, math.pi + 0.1, None, None])
    # plt.ylabel("Count", fontsize=label_fontsize)
    plt.xlabel(r"Angle of $\alpha$ and $\beta$", fontsize=label_fontsize)
    plt.legend(fontsize=label_fontsize)
    plt.xticks(fontsize=label_fontsize)
    plt.yticks(fontsize=label_fontsize)
    plt.title(r"$\mathbf{z}_{\nu=\mathrm{i}} * \mathbf{z}_{\mathrm{syn},\alpha\beta}$", fontsize=label_fontsize+2)

    plt.tight_layout()
    plt.savefig("angle_of_alpha_beta.pdf")
