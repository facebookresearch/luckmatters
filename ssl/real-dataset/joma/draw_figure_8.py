# nrows = int(math.sqrt(len(keys)))
# ncols = math.ceil(len(keys) / nrows)

# fig, axs = plt.subplots(nrows, ncols, figsize=(20,10))
# axs = axs.flatten()

import pandas as pd
import matplotlib.cm as cm
import torch
import os
import matplotlib.pyplot as plt

filename = "wikitext2"

results = torch.load("wikitext2_multilayer_lr_small.pt")
results += torch.load("wikitext2_multilayer_lr.pt")

fixed_num_layer = 5

df = pd.DataFrame(results)
df = df[df["nlayers"] == fixed_num_layer]
# 
cmaps = plt.get_cmap('rainbow', 10)

all_val_loss = []

plt.figure(figsize=(12,3))
plt.rcParams.update({'font.size': 10})

fig_idx = 1

for idx, df2 in df.groupby(["nlayers", "lr"]):
    lr = idx[1]
    nlayers = idx[0]
    
#     if not nlayers in [1, 2, 5, 10]:
#         continue
    
    entropies = None
    val_loss = None
    
    for _, row in df2.iterrows():
        # print(row)
        # print("=======================")
        keys = row["keys"]
        
        this_entropies = row["this_entropies"]
        this_val_loss = row["this_val_loss"]

        if entropies is None:
            entropies = this_entropies
            val_loss = this_val_loss
        else:
            entropies += this_entropies
            val_loss += this_val_loss
        
    entropies /= df2.shape[0]
    val_loss /= df2.shape[0]

    # plt.savefig("layer"+ str(layer) +".pdf", format="pdf", bbox_inches="tight")
    first = 100
    
    plt.subplot(1, 7, fig_idx)
    x = torch.FloatTensor(keys[:first]) / 1000

    for j, layer in enumerate(range(nlayers)):
        label = f"layer{layer}"
        plt.plot(x, entropies[:first, layer], label=label, color=cmaps(j/nlayers))
    
    plt.xlabel("Minibatch (k)")
    if filename.startswith("wikitext2"):
        plt.axis([-2, x.max(), 0, 2.7])
    else:
        plt.axis([-5, x.max(), 0.8, 2.7])
    
    if fig_idx == 1:
        plt.ylabel("Attention Entropy")
    else:
        plt.yticks([])

    if fig_idx == 1:
        plt.legend() 
        
#     if fixed_num_layer < 5:
#         plt.legend()
#     else:
#         if fig_idx == 3:
#             norm_iters = plt.Normalize(0, fixed_num_layer - 1)
#             sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
#             sm.set_array([])
#             cbar = plt.colorbar(sm, ax=plt.gca())
#             cbar.set_label('', rotation=270, labelpad=15)

    val_str = f"{val_loss.min().item():.2f}"
    if lr == 0.0001:
        tt = f"lr: {lr}, val: " + r"$\mathbf{" + val_str + "}$"
    else:
        tt = f"lr: {lr}, val: " + r"$" + val_str + "$"

    plt.title(tt)

    # print(f"Validation loss: {val_loss.min().item()}")

    all_val_loss.append(dict(nlayers=nlayers, lr=lr, val_loss=val_loss.min().item()))
    
    fig_idx += 1
    
plt.tight_layout()

name, _ = os.path.splitext(filename)
plt.savefig(name + f"_lr_nlayer{fixed_num_layer}.pdf")
# plt.show()
        
        
df_val_loss = pd.DataFrame(all_val_loss)
print(df_val_loss)
