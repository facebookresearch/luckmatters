# nrows = int(math.sqrt(len(keys)))
# ncols = math.ceil(len(keys) / nrows)

# fig, axs = plt.subplots(nrows, ncols, figsize=(20,10))
# axs = axs.flatten()

import pandas as pd
import matplotlib.cm as cm
import torch
import os
import matplotlib.pyplot as plt

filename = "wikitext2_multilayer_lr.pt"

results = torch.load(filename)

df = pd.DataFrame(results)
# df = df[df["seed"] == 3]
# 
df = df[df["lr"] == 0.0001]

cmaps = plt.get_cmap('rainbow', 10)

all_val_loss = []

plt.figure(figsize=(16,4))
plt.rcParams.update({'font.size': 14})

fig_idx = 1

for idx, df2 in df.groupby(["nlayers", "lr"]):
    lr = idx[1]
    nlayers = idx[0]
    
    if not nlayers in [1, 2, 5, 10]:
        continue
    
    entropies = None
    val_loss = None
    
    for _, row in df2.iterrows():
        # print(row)
        # print("=======================")
        keys = row["keys"]
        
        if "this_entropies" in row:
            this_entropies = row["this_entropies"]
            this_val_loss = row["this_val_loss"]
        else:
            attns = row["attns"]
            val_losses = row["val_losses"]
            # nlayers = len(attns[0])

            this_entropies = torch.zeros(len(keys), nlayers)
            this_val_loss = torch.zeros(len(keys))

            for j, (key, attn) in enumerate(zip(keys, attns)):
                for layer in range(nlayers):
                    attn_map = attn[layer]
                    # axs[j].set_title(key)
                    # axs[j].imshow(attn_map.cpu())
                    this_entropies[j, layer] = (-(attn_map * (attn_map + 1e-8).log()).sum(dim=1)).mean().item()
                this_val_loss[j] = val_losses[j]

                # if key % 25000 == 0:    
                #     print(f"{key}: avg entropy = {this_entropies[j, :]}")

                # heavy hitter?
                # _, sorted_indices = attn_map.sum(dim=0).sort(descending=True)
                # print(f"iteration: {iteration}")
                # print([seq[i] for i in sorted_indices])

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
    
    plt.subplot(1, 4, fig_idx)
    x = torch.FloatTensor(keys[:first]) / 1000

    for j, layer in enumerate(range(nlayers)):
        
        if nlayers < 5:
            label = f"layer{layer}"
        else:
            label = None
        plt.plot(x, entropies[:first, layer], label=label, color=cmaps(j/nlayers))
    
    if fig_idx >= 3:
        norm_iters = plt.Normalize(0, nlayers - 1)
        sm = cm.ScalarMappable(cmap=cmaps, norm=norm_iters)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('', rotation=270, labelpad=15)
    else:
        # plt.legend(loc="upper right")
        plt.legend()
    plt.xlabel("Minibatch (k)")
    if filename.startswith("wikitext2"):
        plt.axis([-2, x.max(), 1, 2.7])
    else:
        plt.axis([-5, x.max(), 0.8, 2.7])
    
    if fig_idx == 1:
        plt.ylabel("Attention Entropy")
    else:
        plt.yticks([])

#     plt.twinx()
#     plt.plot(x, val_loss[:first], "--", color="k", linewidth=0.5, label="val_loss")
#     plt.legend(loc='lower right')
#     plt.xlabel("Minibatch (k)")
#     if filename.startswith("wikitext2"):
#         plt.axis([-2, 60, 5, 9])
#     else:
#         plt.axis([-10, x.max(), 4.5, 9])
        
#     if fig_idx == 4:
#         plt.ylabel("Val loss")
#     else:
#         plt.yticks([])
    # plt.title(f"Layers: {nlayers}, lr: {lr}, val_loss: {val_loss.min().item():.2f}")
    val_str = f"{val_loss.min().item():.3f}"
    if nlayers == 10:
        tt = f"Layers: {nlayers}, val_loss: " + r"$\mathbf{" + val_str + "}$"
    else:
        tt = f"Layers: {nlayers}, val_loss: " + r"$" + val_str + "$"
    
    plt.title(tt)

    # print(f"Validation loss: {val_loss.min().item()}")

    all_val_loss.append(dict(nlayers=nlayers, lr=lr, val_loss=val_loss.min().item()))
    
    fig_idx += 1
    
plt.tight_layout()

name, _ = os.path.splitext(filename)
plt.savefig(name + ".pdf")
# plt.show()
        
        
df_val_loss = pd.DataFrame(all_val_loss)
print(df_val_loss)
