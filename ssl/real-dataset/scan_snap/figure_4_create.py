import sys
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

root = sys.argv[1]

facts = torch.load(os.path.join(root, "facts.pth"), map_location="cpu")
facts = facts["facts"]

K1s = []
K2s = []
# iters = list(range(0, 1000, 200)) + list(range(1000,5000,1000))
# iters = list(range(0,100,10))
iters = list(range(0,50,2))

for t in iters:
    model = torch.load(os.path.join(root, f"iter-{t}.pth"), map_location="cpu")

    K1 = model["model"]["K1.weight"]
    K2 = model["model"]["K2.weight"]
    
    K1s.append(K1)
    K2s.append(K2)

def get_c(K2, facts):
    rows = []
    for i, f in enumerate(facts):
        lt = f["last_token"]
        c = K2[lt,:].exp()
        # compute c
        sel = torch.BoolTensor(c.size(0)).fill_(0)
        sel[f["tokens"]] = 1

        c[sel] *= f["probs"]
        c[~sel] = 0
        c = c / c.sum()
        
        rows.append((c, i))
        
    return rows

all_rows = []
ts = [0, 1, 2, 3, 4, 5, -1]
for t in ts:
    all_rows.append(get_c(K2s[t], facts))

plt.figure(figsize=(4,4.3))

row_img = torch.stack([rows[0][0] for rows in all_rows], dim=1)
row_img = row_img[:20,:]

ax = plt.gca()
plt.imshow(row_img.t())
plt.axvline(9.5, color='w', linestyle='--', linewidth=1)
plt.axvline(19.5, color='w', linestyle='--', linewidth=1)

# ax.set_yticks([t*2 for t in ts])
plt.yticks(list(range(7)), [f"Iter {t*2}" if t >= 0 else "Final" for t in ts])
# ax.axis([0, row_img.size(0) - 0.5, 0, row_img.size(1) - 0.5])

plt.xlabel('Token #')
# plt.ylabel("Iteration")

plt.title("       Common token  Distinct token (seq1)")

# plt.show()
plt.savefig("plot.pdf")
