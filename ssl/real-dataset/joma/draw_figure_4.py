import torch
import matplotlib.pyplot as plt

# two points with self attention dynamics
# wstar = torch.FloatTensor([0.5, 0.5, 2, 2, 4, 4, 5, 5])
torch.manual_seed(0)

wstar, _ = (torch.rand(10)*3 + 1).sort()
# wstar, _ = (torch.randn(10) * 4).sort()
d = wstar.shape[0]

w = torch.ones(d) * 0.05
print("Wstar")
print(wstar)

print("Iters")
niter = 5000
eta = 0.02
T = 3

grad_sqr_hist = torch.zeros(d)

entropies = torch.zeros(niter)

cmap = plt.get_cmap('rainbow', 10)
norm_iters = plt.Normalize(0, niter)

plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
for i in range(niter):
    if i % 100 == 0:
        # print(f"iter{i}: {w}")
        plt.plot(w, label=f"iter{i}", color=cmap(norm_iters(i)))
        # compute entropy
        
    # w_normalized = w.pow(2).exp() / w.pow(2).exp().sum()
    # entropies[i] = -(w_normalized * (w_normalized.log() + 1e-8)).sum()
    
    sa = (w.pow(2) / T).exp()
    sa = sa / sa.sum()
    entropies[i] = -(sa * (sa.log() + 1e-8)).sum()
    
    dw = (wstar - w) * sa
    
    w = w + eta * dw # / (grad_sqr_hist + 5).sqrt()
    
    grad_sqr_hist = grad_sqr_hist + dw.pow(2) 

plt.plot(wstar, linestyle="--", color="k")

import matplotlib.cm as cm

sm = cm.ScalarMappable(cmap=cmap, norm=norm_iters)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('', rotation=270, labelpad=15)

plt.xlabel("Sorted index of $\mathbf{v}$ components")
plt.ylabel(r"$\mathbf{v}$")
plt.title(r"Colored line: dynamics of $\mathbf{v}(t)$. Dashed line: target $\mathbf{\mu}$")


plt.subplot(1, 2, 2)
plt.plot(entropies)
plt.xlabel("#iteration")
plt.ylabel(r"entropy($\mathbf{v}(t)$)")
plt.title("Entropy changes over time")

plt.tight_layout()
plt.savefig("entropy.pdf")
