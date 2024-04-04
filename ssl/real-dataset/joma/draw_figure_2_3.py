import matplotlib.pyplot as plt
import torch

# simulate the dynamics
def norm_corr(v1, v2):
    v1_s = v1 - v1.mean()
    v1_s = v1_s / (v1_s.norm() + 1e-8)
    v2_s = v2 - v2.mean()
    v2_s = v2_s / (v2_s.norm() + 1e-8)
    return (v1_s @ v2_s).item()


# generate input data
Mdistinct = 10
Mcommon = 10
Mc = Mdistinct + Mcommon
Mq = 1
K = 20

use_gaussian_prob = True

X = torch.rand(Mc, K) / 2
idx = torch.arange(Mdistinct)
# simple initialization
if use_gaussian_prob:
    for i in range(K):
        X[:Mdistinct,i] = (-(idx - i / K * Mdistinct).pow(2) / 30).exp()
X = X / X.sum(dim=0, keepdim=True)

# Create embedding
d = 30
U = torch.zeros(d, Mc + Mq)
U[:Mc+Mq,:Mc+Mq] = torch.eye(Mc + Mq)
T = 0.5

use_layer_norm = False

if use_layer_norm:
    xbar = X.pow(2).mean(dim=1)
    xbar = xbar / xbar.sum()
    # xbar.sum() == 1
else:
    xbar = X.mean(dim=1)
    # xbar.sum() == 1

# Train model
W = torch.randn(d, K, requires_grad=True)
z = torch.randn(Mc, requires_grad=True)
with torch.no_grad():
    W[:] /= 100
    z[:] /= 100
    # compute constant
    W_tilde = U.t() @ W
    z_est = (W_tilde[:Mc,:].pow(2).sum(dim=1) - xbar * W.pow(2).sum()) / 2 / T
    C = z - z_est

# fixed backpropagate gradient
G = torch.eye(K, K) - torch.ones(K,K) / K
ce_loss = torch.nn.CrossEntropyLoss()
labels = torch.arange(K)

niter = 2000

corrs = torch.zeros(niter)
corrs1 = torch.zeros(niter)
corrs2 = torch.zeros(niter)
Ws = []

optimizer = torch.optim.SGD([W, z], lr=0.1)
for t in range(niter):
    optimizer.zero_grad()
    
    Ws.append(W.detach().clone())
    
    B = (z/T).exp()[:,None] * X
    # [Mc, K]
    B = B / B.sum(dim=0, keepdim=True)

    # [d, K]
    # "+ U[:,Mc]" is to add residual connection.
    F = U[:,:Mc] @ B + U[:,Mc][:,None]
    
    # make F unit norm
    if use_layer_norm:
        F = F / F.norm(dim=0, keepdim=True)
    
    # [#samples, #output_dim]
    output = F.t() @ W 
    
    # compute invariance
    with torch.no_grad():
        W_tilde = U.t() @ W
        z_est1 = W_tilde[:Mc,:].pow(2).sum(dim=1) / 2 / T
        z_est2 = - xbar * W.pow(2).sum() / 2 / T 
        z_est = z_est1 + z_est2 + C
        # Check whether the two are aligned.
        # print((z - z_est).norm() / z.norm())
        corrs[t] = norm_corr(z, z_est)
        corrs1[t] = norm_corr(z, z_est1)
        corrs2[t] = norm_corr(z, z_est2)

    # Apply loss function
    # loss = -(G * output).sum() / K
    loss = ce_loss(output, labels)
    loss.backward()
    optimizer.step()

# Draw figure 2

plt.figure(figsize=[6,3])

plt.subplot(1, 2, 1)
plt.imshow(X)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.axis('off')
plt.title("X")
plt.xlabel(r"Class label $y$")
plt.ylabel(r"Contextual token index $l$")

plt.subplot(1, 2, 2)
plt.plot(corrs, 'r', linewidth=2, label=r"$\mathrm{NC}(\hat z_m(t), z_m(t))$")
plt.plot(corrs1, 'g', linestyle="--", label=r"$\mathrm{NC}(\hat z_{m1}(t), z_m(t))$")
plt.plot(corrs2, 'b', linestyle="--", label=r"$\mathrm{NC}(\hat z_{m2}(t), z_m(t))$")
plt.axhline(1.0, color='k', linestyle="--", linewidth=0.5)
# plt.gca().yaxis.tick_right()
plt.ylabel("Normalized Correlation")
plt.xlabel("Number of Batches")
plt.legend()
# plt.show()

plt.tight_layout()
suffix = "gaussian" if use_gaussian_prob else "random"
plt.savefig(f"linear_case_{suffix}.pdf")

# Draw figure 3

import matplotlib.patches as patches

colors = ['b', 'orange', 'g', 'r']

plt.figure(figsize=[10,3])

all_Ws = torch.stack(Ws, dim=0)
all_Vs = torch.bmm(U.t().repeat(all_Ws.size(0), 1, 1), all_Ws)

plt.subplot(1, 3, 1)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.xlabel(r"$K$")
plt.ylabel(r"Number of contextual tokens ($M_c$)")
plt.title(r"$V(t)$ initialization")
fig_w = plt.imshow(U.t() @ Ws[0].detach())
cbar = plt.colorbar(fig_w)

plt.subplot(1, 3, 2)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.xlabel(r"$K$")
# plt.ylabel(r"Number of contextual tokens ($M_c$)")
plt.ylabel(r"$M_c$")
plt.title(r"$V(t)$ after convergence")
fig_w = plt.imshow(U.t() @ W.detach())
cbar = plt.colorbar(fig_w)
for i, c in enumerate(colors):
    rect = patches.Rectangle((0-0.5+0.1, i-0.5+0.1), 0.8, 0.8, linewidth=2, edgecolor=c, facecolor='none')
    plt.gca().add_patch(rect)


plt.subplot(1, 3, 3)
for i, c in enumerate(colors):
    plt.plot(all_Vs[:, i, 0], label=f"component{i}", color=c)
# plt.plot(all_Vs[:, 4, 0], label="component4")
plt.axhline(0.0, color='k', linestyle="--", linewidth=0.5)
plt.xlabel("Number of MiniBatches")
plt.ylabel(r"$\mathbf{v}_0(t)$")
plt.gca().yaxis.tick_right()
plt.legend()

# plt.show()
plt.tight_layout()
plt.savefig("component_growth.pdf")
