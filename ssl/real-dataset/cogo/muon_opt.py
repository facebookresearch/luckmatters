import torch
from torch import Tensor
from typing import Optional

def zeropower_via_newtonschulz5(G: Tensor, use_bf16=True) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16() if use_bf16 else G
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for a, b, c in (
        (4.01, -9.22, 5.80),
        (3.49, -6.38, 3.23),
        (3.34, -6.21, 3.20),
        (3.64, -7.48, 4.43),
        (3.46, -5.35, 2.85),
    ):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(
    p: Tensor,
    grad: Optional[Tensor],
    momentum: Tensor,
    beta1: float, 
    beta2: float,
    lr: float,
    wd: float,
    nesterov: bool=True,
    update_rms_compensate: bool=True, 
    update_spectral_compensate: bool=False,
    use_bf16: bool=True,
    decay_after_update: bool=False,
) -> Tensor:
    
    if grad is None:
        return
    
    # compensation
    compensation_coeff = (
        (grad.size(-2) / grad.size(-1))**0.5 if update_spectral_compensate else
        max(1, grad.size(-2) / grad.size(-1))**0.5 if update_rms_compensate else
        1
    )
    if wd != 0 and not decay_after_update:
        p.mul_(1-lr*wd)

    momentum.lerp_(grad, 1-beta2)
    if nesterov:
        grad.lerp_(momentum, 1-beta1)
    else:
        grad.copy_(momentum)
    grad = zeropower_via_newtonschulz5(grad, use_bf16)
    grad.mul_(compensation_coeff)
    p.add_(grad.reshape(p.shape), alpha=-lr)

    if wd != 0 and decay_after_update:
        p.mul_(1-lr*wd)

class MuonEnhanced(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/
    """
    def __init__(
            self, 
            params, 
            lr=0.01, 
            weight_decay=0.01,
            beta1=0.95, 
            beta2=0.95, 
            nesterov=True,
            update_rms_compensate=True,
            update_spectral_compensate=False, 
            use_bf16=True,
            decay_after_update=False,
        ):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2, nesterov=nesterov,
                        update_rms_compensate=update_rms_compensate,
                        update_spectral_compensate=update_spectral_compensate,
                        use_bf16=use_bf16, decay_after_update=decay_after_update)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                muon_update(
                    p=p, 
                    grad=p.grad,
                    momentum=state["momentum_buffer"], 
                    beta1=group["beta1"],
                    beta2=group["beta2"],
                    lr=group["lr"],
                    wd=group["weight_decay"],
                    nesterov=group["nesterov"],
                    update_rms_compensate=group["update_rms_compensate"],
                    update_spectral_compensate=group["update_spectral_compensate"],
                    use_bf16=group["use_bf16"],
                    decay_after_update=group["decay_after_update"]
                )
        return loss