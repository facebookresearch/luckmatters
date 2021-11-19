import torch
import torch.nn.functional as F

class SpecializedL2Regularizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        assert len(input.size()) == 2
        l2_norms = input.pow(2).sum(dim=1, keepdim=True).sqrt().add(1e-8)
        ctx.l2_norms = l2_norms
        return input / l2_norms

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output / ctx.l2_norms
        return grad_input

a = torch.randn(1, 2, requires_grad=True)
print(a)
z = a.norm(dim=1).detach()
print(z)

reg = SpecializedL2Regularizer.apply
# reg = lambda x: F.normalize(x, dim=1) 
b = reg(a)
b.pow(2).sum().backward()

print(a.grad)

print(2 * a / z / z)
