import math
import torch

def register_kfac_hook(model, ema_decay=1, approximation = 'expand'):
    def forward_hook(self, in_data, out_data):
        self.in_data = in_data[0].clone().detach()
    def backward_hook(self, in_grads, out_grads):
        in_data = self.in_data
        out_grad = out_grads[0]
        if in_data.ndim == 3:
            M, R, P_in = in_data.shape
            P_out = out_grad.shape[2]
            if approximation == 'expand':
                in_data = in_data.view(M*R, P_in) / math.sqrt(R)
                out_grad = out_grad.view(M*R, P_out)
            elif approximation == 'reduce':
                in_data = in_data.mean(dim=1)
                out_grad = out_grad.sum(dim=1)
        if isinstance(self, torch.nn.Embedding):
            counts = torch.stack(
                [torch.bincount(in_data[i].int(), minlength=module.num_embeddings) for i in range(in_data.shape[0])])
            in_data = counts.float().to(module.weight.device)
            out_grad = out_grad.flatten(end_dim=-2)
        A = torch.matmul(in_data.T, in_data) / M
        B = torch.matmul(out_grad.T, out_grad)
        if hasattr(self, 'A'):
            self.A = (1-ema_decay)*self.A + ema_decay*A
            self.B = (1-ema_decay)*self.B + ema_decay*B
        else:
            self.A = A
            self.B = B
    for module in model.children():
        if not is_supported(module):
            continue
        module.register_forward_hook(forward_hook)
        module.register_backward_hook(backward_hook)

def is_supported(module):
    if len(list(module.children())) > 0:
        return False
    if all(not p.requires_grad for p in module.parameters()):
        return False
    return True

def cholesky_inv(X, damping=1e-7):
    diag = torch.diagonal(X)
    diag += damping
    u = torch.linalg.cholesky(X)
    diag -= damping
    return torch.cholesky_inverse(u)

def inverse_curvature(model, damping, regmean_reg=1, eps=1e-10):
    for module in model.children():
        if not is_supported(module):
            continue
        if regmean_reg != 1:
            module.A = regmean_reg * module.A + (1-regmean_reg) * torch.diag(torch.diag(module.A))
            module.B = regmean_reg * module.B + (1-regmean_reg) * torch.diag(torch.diag(module.B))
            damping_A ,damping_B= eps, eps
        A_eig_mean = (module.A.trace()) / module.A.shape[-1]
        B_eig_mean = (module.B.trace()) / module.B.shape[-1]
        pi = torch.sqrt(A_eig_mean / B_eig_mean)
        if pi != 0 and pi != float('inf'):
            r = damping**0.5
            damping_A = max(r * pi, eps)
            damping_B = max(r / pi, eps)
        module.A_inv = cholesky_inv(module.A, damping_A)
        module.B_inv = cholesky_inv(module.B, damping_B)

def precondition_grad(model):
    for module in model.children():
        if not is_supported(module):
            continue
        module.grad = module.B_inv @ module.grad @ module.A_inv