import math
import torch

def register_kfac_hook(model, ema_decay=1, approximation = 'expand'):
    def forward_hook(self, in_data, out_data):
        in_data = in_data[0].clone().detach()
        def backward_hook(out_grads):
            if in_data.ndim == 3:
                M, R, P_in = in_data.shape
                P_out = out_grads.shape[2]
                if approximation == 'expand':
                    in_data = in_data.view(M*R, P_in) / math.sqrt(R)
                    out_grads = out_grads.view(M*R, P_out)
                elif approximation == 'reduce':
                    in_data = in_data.mean(dim=1)
                    out_grads = out_grads.sum(dim=1)
            if isinstance(self, torch.nn.Embedding):
                counts = torch.stack(
                    [torch.bincount(in_data[i].int(), minlength=module.num_embeddings) for i in range(in_data.shape[0])])
                in_data = counts.float().to(module.weight.device)
                out_grads = out_grads.flatten(end_dim=-2)
            A = torch.matmul(in_data.T, in_data) / M
            B = torch.matmul(out_grads.T, out_grads)
            if hasattr(self, 'A'):
                self.A = (1-ema_decay)*self.A + ema_decay*A
                self.B = (1-ema_decay)*self.B + ema_decay*B
            else:
                self.A = A
                self.B = B
        out_data.register_hook(backward_hook)
    for module in model.children():
        module.register_forward_hook(forward_hook)

def cholesky_inv(X, damping=1e-7):
    diag = torch.diagonal(X)
    diag += damping
    u = torch.linalg.cholesky(X)
    diag -= damping
    return torch.cholesky_inverse(u)

def inverse_curvature(model, damping, eps=1e-10):
    for module in model.children():
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
        module.grad = module.B_inv @ module.grad @ module.A_inv