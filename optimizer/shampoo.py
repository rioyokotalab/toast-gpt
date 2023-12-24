# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#                 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytorch implementation of Shampoo."""

from __future__ import print_function
import enum
import itertools
import math

from dataclasses import dataclass
import numpy as np
import torch
import torch.optim as optim


# Grafting is a technique to fix the layerwise scale of Shampoo optimizer.
# https://arxiv.org/pdf/2002.11803.pdf studies this in detail. This
# allows us to plugin the Shampoo optimizer into settings where SGD/AdaGrad
# is already well tuned. Grafting onto Shampoo means take the Shampoo direction,
# but use the step magnitude from the grafted optimizer such as Adagrad or SGD.
class LayerwiseGrafting(enum.IntEnum):
        NONE = 0
        SGD = 1
        ADAGRAD = 2


@dataclass
class ShampooHyperParams:
        """Shampoo hyper parameters."""
        beta2: float = 1.0
        diagonal_eps: float = 1e-6
        matrix_eps: float = 1e-12
        weight_decay: float = 0.0
        inverse_exponent_override: int = 0        # fixed exponent for preconditioner, if >0
        start_preconditioning_step: int = 1
        # Performance tuning params for controlling memory and compute requirements.
        # How often to compute preconditioner.
        preconditioning_compute_steps: int = 1
        # How often to compute statistics.
        statistics_compute_steps: int = 1
        # Block size for large layers (if > 0).
        # Block size = 1 ==> Adagrad (Don't do this, extremely inefficient!)
        # Block size should be as large as feasible under memory/time constraints.
        block_size: int = 128
        # Automatic shape interpretation (for eg: [4, 3, 1024, 512] would result in
        # 12 x [1024, 512] L and R statistics. Disabled by default which results in
        # Shampoo constructing statistics [4, 4], [3, 3], [1024, 1024], [512, 512].
        best_effort_shape_interpretation: bool = True
        # Type of grafting (SGD or AdaGrad).
        # https://arxiv.org/pdf/2002.11803.pdf
        graft_type: int = None
        # Nesterov momentum
        nesterov: bool = True
        # Clipping
        gradient_value_clip: float = -1
        # Nesterov momentum
        reuse_matrix: bool = False
        # Inverse
        use_inverse: bool = False
        dmp_opt: str = 'mean'
        
        # early curvature warmup
        early_phase_iters: float = 0
        early_statistics_compute_steps: int = 1
        early_preconditioning_compute_steps: int = 1

        # Interval Scheduling
        interval_cosine_thres: float = -1
        interval_scheduling_factor: float = 1
        total_iters: int = 1

class Graft:
        """Base class to perform grafting onto Shampoo. This class does no grafting.
        """

        def __init__(self, hps, unused_var):
                self.hps = hps

        def add_statistics(self, grad):
                pass

        def precondition_gradient(self, grad):
                return grad

        def update_momentum(self, update, unused_beta1):
                return update


class SGDGraft(Graft):
        """Graft using SGD+momentum.

        momentum maintains an exponentially weighted moving average of gradients.
        """

        def __init__(self, hps, var):
                super(SGDGraft, self).__init__(hps, var)
                self.momentum = torch.zeros_like(var.data, device=var.get_device())

        def update_momentum(self, update, beta1):
                self.momentum.mul_(beta1).add_(update)
                return self.momentum


class AdagradGraft(SGDGraft):
        """Graft using Adagrad.

        Essentially an implementation of Adagrad with momentum.
        """

        def __init__(self, hps, var):
                super(AdagradGraft, self).__init__(hps, var)
                self.statistics = torch.zeros_like(var.data, device=var.get_device())

        def add_statistics(self, grad):
                self.statistics.add_(grad * grad)

        def precondition_gradient(self, grad):
                return grad / (torch.sqrt(self.statistics) + self.hps.diagonal_eps)


class BlockPartitioner:
        """Partitions a tensor into smaller tensors for preconditioning.

                For example, if a variable has shape (4096, 512), we might split the
                4096 into 4 blocks, so we effectively have 4 variables of size
                (1024, 512) each.
        """

        def __init__(self, var, hps):
                self._shape = var.shape
                self._splits = []
                self._split_sizes = []
                split_sizes = []
                # We split var into smaller blocks. Here we store the metadata to make
                # that split.
                for i, d in enumerate(var.shape):
                        if hps.block_size > 0 and d > hps.block_size:
                                # d-1, otherwise split appends a 0-size array.
                                nsplit = (d-1) // hps.block_size
                                indices = (np.arange(nsplit, dtype=np.int32) + 1) * hps.block_size
                                sizes = np.ones(nsplit + 1, dtype=np.int32) * hps.block_size
                                sizes[-1] = d - indices[-1]
                                self._splits.append((i, indices))
                                self._split_sizes.append((i, sizes))
                                split_sizes.append(sizes)
                        else:
                                split_sizes.append(np.array([d], dtype=np.int32))
                self._num_splits = len(split_sizes)
                self._preconditioner_shapes = []
                for t in itertools.product(*split_sizes):
                        self._preconditioner_shapes.extend([[d, d] for d in t])

        def shapes_for_preconditioners(self):
                return self._preconditioner_shapes

        def num_splits(self):
                return self._num_splits

        def partition(self, tensor):
                """Partition tensor into blocks."""

                assert tensor.shape == self._shape
                tensors = [tensor]
                for (i, sizes) in self._split_sizes:
                        tensors_local = []
                        for t in tensors:
                                tensors_local.extend(
                                                torch.split(t, tuple(sizes), dim=i))
                        tensors = tensors_local
                return tensors

        def merge_partitions(self, partitions):
                """Merge partitions back to original shape."""

                for (i, indices) in reversed(self._splits):
                        n = len(indices) + 1
                        partial_merged_tensors = []
                        ind = 0
                        while ind < len(partitions):
                                partial_merged_tensors.append(
                                                torch.cat(partitions[ind:ind + n], axis=i))
                                ind += n
                        partitions = partial_merged_tensors
                assert len(partitions) == 1
                return partitions[0]


def _merge_small_dims(shape_to_merge, max_dim):
        """Merge small dimensions.

        If there are some small dimensions, we collapse them:
        e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
                         [1, 2, 768, 1, 2048] --> [2, 768, 2048]

        Args:
                shape_to_merge: Shape to merge small dimensions.
                max_dim: Maximal dimension of output shape used in merging.

        Returns:
                Merged shape.
        """
        resulting_shape = []
        product = 1
        for d in shape_to_merge:
                if product * d <= max_dim:
                        product *= d
                else:
                        if product > 1:
                                resulting_shape.append(product)
                        product = d
        if product > 1:
                resulting_shape.append(product)
        return resulting_shape


class Preconditioner:
        """Compute statistics/shape from gradients for preconditioning."""

        def __init__(self, var, hps):
                self._hps = hps
                self._original_shape = var.shape
                self._transformed_shape = var.shape
                if hps.best_effort_shape_interpretation:
                        self._transformed_shape = _merge_small_dims(
                                        self._original_shape, hps.block_size)

                reshaped_var = torch.reshape(var, self._transformed_shape)
                self._partitioner = BlockPartitioner(reshaped_var, hps)
                shapes = self._partitioner.shapes_for_preconditioners()
                rank = len(self._transformed_shape)
                device = var.get_device()
                if rank <= 1:
                        self.statistics = []
                        self.preconditioners = []
                        self.max_eigens = []
                        self.meam_eigens = []
                else:
                        eps = self._hps.matrix_eps
                        self.statistics = [eps * torch.eye(s[0], device=device) for s in shapes]
                        self.preconditioners = [torch.eye(s[0], device=device) for s in shapes]
                        self.max_eigens = [None for s in shapes]
                        self.mean_eigens = [None for s in shapes]
                self.max_eigen_dict = {}
                self.mean_eigen_dict = {}

        def add_statistics(self, grad):
                """Compute statistics from gradients and add to the correct state entries.

                Args:
                        grad: Gradient to compute statistics from.
                """
                if not self.statistics: return
                reshaped_grad = torch.reshape(grad, self._transformed_shape)
                partitioned_grads = self._partitioner.partition(reshaped_grad)
                w1 = self._hps.beta2
                w2 = 1.0 if w1 == 1.0 else (1.0 - w1)
                rank = len(self._transformed_shape)
                for j, grad in enumerate(partitioned_grads):
                        for i in range(rank):
                                axes = list(range(i)) + list(range(i + 1, rank))
                                stat = torch.tensordot(grad, grad, [axes, axes])
                                self.statistics[j*rank + i].mul_(w1).add_(stat, alpha=w2)

        def exponent_for_preconditioner(self):
                """Returns exponent to use for inverse-pth root M^{-1/p}."""
                if self._hps.inverse_exponent_override > 0:
                        return self._hps.inverse_exponent_override
                return 2 * len(self._transformed_shape)

        def compute_preconditioners(self):
                """Compute L^{-1/exp} for each stats matrix L."""
                exp = self.exponent_for_preconditioner()
                eps = self._hps.matrix_eps
                for i, stat in enumerate(self.statistics):
                        if self._hps.use_inverse:
                                self.preconditioners[i], self.max_eigens[i] = ComputeInverse(
                                        stat, eps, self._hps.dmp_opt
                                )
                        else:
                                self.preconditioners[i], self.max_eigens[i] = ComputePower(
                                        stat, exp, ridge_epsilon=eps)
                        self.mean_eigens[i] = (torch.trace(stat).item() / stat.shape[-1])

        def preconditioned_grad(self, grad):
                """Precondition the gradient.

                Args:
                        grad: A gradient tensor to precondition.

                Returns:
                        A preconditioned gradient.
                """
                if not self.preconditioners: return grad
                reshaped_grad = torch.reshape(grad, self._transformed_shape)
                partitioned_grads = self._partitioner.partition(reshaped_grad)
                preconditioned_partitioned_grads = []
                num_splits = self._partitioner.num_splits()
                max_eigen_dict = {}
                mean_eigen_dict = {}
                for i, grad in enumerate(partitioned_grads):
                        preconditioners_for_grad = self.preconditioners[i * num_splits:(i + 1) * num_splits]
                        max_eigens_for_grad = self.max_eigens[i * num_splits:(i + 1) * num_splits]
                        mean_eigens_for_grad = self.mean_eigens[i * num_splits:(i + 1) * num_splits]
                        max_eigen_dict[i] = {}
                        mean_eigen_dict[i] = {}
                        rank = len(grad.shape)
                        precond_grad = grad
                        for j in range(rank):
                                preconditioner = preconditioners_for_grad[j]
                                precond_grad = torch.tensordot(
                                                precond_grad, preconditioner, [[0], [0]])
                                if max_eigens_for_grad[j] is not None:
                                        max_eigen_dict[i][j] = max_eigens_for_grad[j]
                                if mean_eigens_for_grad[j] is not None:
                                        mean_eigen_dict[i][j] = mean_eigens_for_grad[j]
                        preconditioned_partitioned_grads.append(precond_grad)
                merged_grad = self._partitioner.merge_partitions(
                                preconditioned_partitioned_grads)
                self.max_eigen_dict = max_eigen_dict
                self.mean_eigen_dict = mean_eigen_dict
                return torch.reshape(merged_grad, self._original_shape)

STEP = 'step'
MOMENTUM = 'momentum'
PRECONDITIONER = 'preconditioner'
GRAFT = 'graft'
PRECONDITIONER_INTERVAL = 'preconditioner_interval'

class Shampoo(optim.Optimizer):
        """The Shampoo optimizer."""

        def __init__(self,
                                params,
                                lr=1.0,
                                momentum=0.9,
                                hyperparams=ShampooHyperParams(),
                                param_names = None):
                defaults = dict(lr=lr, momentum=momentum)
                self.hps = hyperparams
                self.param_names = param_names
                self.cosine_dict = {}
                self.cosine_layer_dict = {}
                self.max_eigen_layer_dict = {}
                self.mean_eigen_layer_dict = {}
                self.interval_layer_dict = {}
                self.update_times_layer_dict = {}
                for mname in param_names.values():
                       if 'bn' not in mname and 'bias' not in mname and 'norm' not in mname:
                        self.update_times_layer_dict[mname] = 0
                super(Shampoo, self).__init__(params, defaults)

        def init_var_state(self, var, state):
                """Initialize the PyTorch state of for a single variable."""
                state[STEP] = 0
                state[PRECONDITIONER_INTERVAL] = self.hps.preconditioning_compute_steps
                state[MOMENTUM] = torch.zeros_like(var.data, device=var.get_device())
                state[PRECONDITIONER] = Preconditioner(var, self.hps)
                if self.hps.graft_type == LayerwiseGrafting.ADAGRAD:
                        state[GRAFT] = AdagradGraft(self.hps, var)
                elif self.hps.graft_type == LayerwiseGrafting.SGD:
                        state[GRAFT] = SGDGraft(self.hps, var)
                else:
                        state[GRAFT] = Graft(self.hps, var)

        def step(self, closure=None):
                hps = self.hps
                original_grad_norm_sum = 0
                shampoo_norm_sum = 0
                graft_norm_sum = 0

                for group in self.param_groups:
                        lr = group['lr']
                        for p in group['params']:
                                if p.grad is None: continue
                                grad = p.grad.data
                                original_grad_norm = float(torch.norm(grad))

                                if grad.is_sparse:
                                        raise RuntimeError('Shampoo does not support sparse yet')
                                state = self.state[p]
                                if not state:
                                        self.init_var_state(p, state)
                                state[STEP] += 1

                                preconditioner = state[PRECONDITIONER]
                                graft = state[GRAFT]

                                shampoo_prev_grad = None
                                if state[STEP] >= self.hps.start_preconditioning_step:
                                        if state[STEP] >= hps.early_phase_iters and state[STEP] % state[PRECONDITIONER_INTERVAL] == 0:
                                                shampoo_prev_grad = preconditioner.preconditioned_grad(grad)
                                        if state[STEP] < hps.early_phase_iters and state[STEP] % hps.early_preconditioning_compute_steps == 0:
                                                shampoo_prev_grad = preconditioner.preconditioned_grad(grad)

                                # Gather statistics, compute preconditioners
                                graft.add_statistics(grad)
                                if state[STEP] >= hps.early_phase_iters:
                                        if state[STEP] % hps.statistics_compute_steps == 0:
                                                preconditioner.add_statistics(grad)
                                        if state[STEP] % state[PRECONDITIONER_INTERVAL] == 0:
                                                preconditioner.compute_preconditioners()
                                                if 'bn' not in self.param_names[p] and 'bias' not in self.param_names[p] and 'norm' not in self.param_names[p]:
                                                        self.update_times_layer_dict[self.param_names[p]] += 1
                                else:
                                        if state[STEP] % hps.early_statistics_compute_steps == 0:
                                                preconditioner.add_statistics(grad)
                                        if state[STEP] % hps.early_preconditioning_compute_steps == 0:
                                                preconditioner.compute_preconditioners()
                                                if 'bn' not in self.param_names[p] and 'bias' not in self.param_names[p] and 'norm' not in self.param_names[p]:
                                                        self.update_times_layer_dict[self.param_names[p]] += 1

                                # Precondition gradients
                                graft_grad = graft.precondition_gradient(grad)
                                shampoo_grad = grad
                                if state[STEP] >= self.hps.start_preconditioning_step:
                                        shampoo_grad = preconditioner.preconditioned_grad(grad)

                                # Grafting
                                graft_norm = torch.norm(graft_grad)
                                shampoo_norm = torch.norm(shampoo_grad)
                                shampoo_grad.mul_(graft_norm / (shampoo_norm + 1e-16))

                                # total norm
                                original_grad_norm_sum += float(original_grad_norm**2)
                                shampoo_norm_sum += float(shampoo_norm**2)
                                graft_norm_sum += float(graft_norm**2)

                                # For Cosine Similarity
                                if shampoo_prev_grad is not None:
                                        if 'bn' not in self.param_names[p] and 'bias' not in self.param_names[p] and 'norm' not in self.param_names[p]:
                                                cos = torch.nn.CosineSimilarity(dim=0)
                                                cosine_sim_value = cos(shampoo_grad.view(-1), shampoo_prev_grad.view(-1))
                                                self.cosine_layer_dict[self.param_names[p]] = cosine_sim_value
                                                self.max_eigen_layer_dict[self.param_names[p]] = preconditioner.max_eigen_dict
                                                self.mean_eigen_layer_dict[self.param_names[p]] = preconditioner.mean_eigen_dict
                                
                                if 'bn' not in self.param_names[p] and 'bias' not in self.param_names[p] and 'norm' not in self.param_names[p]:
                                        if shampoo_prev_grad is not None and state[STEP] >= self.hps.start_preconditioning_step and self.hps.interval_cosine_thres != -1:
                                                interval_mag_exp = (cosine_sim_value - self.hps.interval_cosine_thres) / (1 - self.hps.interval_cosine_thres)
                                                interval_mag = self.hps.interval_scheduling_factor ** interval_mag_exp
                                                preconditioner_int = math.ceil(state[PRECONDITIONER_INTERVAL] * interval_mag)
                                                if preconditioner_int > self.hps.total_iters:
                                                       preconditioner_int = self.hps.total_iters
                                                state[PRECONDITIONER_INTERVAL] = preconditioner_int
                                        self.interval_layer_dict[self.param_names[p]] = state[PRECONDITIONER_INTERVAL]

                                # Weight decay
                                if self.hps.weight_decay != 0.0:
                                        shampoo_grad.add_(p.data, alpha=self.hps.weight_decay)
                                        graft_grad.add_(p.data, alpha=self.hps.weight_decay)

                                # Momentum and Nesterov momentum, if needed
                                state[MOMENTUM].mul_(group['momentum']).add_(shampoo_grad)
                                graft_momentum = graft.update_momentum(grad, group['momentum'])

                                if state[STEP] >= self.hps.start_preconditioning_step:
                                        momentum_update = state[MOMENTUM]
                                        wd_update = shampoo_grad
                                else:
                                        momentum_update = graft_momentum
                                        wd_update = graft_grad

                                if hps.nesterov:
                                        momentum_update.mul_(group['momentum']).add_(wd_update)

                                if hps.gradient_value_clip != -1:
                                        rho = hps.gradient_value_clip
                                        momentum_update = momentum_update.clamp(min=-rho, max=rho)

                                # Final update
                                p.data.add_(momentum_update, alpha=-lr)

                self.norm_dict = {
                        'original_grad_norm' : original_grad_norm_sum**0.5,
                        'shampoo_norm' : shampoo_norm_sum**0.5,
                        'grad_norm_ratio' : (shampoo_norm_sum**0.5) / (original_grad_norm_sum**0.5),
                        'graft_norm' : graft_norm_sum**0.5
                }

                if shampoo_prev_grad is not None:
                       cosine_sim = self.cosine_layer_dict.values()
                       self.cosine_dict['prev_cos_sim_mean'] = float(sum(cosine_sim) / len(cosine_sim))
                       self.cosine_dict['prev_cos_sim_not_1_count_th_1e-2'] =count_non_ones(cosine_sim, thres = 1e-2)
                       self.cosine_dict['prev_cos_sim_not_1_ratio_th_1e-2'] =float(count_non_ones(cosine_sim, thres = 1e-2) / len(cosine_sim))
                       self.cosine_dict['prev_cos_sim_not_1_count_th_3e-2'] =count_non_ones(cosine_sim, thres = 3e-2)
                       self.cosine_dict['prev_cos_sim_not_1_ratio_th_3e-2'] =float(count_non_ones(cosine_sim, thres = 3e-2) / len(cosine_sim))
                       self.cosine_dict['prev_cos_sim_not_1_count_th_1e-1'] =count_non_ones(cosine_sim, thres = 1e-1)
                       self.cosine_dict['prev_cos_sim_not_1_ratio_th_1e-1'] =float(count_non_ones(cosine_sim, thres = 1e-1) / len(cosine_sim))

                       sorted_nums = sorted(cosine_sim, reverse=False)
                       self.cosine_dict['prev_cos_sim_max_1'] = float(sorted_nums[-1])
                       self.cosine_dict['prev_cos_sim_min_1'] = float(sorted_nums[0])
                       self.cosine_dict['prev_cos_sim_min_2'] = float(sorted_nums[1])
                       self.cosine_dict['prev_cos_sim_min_3'] = float(sorted_nums[2])
                       self.cosine_dict['prev_cos_sim_min_4'] = float(sorted_nums[3])
                       self.cosine_dict['prev_cos_sim_min_5'] = float(sorted_nums[4])

def count_non_ones(lst, thres = 1e-3):
    count = 0
    for element in lst:
        if (1 - element) >= thres:
            count += 1
    return count

@torch.no_grad()
def PowerIter(mat_g, error_tolerance=1e-6, num_iters=100):
        """Power iteration.

        Compute the maximum eigenvalue of mat, for scaling.
        v is a random vector with values in (-1, 1)

        Args:
                mat_g: the symmetric PSD matrix.
                error_tolerance: Iterative exit condition.
                num_iters: Number of iterations.

        Returns:
                eigen vector, eigen value, num_iters
        """
        v = torch.rand(list(mat_g.shape)[0], device=mat_g.get_device()) * 2 - 1
        error = 1
        iters = 0
        singular_val = 0
        while error > error_tolerance and iters < num_iters:
                v = v / torch.norm(v)
                mat_v = torch.mv(mat_g, v)
                s_v = torch.dot(v, mat_v)
                error = torch.abs(s_v - singular_val)
                v = mat_v
                singular_val = s_v
                iters += 1
        return singular_val, v / torch.norm(v), iters


@torch.no_grad()
def MatPower(mat_m, p):
        """Computes mat_m^p, for p a positive integer.

        Args:
                mat_m: a square matrix
                p: a positive integer

        Returns:
                mat_m^p
        """
        if p in [1, 2, 4, 8, 16, 32]:
                p_done = 1
                res = mat_m
                while p_done < p:
                        res = torch.matmul(res, res)
                        p_done *= 2
                return res

        power = None
        while p > 0:
                if p % 2 == 1:
                        power = torch.matmul(mat_m, power) if power is not None else mat_m
                p //= 2
                mat_m = torch.matmul(mat_m, mat_m)
        return power

@torch.no_grad()
def matrix_neg_power(mat_g, p):
    U, S, V = torch.svd(mat_g)
    S_neg_power = torch.diag(S ** (-1 / p))
    return U @ S_neg_power @ V.t()


@torch.no_grad()
def ComputePower(mat_g, p,
                                                                 iter_count=100,
                                                                 error_tolerance=1e-6,
                                                                 ridge_epsilon=1e-6,
                                                                 recursive = False):
        """A method to compute G^{-1/p} using a coupled Newton iteration.

        See for example equation 3.2 on page 9 of:
        A Schur-Newton Method for the Matrix p-th Root and its Inverse
        by Chun-Hua Guo and Nicholas J. Higham
        SIAM Journal on Matrix Analysis and Applications,
        2006, Vol. 28, No. 3 : pp. 788-804
        https://pdfs.semanticscholar.org/0abe/7f77433cf5908bfe2b79aa91af881da83858.pdf

        Args:
                mat_g: A square positive semidefinite matrix
                p: a positive integer
                iter_count: Stop iterating after this many rounds.
                error_tolerance: Threshold for stopping iteration
                ridge_epsilon: We add this times I to G, to make is positive definite.
                                                                         For scaling, we multiply it by the largest eigenvalue of G.
        Returns:
                (mat_g + rI)^{-1/p} (r = ridge_epsilon * max_eigenvalue of mat_g).
        """
        tf32_flag = torch.backends.cuda.matmul.allow_tf32
        mat_g_raw = torch.clone(mat_g)
        shape = list(mat_g.shape)
        if len(shape) == 1:
                return torch.pow(mat_g + ridge_epsilon, -1/p)
        identity = torch.eye(shape[0], device=mat_g.get_device())
        if shape[0] == 1:
                return identity
        alpha = -1.0/p
        max_ev, _, _ = PowerIter(mat_g)
        ridge_epsilon *= max_ev
        mat_g = mat_g + ridge_epsilon * identity
        z = (1 + p) / (2 * torch.norm(mat_g))
        # The best value for z is
        # (1 + p) * (c_max^{1/p} - c_min^{1/p}) /
        #                                                (c_max^{1+1/p} - c_min^{1+1/p})
        # where c_max and c_min are the largest and smallest singular values of
        # mat_g.
        # The above estimate assumes that c_max > c_min * 2^p
        # Can replace above line by the one below, but it is less accurate,
        # hence needs more iterations to converge.
        # z = (1 + p) / tf.trace(mat_g)
        # If we want the method to always converge, use z = 1 / norm(mat_g)
        # or z = 1 / tf.trace(mat_g), but these can result in many
        # extra iterations.

        mat_root = identity * torch.pow(z, 1.0/p)
        mat_m = mat_g * z
        error = torch.max(torch.abs(mat_m - identity))
        count = 0
        while error > error_tolerance and count < iter_count:
                tmp_mat_m = (1 - alpha) * identity + alpha * mat_m
                new_mat_root = torch.matmul(mat_root, tmp_mat_m)
                new_mat_m = torch.matmul(MatPower(tmp_mat_m, p), mat_m)
                new_error = torch.max(torch.abs(new_mat_m - identity))
                if new_error > error:
                        if torch.backends.cuda.matmul.allow_tf32:
                                torch.backends.cuda.matmul.allow_tf32 = False
                                continue
                        if new_error > error * 1.2:
                                max_ev = None
                                if not recursive:
                                        mat_root, max_ev = ComputePower(mat_g_raw, p, iter_count, error_tolerance, ridge_epsilon, True)
                                else:
                                        mat_root = matrix_neg_power(mat_g_raw, p)
                                break
                if torch.isclose(new_error, error):
                        torch.backends.cuda.matmul.allow_tf32 = False
                mat_m = new_mat_m
                mat_root = new_mat_root
                error = new_error
                count += 1
        # If you want to check the correctness:
        # if random.randint(0, 1000) == 0:
        #         ground_truth = matrix_neg_power(mat_g_raw, p)
        #         error_gt = (mat_root - ground_truth).norm() / ground_truth.norm()
        #         print(f'ComputePower: count={count}, error={error:.7f}, gt_error={error_gt:.7f}')
        torch.backends.cuda.matmul.allow_tf32 = tf32_flag
        return mat_root, max_ev


VAR_SHAPE = 'var_shape'
VAR_SPLITS = 'var_splits'


def merge_small_dims(var_shape, reshape_size):
    """Computes the shape of the variable for preconditioning.

    If the variable has several small dimensions, we can reshape it so
    that there are fewer big ones. e.g for a convolution (512, 5, 5, 1024)
    we could reshape it into (512, 25, 1024).

    Args:
        var_shape: the shape of the variable
        reshape_size: maximum size of a reshaped dimension
    Returns:
        shape: a list of integers. Product(shape) = number of elements in var.
    """
    if var_shape and np.all(np.array(var_shape) == 1):
        return [1]
    shape = []
    product = 1
    for d in var_shape:
        if product * d <= reshape_size:
            product *= d
        else:
            if product > 1: shape.append(product)
            product = d
    if product > 1:
        shape.append(product)
    return shape

def compute_splits(var_shape, block_size):
    """Splits larger dimensions into smaller ones, for preconditioning.

    For example, if a variable has shape (4096, 512), we might split the
    4096 into 4 blocks, so we effectively have 4 variables of size
    (1024, 512) each.

    Args:
        var_shape: list of integers, the shape to be split
        block_size: the maximum dimension of each block
    Returns:
        splits: set of tuples (i, split) if the i-th dimension should be split
        split_sizes: an array of tuples, one per dimension, each indicating how
                                 to split that dimension.
    """
    splits = []
    split_sizes = []
    for i, d in enumerate(var_shape):
        if block_size > 0 and d > block_size:
            nsplit = math.ceil(d / block_size)
            sizes = np.ones(nsplit, dtype=np.int32) * block_size
            if d % block_size > 0:
                sizes[-1] = d % block_size
            splits.append((i, tuple(sizes)))
            split_sizes.append(sizes)
        else:
            split_sizes.append(np.array([d], dtype=np.int32))
    return splits, split_sizes


def split_grad(state, grad):
    """Split up the gradient according to the blocking strategy."""
    if len(state[VAR_SHAPE]) < len(list(grad.shape)):
        grad = torch.reshape(grad, state[VAR_SHAPE])
    grads = [grad]
    for i, split_sizes in state[VAR_SPLITS]:
        split_grads = []
        for grad in grads:
            split_grads.extend(torch.split(grad, split_sizes, dim=i))
        grads = split_grads
    return grads


def merge_grads(state, grads):
    """Merge the split gradients back into a single array."""
    for i, split_sizes in reversed(state[VAR_SPLITS]):
        n = len(split_sizes)
        conc_grads = []
        ind = 0
        while ind < len(grads):
            conc_grads.append(torch.cat(grads[ind:ind+n], axis=i))
            ind += n
        grads = conc_grads
    assert len(grads) == 1
    return grads[0]

def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))

def ComputeInverse(stat, damping, dmp_opt = 'mean'):
        if dmp_opt == 'mean':
                eps = damping * torch.trace(stat).item() / stat.shape[-1]
                return cholesky_inv(stat, eps), None
        elif dmp_opt == 'max':
                max_eig = power_method_max_eigen(stat)
                eps = max_eig*damping
                return cholesky_inv(stat, eps), max_eig

def cholesky_inv(X, damping=1e-12):
    diag = torch.diagonal(X)
    diag += damping
    u = torch.linalg.cholesky(X)
    diag -= damping
    return torch.cholesky_inverse(u)

def power_method_max_eigen(matrix, max_iterations=100, tolerance=1e-5):
    n, m = matrix.size()
    if n != m:
        raise ValueError("Matrix must be square.")

    device = matrix.device  # matrixのデバイスを取得

    # 初期ベクトル (デバイスを揃える)
    b_k = torch.rand(n, device=device)

    for _ in range(max_iterations):
        # 行列Aとベクトルb_kの積
        b_k1 = torch.matmul(matrix, b_k)
        
        # 次のベクトル
        b_k1_norm = torch.norm(b_k1).item()
        b_k_next = b_k1 / b_k1_norm
        
        if torch.norm(b_k_next - b_k).item() < tolerance:
            break
        
        b_k = b_k_next

    # 固有値を返す (デバイスを揃える)
    return float(b_k1_norm)