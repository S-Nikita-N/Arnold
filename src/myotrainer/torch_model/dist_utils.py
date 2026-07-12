"""
Safe log_prob for LowRankMultivariateNormal.

On macOS, Accelerate uses Metal-backed BLAS for large non-contiguous batch
matmuls. In forked processes (when the parent has initialized MPS), this
crashes with SIGSEGV. PyTorch's LowRankMultivariateNormal.log_prob internally
creates non-contiguous transposes for the Woodbury identity computation,
triggering the crash for large latent dimensions (e.g. 512 in lattice/MoE).

This module provides a drop-in replacement that ensures all matmul operands
are contiguous before multiplication, avoiding the Metal BLAS crash path.
"""

import math
import torch

from torch.distributions import LowRankMultivariateNormal


########################################
#            Safe log_prob             #
########################################


def safe_lrmvn_log_prob(
    dist: LowRankMultivariateNormal,
    value: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log_prob identical to dist.log_prob(value), but with all matmul
    operands forced to be contiguous.

    Root cause: W.mT produces a non-contiguous [B, k, n] tensor. For large k
    (e.g. 512), Accelerate dispatches the bmm to a Metal BLAS kernel, which
    segfaults in forked processes that inherited MPS context from the parent.
    Making tensors contiguous before matmul forces the CPU BLAS path.
    """
    diff = value - dist.loc  # [B, n]
    W = dist._unbroadcasted_cov_factor  # [B, n, k]
    D = dist._unbroadcasted_cov_diag  # [B, n]
    cap_tril = dist._capacitance_tril  # [B, k, k]
    n = dist.event_shape[0]

    # Mahalanobis distance via Woodbury identity:
    #   (x-μ)^T Σ^{-1} (x-μ) = diff^T D^{-1} diff - v^T C^{-1} v
    # where v = W^T D^{-1} diff, C = I + W^T D^{-1} W (capacitance)
    D_inv = 1.0 / D
    # W * D_inv gives D^{-1}W column-wise, then transpose → contiguous
    Wt_Dinv = (
        (W * D_inv.unsqueeze(-1)).transpose(-2, -1).contiguous()
    )  # [B, k, n]
    Wt_Dinv_diff = Wt_Dinv @ diff.unsqueeze(-1)  # [B, k, 1]

    # C^{-1} v  via  solve_triangular  (L x = v,  then L^T y = x)
    M = torch.linalg.solve_triangular(cap_tril, Wt_Dinv_diff, upper=False)

    Dinv_diff = diff * D_inv
    mahal = (diff * Dinv_diff).sum(-1) - (M * M).sum((-2, -1))

    # log|Σ| = log|D| + log|C|,  log|C| = 2·Σ log(diag(L))
    log_det_D = D.log().sum(-1)
    log_det_C = 2.0 * cap_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    log_det = log_det_D + log_det_C

    return -0.5 * (n * math.log(2 * math.pi) + log_det + mahal)
