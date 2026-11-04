"""Entropic optimal transport (Sinkhorn-Knopp) between K latent states and a reasoning trace's states.

Pure PyTorch and batched, with a log-domain solver for small epsilon and an optional mask for
variable-length traces. The latent model's alignment loss (latent/train.py) uses
`sinkhorn_optimal_transport` with the coupling held fixed.
"""

from typing import Optional, Tuple

import torch


def sinkhorn_cost_matrix(z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distance cost matrix C.

    Args:
        z: (B, K, d) or (K, d) student latent embeddings
        h: (B, S, d) or (S, d) teacher explicit hidden states

    Returns:
        C: (B, K, S) or (K, S) cost matrix where C[..., k, j] = ||z_k - h_j||_2^2
    """
    has_batch = z.dim() == 3
    if not has_batch:
        z = z.unsqueeze(0)
        h = h.unsqueeze(0)

    # ||z - h||^2 = ||z||^2 + ||h||^2 - 2 * z @ h^T
    z_sq = torch.sum(z ** 2, dim=-1, keepdim=True)       # (B, K, 1)
    h_sq = torch.sum(h ** 2, dim=-1, keepdim=True)       # (B, S, 1)
    cross = torch.bmm(z, h.transpose(1, 2))              # (B, K, S)

    c = z_sq + h_sq.transpose(1, 2) - 2.0 * cross
    c = torch.clamp(c, min=0.0)

    if not has_batch:
        return c.squeeze(0)
    return c


def sinkhorn_optimal_transport(
    z: torch.Tensor,
    h: torch.Tensor,
    epsilon: float = 0.1,
    max_iters: int = 50,
    tol: float = 1e-6,
    log_domain: bool = False,
    detach_coupling: bool = True,
    mask_h: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Sinkhorn Optimal Transport cost and coupling matrix between Z and H.

    Args:
        z: (B, K, d) student latent states
        h: (B, S, d) teacher explicit reasoning hidden states
        epsilon: Entropic regularization parameter
        max_iters: Maximum Sinkhorn-Knopp iterations
        tol: Convergence tolerance threshold
        log_domain: Use log-domain stabilization (recommended for epsilon < 0.05)
        detach_coupling: Whether to detach coupling Pi* in the loss computation
        mask_h: Optional (B, S) boolean mask indicating valid tokens in teacher span

    Returns:
        loss: (B,) or scalar transport cost <Pi*, C>
        pi: (B, K, S) optimal transport coupling matrix
    """
    has_batch = z.dim() == 3
    if not has_batch:
        z = z.unsqueeze(0)
        h = h.unsqueeze(0)
        if mask_h is not None and mask_h.dim() == 1:
            mask_h = mask_h.unsqueeze(0)

    b_size, k_len, d_dim = z.shape
    _, s_len, _ = h.shape

    # Compute pairwise squared Euclidean cost C: (B, K, S)
    cost = sinkhorn_cost_matrix(z, h)

    # Marginals a (1/K) and b (1/S)
    a = torch.full((b_size, k_len), 1.0 / float(k_len), device=z.device, dtype=z.dtype)

    if mask_h is not None:
        # Variable-length spans: normalize b over valid tokens
        mask_float = mask_h.to(dtype=h.dtype)
        s_counts = torch.clamp(mask_float.sum(dim=-1, keepdim=True), min=1.0)
        b = mask_float / s_counts
    else:
        b = torch.full((b_size, s_len), 1.0 / float(s_len), device=h.device, dtype=h.dtype)

    scale = (cost.max() - cost.min()) / max(epsilon, 1e-6)
    if log_domain or epsilon < 0.5 or scale > 20.0:
        # Log-domain stabilized Sinkhorn iterations
        log_a = torch.log(torch.clamp(a, min=1e-12))
        log_b = torch.log(torch.clamp(b, min=1e-12))

        f = torch.zeros_like(log_a)
        g = torch.zeros_like(log_b)

        for _ in range(max_iters):
            # f_i = epsilon * (log_a - logsumexp_j((g_j - C_ij) / epsilon))
            kernel_g = (g.unsqueeze(1) - cost) / epsilon
            if mask_h is not None:
                kernel_g = kernel_g.masked_fill(~mask_h.unsqueeze(1), -1e9)
            new_f = epsilon * (log_a - torch.logsumexp(kernel_g, dim=-1))

            # g_j = epsilon * (log_b - logsumexp_i((f_i - C_ij) / epsilon))
            kernel_f = (new_f.unsqueeze(-1) - cost) / epsilon
            new_g = epsilon * (log_b - torch.logsumexp(kernel_f, dim=1))

            diff = torch.max(torch.abs(new_f - f))
            f, g = new_f, new_g
            if diff.item() < tol:
                break

        # Pi = exp((f + g - C) / epsilon)
        log_pi = (f.unsqueeze(-1) + g.unsqueeze(1) - cost) / epsilon
        if mask_h is not None:
            log_pi = log_pi.masked_fill(~mask_h.unsqueeze(1), -1e9)
        pi = torch.exp(log_pi)

    else:
        # Scaled Gibbs Kernel K = exp(-C / epsilon)
        # Shift cost by min along dim to prevent numerical underflow
        min_c = cost.min(dim=-1, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        c_shifted = (cost - min_c) / epsilon
        c_clamped = torch.clamp(c_shifted, max=80.0)
        kernel = torch.exp(-c_clamped)

        if mask_h is not None:
            kernel = kernel * mask_h.unsqueeze(1).to(dtype=kernel.dtype)

        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(max_iters):
            # u = a / (K @ v)
            kv = torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1)
            u = a / torch.clamp(kv, min=1e-12)

            # v = b / (K^T @ u)
            ktu = torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
            v = b / torch.clamp(ktu, min=1e-12)

        # Pi* = diag(u) @ K @ diag(v) = u_i * K_ij * v_j
        pi = u.unsqueeze(-1) * kernel * v.unsqueeze(1)

    if detach_coupling:
        pi_used = pi.detach()
    else:
        pi_used = pi

    # Transport cost <Pi*, C> = sum_{k, j} Pi_{k, j} * C_{k, j}
    cost_per_batch = torch.sum(pi_used * cost, dim=(-2, -1))

    if not has_batch:
        return cost_per_batch.squeeze(0), pi.squeeze(0)
    return cost_per_batch, pi
