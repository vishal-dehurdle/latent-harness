"""Unit tests for the Sinkhorn optimal-transport solver."""

import torch

from latent_harness.sinkhorn import sinkhorn_cost_matrix, sinkhorn_optimal_transport


def test_sinkhorn_cost_matrix():
    # z: (2, 3, 4), h: (2, 5, 4)
    z = torch.randn(2, 3, 4)
    h = torch.randn(2, 5, 4)
    cost = sinkhorn_cost_matrix(z, h)
    assert cost.shape == (2, 3, 5)
    assert (cost >= 0.0).all()

    # Manual distance check for item (0, 0) and (0, 0)
    expected_dist = torch.sum((z[0, 0] - h[0, 0]) ** 2)
    assert torch.isclose(cost[0, 0, 0], expected_dist, atol=1e-5)


def test_sinkhorn_optimal_transport_convergence():
    z = torch.nn.functional.normalize(torch.randn(2, 4, 32), dim=-1)
    h = torch.nn.functional.normalize(torch.randn(2, 16, 32), dim=-1)

    cost, pi = sinkhorn_optimal_transport(
        z=z,
        h=h,
        epsilon=0.1,
        max_iters=50,
        tol=1e-6,
    )

    assert cost.shape == (2,)
    assert pi.shape == (2, 4, 16)
    assert (cost > 0.0).all()

    # Row marginal constraint: sum_j Pi_ij = 1/K = 1/4 = 0.25
    row_sums = pi.sum(dim=-1)
    assert torch.allclose(row_sums, torch.full_like(row_sums, 0.25), atol=1e-2)

    # Column marginal constraint: sum_i Pi_ij = 1/S = 1/16 = 0.0625
    col_sums = pi.sum(dim=1)
    assert torch.allclose(col_sums, torch.full_like(col_sums, 0.0625), atol=1e-2)


def test_sinkhorn_log_domain():
    z = torch.randn(2, 4, 16)
    h = torch.randn(2, 8, 16)

    # Low epsilon requires log-domain stabilization
    cost_log, pi_log = sinkhorn_optimal_transport(
        z=z,
        h=h,
        epsilon=0.01,
        log_domain=True,
    )
    assert not torch.isnan(cost_log).any()
    assert not torch.isnan(pi_log).any()


def test_sinkhorn_gradient_flow():
    z = torch.randn(2, 4, 16, requires_grad=True)
    h = torch.randn(2, 8, 16)

    cost, _ = sinkhorn_optimal_transport(z, h, epsilon=0.1, log_domain=True)

    cost.mean().backward()
    assert z.grad is not None
    assert z.grad.shape == z.shape
    assert not torch.isnan(z.grad).any()


def test_sinkhorn_with_mask():
    z = torch.randn(2, 4, 16)
    h = torch.randn(2, 8, 16)
    mask = torch.tensor([
        [True, True, True, True, False, False, False, False],
        [True, True, True, True, True, True, False, False],
    ])

    cost, pi = sinkhorn_optimal_transport(
        z=z,
        h=h,
        epsilon=0.1,
        mask_h=mask,
    )
    assert cost.shape == (2,)
    assert pi.shape == (2, 4, 8)
    # Masked positions should have 0 transport mass
    assert torch.all(pi[0, :, 4:] == 0.0)
    assert torch.all(pi[1, :, 6:] == 0.0)
