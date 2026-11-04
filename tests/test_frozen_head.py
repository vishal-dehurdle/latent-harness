"""Unit tests for Module C: Frozen-Head Decoding Constraint."""

import pytest
import torch
import torch.nn as nn

from latent_harness.frozen_head import (
    SemanticKeywordExtractor,
    FrozenHeadConstraint,
)


def test_semantic_keyword_extractor_text():
    extractor = SemanticKeywordExtractor(top_k=5)
    text = "First we compute the integral from 0 to 1 of x^2 dx which evaluates to 1/3."
    keywords = extractor.extract_from_text(text)

    assert len(keywords) > 0
    assert len(keywords) <= 5
    # Common stop words like 'we', 'the', 'from', 'to', 'of' must be excluded
    for w in ["the", "we", "from", "to", "of"]:
        assert w not in keywords


def test_frozen_head_constraint_forward():
    hidden_size = 64
    vocab_size = 500
    k = 4
    batch_size = 2

    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    constraint = FrozenHeadConstraint(
        lm_head=lm_head,
        hidden_size=hidden_size,
    )

    # Check weights are frozen
    for param in constraint.lm_head.parameters():
        assert not param.requires_grad

    z = torch.randn(batch_size, k, hidden_size, requires_grad=True)
    keywords = [[10, 25, 42], [50, 100, 200]]

    loss, logits, eff_rank = constraint(z, keyword_ids=keywords)

    assert loss.dim() == 0  # scalar
    assert loss.item() > 0.0
    assert logits.shape == (batch_size, k, vocab_size)
    assert eff_rank >= 1.0

    # Test backpropagation to student latent representations
    loss.backward()
    assert z.grad is not None
    # Frozen head parameters should NOT receive gradients
    for param in constraint.lm_head.parameters():
        assert param.grad is None


def test_effective_rank_calculation():
    # 1. Collapsed rank-1 tensor: all tokens are identical
    v = torch.randn(1, 64)
    z_collapsed = v.repeat(16, 1)  # (16, 64) all identical
    rank_collapsed = FrozenHeadConstraint.compute_effective_rank(z_collapsed)
    assert rank_collapsed <= 1.5

    # 2. Orthogonal/rich tensor: high effective rank
    q, _ = torch.linalg.qr(torch.randn(64, 64))
    z_rich = q[:16, :]  # 16 orthogonal vectors
    rank_rich = FrozenHeadConstraint.compute_effective_rank(z_rich)
    assert rank_rich > 5.0
