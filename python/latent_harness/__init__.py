"""latent-harness: a controlled test bed for latent-token reasoning on DeepSeek-R1-Distill models.

- `latent_harness.evals`: benchmark evaluation (GSM8K, MATH-500, AIME 2024, GPQA-Diamond) under the
  think / no_think / direct conditions, with bootstrap CIs and per-run manifests.
- `latent_harness.latent`: K latent pause tokens refined by a LOTUS loop over the backbone, trained with
  Sinkhorn alignment to the model's own reasoning; K=0 is the fine-tuned no-latent control.
- `sinkhorn`, `frozen_head`: the alignment and keyword losses the latent model trains with.
"""

from .frozen_head import FrozenHeadConstraint, SemanticKeywordExtractor
from .sinkhorn import sinkhorn_cost_matrix, sinkhorn_optimal_transport

__version__ = "0.1.0"

__all__ = [
    "FrozenHeadConstraint",
    "SemanticKeywordExtractor",
    "sinkhorn_cost_matrix",
    "sinkhorn_optimal_transport",
]
