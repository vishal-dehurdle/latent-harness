"""Frozen-head keyword loss for latent states.

Decodes each latent state through the model's frozen LM head and scores it against keywords from the
reasoning trace, so the latents stay readable in the vocabulary basis. It was proposed as a guard against
the latents collapsing; in the GSM8K study they collapsed with it on (see the README).
"""

from typing import Any, List, Optional, Set, Tuple, Union
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


# Default punctuation and stop tokens to filter from explicit teacher spans
DEFAULT_STOP_WORDS: Set[str] = {
    "", " ", "\n", "\t", "the", "a", "an", "is", "are", "was", "were",
    "of", "in", "to", "for", "with", "on", "at", "by", "from", "up",
    "about", "into", "over", "after", "it", "its", "this", "that", "these",
    "those", "and", "or", "but", "so", "then", "because", "as", "if",
    "we", "i", "me", "my", "us", "our", "you", "your", "they", "them",
    "their", "he", "him", "his", "she", "her", "which", "who", "whom",
    "what", "where", "when", "why", "how", "can", "will", "would",
    "should", "could", "have", "has", "had", "do", "does", "did",
    "be", "been", "being",
    ".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "\"",
    "'", "`", "$", "=", "+", "-", "*", "/", "<", ">", "<think>", "</think>",
}


class SemanticKeywordExtractor:
    """Extracts informative semantic reasoning anchor tokens from teacher spans."""

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        top_k: int = 16,
        min_token_len: int = 2,
        custom_stop_words: Optional[Set[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.min_token_len = min_token_len
        self.stop_words = set(DEFAULT_STOP_WORDS)
        if custom_stop_words:
            self.stop_words.update(custom_stop_words)

    def extract_from_text(self, text: str) -> List[str]:
        """Extract key reasoning terms and formulas from raw text."""
        # Clean special thinking tags
        cleaned = re.sub(r"</?think>", " ", text)
        # Tokenize by whitespace and punctuation boundaries while preserving numbers and math
        words = re.findall(r"[A-Za-z0-9_]+|[\+\-\*\/\^\=]", cleaned)
        
        candidates = []
        for w in words:
            w_lower = w.lower()
            if w_lower in self.stop_words or len(w) < self.min_token_len:
                continue
            candidates.append(w)

        # Count frequency of candidates
        counts = {}
        for c in candidates:
            counts[c] = counts.get(c, 0) + 1

        # Sort by frequency and take top_k
        sorted_candidates = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_candidates[: self.top_k]]

    def extract_token_ids(
        self,
        text_or_ids: Union[str, List[int], torch.Tensor],
        max_keywords: Optional[int] = None,
    ) -> List[int]:
        """Convert teacher span to unique high-information token IDs."""
        limit = max_keywords or self.top_k

        if isinstance(text_or_ids, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer required to extract token IDs from string.")
            keywords = self.extract_from_text(text_or_ids)
            token_ids: List[int] = []
            for kw in keywords:
                encoded = self.tokenizer.encode(kw, add_special_tokens=False)
                token_ids.extend(encoded)
            # Unique preserve order
            seen = set()
            unique_ids = [tid for tid in token_ids if not (tid in seen or seen.add(tid))]
            return unique_ids[:limit]

        if isinstance(text_or_ids, torch.Tensor):
            id_list = text_or_ids.flatten().tolist()
        else:
            id_list = list(text_or_ids)

        if self.tokenizer is not None:
            # Filter stop tokens by decoding
            filtered = []
            seen = set()
            for tid in id_list:
                if tid in seen:
                    continue
                seen.add(tid)
                decoded = self.tokenizer.decode([tid]).strip().lower()
                if decoded not in self.stop_words and len(decoded) >= self.min_token_len:
                    filtered.append(tid)
                if len(filtered) >= limit:
                    break
            return filtered

        # Fallback if no tokenizer provided: unique IDs
        seen = set()
        unique = [tid for tid in id_list if not (tid in seen or seen.add(tid))]
        return unique[:limit]


class FrozenHeadConstraint(nn.Module):
    """Enforces the Frozen-Head Decoding Constraint:

    1. Applies optional projection phi(z_k) to student latent states.
    2. Projects through frozen lm_head (W, b) to obtain logits p(. | z_k).
    3. Computes cross-entropy loss against teacher span keywords or soft teacher logits.
    4. Computes covariance matrix effective rank to monitor dimensional collapse.
    """

    def __init__(
        self,
        lm_head: nn.Module,
        hidden_size: int = 3584,
        proj_dim: Optional[int] = None,
        loss_type: str = "keyword_ce",
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        """Initialize FrozenHeadConstraint.

        Args:
            lm_head: Frozen pre-trained linear head (hidden_size -> vocab_size)
            hidden_size: Hidden dimension of latent states
            proj_dim: Dimension for optional projection phi. If None, uses hidden_size.
            loss_type: 'keyword_ce' (multi-label keyword CE) or 'soft_distillation' (KL with teacher logits)
            temperature: Softmax temperature for logits
            label_smoothing: Smoothing factor for target distributions
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.proj_dim = proj_dim or hidden_size
        self.loss_type = loss_type
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        # Freeze the pre-trained lm_head completely
        self.lm_head = lm_head
        for param in self.lm_head.parameters():
            param.requires_grad = False

        # Optional projection phi(z)
        if self.proj_dim != self.hidden_size:
            self.phi = nn.Sequential(
                nn.Linear(self.hidden_size, self.proj_dim, bias=False),
                nn.RMSNorm(self.proj_dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(self.proj_dim),
            )
        else:
            self.phi = nn.Identity()

    @staticmethod
    def compute_effective_rank(z: torch.Tensor, eps: float = 1e-12) -> float:
        """Compute the continuous effective rank of latent representations Z.

        Effective rank is defined as the exponential of the spectral entropy:
            erank(Z) = exp(-sum_i p_i * log(p_i))
            where p_i = sigma_i / sum_j sigma_j (normalized singular values).

        Args:
            z: (B, K, d) or (N, d) latent state tensor

        Returns:
            erank: Effective rank in [1, min(N, d)]
        """
        with torch.no_grad():
            flat_z = z.reshape(-1, z.shape[-1]).float()
            n, d = flat_z.shape
            if n <= 1:
                return 1.0

            # SVD of representation matrix Z
            try:
                s = torch.linalg.svdvals(flat_z.cpu())
            except Exception:
                cov = (flat_z.T @ flat_z) / n
                s = torch.linalg.eigvalsh(cov.cpu()).clamp(min=0.0).sqrt()

            s_sum = s.sum()
            if s_sum < eps:
                return 1.0

            p = s / s_sum
            p = p[p > eps]
            entropy = -torch.sum(p * torch.log(p)).item()
            return math.exp(entropy)

    def forward(
        self,
        z: torch.Tensor,
        keyword_ids: Optional[Union[List[List[int]], torch.Tensor]] = None,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Compute the Frozen-Head Decoding Loss.

        Args:
            z: (B, K, d) Student latent states
            keyword_ids: List of keyword token ID lists per batch item, or padded (B, M) Tensor
            teacher_logits: Optional (B, S, V) teacher span logits for soft distillation

        Returns:
            loss: Scalar auxiliary projection cross-entropy loss
            student_logits: (B, K, V) logits over vocabulary
            eff_rank: Effective rank of student latent states
        """
        b_size, k_len, _ = z.shape

        # Step 1: Optional projection phi(z)
        z_proj = self.phi(z)

        # Step 2: Pass through frozen lm_head: (B, K, V)
        student_logits = self.lm_head(z_proj) / self.temperature
        vocab_size = student_logits.shape[-1]

        # Monitor effective rank
        eff_rank = self.compute_effective_rank(z)

        # Step 3: Compute Loss
        if self.loss_type == "soft_distillation" and teacher_logits is not None:
            # KL divergence / cross-entropy with mean-pooled teacher logits
            # Teacher target: average distribution over reasoning span
            teacher_probs = F.softmax(teacher_logits.mean(dim=1, keepdim=True) / self.temperature, dim=-1)
            # Broadcast to all K latent tokens: (B, K, V)
            teacher_probs = teacher_probs.expand(-1, k_len, -1)
            log_student_probs = F.log_softmax(student_logits, dim=-1)
            loss = F.kl_div(log_student_probs, teacher_probs, reduction="batchmean")

        elif keyword_ids is not None:
            # Keyword Cross-Entropy: latent tokens should decode into the anchor keywords
            loss_list = []
            for b in range(b_size):
                if isinstance(keyword_ids, torch.Tensor):
                    kw = keyword_ids[b]
                    kw = kw[kw >= 0].tolist()
                else:
                    kw = keyword_ids[b] if b < len(keyword_ids) else []

                if not kw:
                    continue

                kw_tensor = torch.tensor(kw, device=z.device, dtype=torch.long)
                # Ensure valid indices
                kw_tensor = kw_tensor[kw_tensor < vocab_size]
                if len(kw_tensor) == 0:
                    continue

                # Target distribution: uniform over keywords
                target_dist = torch.zeros(vocab_size, device=z.device, dtype=student_logits.dtype)
                target_dist.scatter_(0, kw_tensor, 1.0 / len(kw_tensor))

                if self.label_smoothing > 0.0:
                    smooth = self.label_smoothing / vocab_size
                    target_dist = (1.0 - self.label_smoothing) * target_dist + smooth

                # Student log-softmax for all K tokens in this batch item: (K, V)
                log_probs = F.log_softmax(student_logits[b], dim=-1)
                # Cross-entropy: - sum_{v} target_v * log_prob_kv
                item_ce = -torch.sum(target_dist.unsqueeze(0) * log_probs, dim=-1).mean()
                loss_list.append(item_ce)

            if loss_list:
                loss = torch.stack(loss_list).mean()
            else:
                loss = torch.tensor(0.0, device=z.device, dtype=z.dtype, requires_grad=True)

        else:
            # Unsupervised rank regularization fallback if no keywords provided
            # Penalize entropy collapse by maximizing entropy of predicted distribution
            probs = F.softmax(student_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-9)), dim=-1).mean()
            # Target maximum vocabulary utilization
            loss = -0.01 * entropy

        return loss, student_logits, eff_rank
