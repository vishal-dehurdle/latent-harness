#!/usr/bin/env python3
"""Train the latent reasoner on data from build_latent_data.py.

The student is always the model (and exact revision) that produced the teacher
states. Stage 1 aligns the latent states to the teacher's reasoning with
Sinkhorn OT; stage 2 trains the answer with the OT and frozen-head losses
alongside. Interrupted runs resume from <out-dir>/checkpoint.

Examples:
  # The full method, K=4 latents, R=4 iterations
  python scripts/train_latent.py --data-dir outputs/latent_data/gsm8k_train --out-dir outputs/latent/k4

  # The control every result needs: the same fine-tuning with no latents
  python scripts/train_latent.py --data-dir outputs/latent_data/gsm8k_train --out-dir outputs/latent/k0 \\
      --num-latents 0 --stage1-steps 0 --ot-weight 0 --frozen-head-weight 0

  # Evaluate a run exactly like the baseline's direct condition
  python scripts/eval_baseline.py --backend latent --model outputs/latent/k4/final --benchmarks gsm8k
"""

import argparse
import json
import os
import sys

# PyTorch lets the MPS allocator grow to 1.7x the GPU's recommended working set, which on a 16 GB Mac is
# past physical memory: training then swaps for hours instead of failing. Cap it at the working set; the
# low (soft) watermark defaults to 1.4 and must stay below the cap.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")

import torch

from latent_harness import provenance
from latent_harness.latent import LatentConfig, LatentReasoner, LatentTrainingSet, TeacherStore, TrainConfig, resolve_device, train


def main() -> None:
    defaults_latent, defaults_train = LatentConfig(), TrainConfig(data_dir="", out_dir="")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto")

    model_args = parser.add_argument_group("latent model")
    model_args.add_argument("--num-latents", type=int, default=defaults_latent.num_latents, help="K")
    model_args.add_argument("--num-iterations", type=int, default=defaults_latent.num_iterations, help="R")
    model_args.add_argument("--no-spectral-norm", action="store_true", help="Ablation: unconstrained feedback map")
    model_args.add_argument("--feedback-scale", type=float, default=defaults_latent.feedback_scale, help="gamma")
    model_args.add_argument("--lora-rank", type=int, default=defaults_latent.lora_rank)
    model_args.add_argument("--lora-alpha", type=float, default=defaults_latent.lora_alpha)

    train_args = parser.add_argument_group("training")
    for name in ("stage1_steps", "stage2_steps", "batch_size", "grad_accum", "warmup_steps", "ot_iters",
                 "seed", "save_every", "val_every"):
        train_args.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults_train, name))
    for name in ("lr", "min_lr_ratio", "weight_decay", "max_grad_norm", "ot_weight", "frozen_head_weight",
                 "ot_epsilon", "val_fraction"):
        train_args.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults_train, name))
    train_args.add_argument("--ot-standardize", action="store_true",
                            help="OT cost on per-channel standardised states, so shared and massive channels don't dominate")
    train_args.add_argument("--max-examples", type=int, help="Train on a random subset (pilot runs)")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    store = TeacherStore(args.data_dir)
    latent_config = LatentConfig(
        model=store.meta["model"],
        revision=store.meta["revision"],
        num_latents=args.num_latents,
        num_iterations=args.num_iterations,
        spectral_norm=not args.no_spectral_norm,
        feedback_scale=args.feedback_scale,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    train_config = TrainConfig(
        data_dir=os.path.abspath(args.data_dir),
        out_dir=os.path.abspath(args.out_dir),
        **{name: getattr(args, name) for name in TrainConfig.__dataclass_fields__ if name not in ("data_dir", "out_dir")},
    )

    device = resolve_device(args.device)
    torch.manual_seed(train_config.seed)  # pause-embedding initialisation
    model = LatentReasoner.from_pretrained(latent_config, device)
    tokenizer = AutoTokenizer.from_pretrained(latent_config.model, revision=latent_config.revision)

    os.makedirs(args.out_dir, exist_ok=True)
    started_at = provenance.now()
    with open(os.path.join(args.out_dir, f"manifest-{started_at.replace(':', '')}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "started_at": started_at,
            "command": sys.argv,
            "data_meta": store.meta,
            "device": str(device),
            "dtype": str(model.dtype),
            "git": provenance.git_state(),
            "packages": provenance.package_versions(),
            "host": provenance.host(),
        }, f, indent=2)

    final = train(model, tokenizer, LatentTrainingSet(store, tokenizer), train_config)
    print(f"[train] done: {final}\nEvaluate with:\n  python scripts/eval_baseline.py --backend latent --model {final}")


if __name__ == "__main__":
    main()
