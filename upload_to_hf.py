"""Convert a trainer .pt checkpoint to a TAEHV-compatible .pth and upload to HF.

Training saves a dict {model, ema, opt, scaler, steps, ...} where model/ema keys carry
DDP ("module.") and EMA ("ema_model." / "ema_model.module.") prefixes. Upstream TAEHV
expects a flat state_dict matching its own module layout. This script strips those
prefixes, verifies the state dict loads into TAEHV, writes a .pth, and (optionally)
uploads to a Hugging Face repo.

Usage:
    # local only
    python upload_to_hf.py --ckpt checkpoints/taehv_hy1_5/step_30000.pt \
                          --out  checkpoints/pretrained/taehv_hy1_5_ft.pth

    # also upload
    python upload_to_hf.py --ckpt checkpoints/taehv_hy1_5/step_30000.pt \
                          --out  checkpoints/pretrained/taehv_hy1_5_ft.pth \
                          --hf-repo your-username/taehv-hy1_5-ft \
                          --path-in-repo taehv_hy1_5_ft.pth

Auth for HF: set HF_TOKEN in the environment, or pre-run `huggingface-cli login`.
"""

import argparse
import os
from pathlib import Path

import torch

from owl_vaes.models.taehv import TAEHV


# Arch presets keyed by ref_vae_id (mirrors REF_VAE_SPECS in the trainer).
# These match the upstream madebyollin/taehv conventions.
ARCH_PRESETS = {
    "wan2.1": dict(patch_size=1, latent_channels=16),
    "hy1.5":  dict(patch_size=2, latent_channels=32),
}


def _strip_prefix(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def extract_taehv_state_dict(ckpt: dict, use_ema: bool) -> dict:
    """Pull a clean TAEHV state dict out of a training checkpoint.

    Handles four shapes:
      - ema   under DDP: 'ema_model.module.<name>'
      - ema   no DDP:    'ema_model.<name>'
      - model under DDP: 'module.<name>'
      - model no DDP:    '<name>'
    """
    if use_ema:
        if "ema" not in ckpt:
            raise KeyError("use_ema=True but checkpoint has no 'ema' key")
        sd = ckpt["ema"]
        for prefix in ("ema_model.module.", "ema_model."):
            out = _strip_prefix(sd, prefix)
            if out:
                return out
        raise RuntimeError("EMA state_dict had neither ema_model. nor ema_model.module. prefix")
    sd = ckpt["model"]
    if all(k.startswith("module.") for k in sd):
        return _strip_prefix(sd, "module.")
    return dict(sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to step_<N>.pt")
    ap.add_argument("--out", type=Path, required=True, help="Output .pth path")
    ap.add_argument("--arch", choices=sorted(ARCH_PRESETS), default="hy1.5",
                    help="Architecture preset matching the training config")
    ap.add_argument("--encoder-channels", type=int, default=64)
    ap.add_argument("--decoder-channels", type=int, nargs=4, default=[256, 128, 64, 64])
    ap.add_argument("--blocks-per-stage", type=int, default=3)
    ap.add_argument("--no-ema", action="store_true", help="Export raw model instead of EMA")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16",
                    help="Weight dtype for the exported .pth (upstream uses fp16)")
    ap.add_argument("--hf-repo", type=str, default=None,
                    help="Hugging Face repo id (e.g. user/repo). Skipped if unset.")
    ap.add_argument("--path-in-repo", type=str, default=None,
                    help="Path within the HF repo (defaults to basename of --out)")
    ap.add_argument("--private", action="store_true",
                    help="Create repo as private if it doesn't exist yet")
    args = ap.parse_args()

    # 1. Load trainer checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = extract_taehv_state_dict(ckpt, use_ema=not args.no_ema)
    print(f"[extract] {len(sd)} tensors, source={'model' if args.no_ema else 'ema'}")

    # 2. Verify it loads cleanly into TAEHV at the specified arch
    preset = ARCH_PRESETS[args.arch]
    model = TAEHV(
        checkpoint_path=None,
        patch_size=preset["patch_size"],
        latent_channels=preset["latent_channels"],
        encoder_channels=args.encoder_channels,
        decoder_channels=list(args.decoder_channels),
        blocks_per_stage=args.blocks_per_stage,
        encoder_time_downscale=(True, True, False),
        decoder_time_upscale=(False, True, True),
        decoder_space_upscale=(True, True, True),
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if missing: print("  first missing:", missing[:3])
        if unexpected: print("  first unexpected:", unexpected[:3])
        raise RuntimeError("State dict does not match TAEHV arch; pass the right --encoder-channels / --decoder-channels / --blocks-per-stage")
    print(f"[verify] loaded into TAEHV ({sum(p.numel() for p in model.parameters()):,} params)")

    # 3. Cast and save .pth
    cast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    sd = {k: v.to(cast_dtype) if v.is_floating_point() else v for k, v in sd.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, args.out)
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"[save] {args.out}  ({size_mb:.1f} MB, dtype={args.dtype})")

    # 4. Optional HF upload
    if args.hf_repo is None:
        return
    from dotenv import load_dotenv
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set (expected in env or .env file)")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(args.hf_repo, private=args.private, exist_ok=True)
    path_in_repo = args.path_in_repo or args.out.name
    api.upload_file(
        path_or_fileobj=str(args.out),
        path_in_repo=path_in_repo,
        repo_id=args.hf_repo,
        commit_message=f"Upload {args.out.name} (from {args.ckpt.name})",
    )
    print(f"[upload] https://huggingface.co/{args.hf_repo}/blob/main/{path_in_repo}")


if __name__ == "__main__":
    main()
