"""Compare two TAEHV autoencoders on videos from a hardcoded data dir.

For each sampled clip, encodes it with the appropriate reference VAE (the one
each model was distilled against), decodes it with both TAEHVs, and writes a
side-by-side MP4: [original | model1 | model2].

Each model can be specified either by a local checkpoint path or by an HF URI
(e.g. "Overworld-Models/taehv-hy1_5-ft"). HF repos are expected to host a flat
TAEHV .pth (as produced by upload_to_hf.py).

Run:
    python -m owl_vaes.utils.compare
"""

import os

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ..configs import Config
from ..data.video_dir_loader import RandomRGBFromMP4s
from ..models.taehv import TAEHV
from ..trainers.taehv_trainer import REF_VAE_SPECS, _get_ref_vae
from ..utils import versatile_load


# ============================== Hardcoded settings ==============================

DATA_SOURCE = "/mnt/data/waypoint_1/owl_control/processed/*/recording.mp4"
TARGET_SIZE = (720, 1280)        # (H, W) — must match what models were trained on
N_FRAMES = 161                    # 4k+1 — works for both wan2.1 and hy1.5
N_VIDEOS = 4                     # number of clips to render
SEED = 42
FPS = 24

MODEL1 = {
    "label": "hy1_5",
    "config": "configs/taehv/hy1_5.yml",
    # Either set "checkpoint" to a local .pt/.pth, or "hf_uri" to a HF repo id.
    # If "hf_uri" is set, "hf_filename" picks the file inside the repo
    # (defaults to the first *.pth found there).
    "hf_uri": "Overworld-Models/taehv-hy1_5-ft",
    # "hf_filename": "taehv1_5.pth",
    # "checkpoint": "checkpoints/taehv_hy1_5/step_30000.pt",
}
MODEL2 = {
    "label": "c512",
    "config": "configs/taehv/taehv_c512_enc.yml",
    "checkpoint": "/mnt/data/laplace/checkpoints/taehv_c512_enc/step_50000.pt",
}

OUTPUT_DIR = "compare_output"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ================================================================================


def resolve_checkpoint(spec):
    """Return a local path for spec['checkpoint'] or spec['hf_uri']."""
    if spec.get("checkpoint"):
        return spec["checkpoint"]
    if spec.get("hf_uri"):
        from huggingface_hub import hf_hub_download, list_repo_files
        repo_id = spec["hf_uri"]
        filename = spec.get("hf_filename")
        if filename is None:
            files = list_repo_files(repo_id)
            pths = [f for f in files if f.endswith(".pth") or f.endswith(".pt")]
            if not pths:
                raise RuntimeError(f"No .pth/.pt found in HF repo {repo_id}: {files}")
            filename = pths[0]
        print(f"  downloading {repo_id}/{filename}")
        return hf_hub_download(repo_id=repo_id, filename=filename)
    raise ValueError(f"Model spec needs 'checkpoint' or 'hf_uri': {spec}")


def build_taehv_from_config(config_path, ckpt_path, device, dtype):
    """Build a TAEHV with architecture from YAML, then load weights from ckpt.
    Accepts both trainer .pt (with ema/model dict) and flat TAEHV .pth — versatile_load
    falls through to the raw dict when neither key is present."""
    cfg = Config.from_yaml(config_path)
    mc = cfg.model
    ref_vae_id = getattr(cfg.train, "ref_vae_id", "wan2.1")

    kwargs = dict(
        checkpoint_path=None,
        latent_channels=REF_VAE_SPECS[ref_vae_id]["latent_channels"],
        patch_size=getattr(mc, "patch_size", 1),
        encoder_channels=getattr(mc, "encoder_channels", 64),
        blocks_per_stage=getattr(mc, "blocks_per_stage", 3),
    )
    if getattr(mc, "decoder_channels", None) is not None:
        kwargs["decoder_channels"] = list(mc.decoder_channels)
    if getattr(mc, "encoder_time_downscale", None) is not None:
        kwargs["encoder_time_downscale"] = tuple(mc.encoder_time_downscale)
    if getattr(mc, "decoder_time_upscale", None) is not None:
        kwargs["decoder_time_upscale"] = tuple(mc.decoder_time_upscale)
    if getattr(mc, "decoder_space_upscale", None) is not None:
        kwargs["decoder_space_upscale"] = tuple(mc.decoder_space_upscale)

    model = TAEHV(**kwargs)
    state = versatile_load(ckpt_path)
    model.load_state_dict(state)
    return model.to(device, dtype).eval(), ref_vae_id


def label_strip(width, label, height=28):
    """Render a single label strip as uint8 [H, W, 3]."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2 - 2), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)


@torch.no_grad()
def encode_decode(ref_vae, models, sample_01):
    """Encode once with `ref_vae`, decode with each TAEHV in `models`. sample_01 in [0,1]."""
    latent = ref_vae.encode(sample_01)
    return [m.decode_video(latent, parallel=True, show_progress_bar=False) for m in models]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model1 ({MODEL1['label']}) from {MODEL1['config']}")
    ckpt1 = resolve_checkpoint(MODEL1)
    model1, ref_id1 = build_taehv_from_config(MODEL1["config"], ckpt1, DEVICE, DTYPE)
    print(f"Loading model2 ({MODEL2['label']}) from {MODEL2['config']}")
    ckpt2 = resolve_checkpoint(MODEL2)
    model2, ref_id2 = build_taehv_from_config(MODEL2["config"], ckpt2, DEVICE, DTYPE)

    print(f"Loading reference VAE ({ref_id1})...")
    ref_vae1 = _get_ref_vae(ref_id1, DTYPE).to(DEVICE).eval()
    if hasattr(ref_vae1, "set_tiling"):
        # Tiling lets us encode the longer (sample-time) clip without OOM. Trainer
        # warns this introduces seams in the latent, but for visual comparison
        # this is the same trade-off it makes during sample logging.
        ref_vae1.set_tiling(True)

    if ref_id2 == ref_id1:
        ref_vae2 = ref_vae1
    else:
        print(f"Loading reference VAE ({ref_id2})...")
        ref_vae2 = _get_ref_vae(ref_id2, DTYPE).to(DEVICE).eval()
        if hasattr(ref_vae2, "set_tiling"):
            ref_vae2.set_tiling(True)

    print(f"Reading clips from {DATA_SOURCE}")
    reader = RandomRGBFromMP4s(
        DATA_SOURCE,
        seed=SEED,
        target_size=TARGET_SIZE,
        window_length=N_FRAMES,
        suppress_warnings=False,
    )
    reader_iter = iter(reader)

    H, W = TARGET_SIZE
    header = np.concatenate(
        [label_strip(W, "original"), label_strip(W, MODEL1["label"]), label_strip(W, MODEL2["label"])],
        axis=1,
    )

    for i in range(N_VIDEOS):
        print(f"[{i + 1}/{N_VIDEOS}] sampling clip...")
        clip_np = next(reader_iter)  # [T, H, W, C] uint8

        clip = torch.from_numpy(clip_np).permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
        clip = clip.to(DEVICE, DTYPE).div_(255.0).unsqueeze(0)              # [1, T, C, H, W] in [0,1]

        if ref_vae2 is ref_vae1:
            dec1, dec2 = encode_decode(ref_vae1, [model1, model2], clip)
        else:
            (dec1,) = encode_decode(ref_vae1, [model1], clip)
            (dec2,) = encode_decode(ref_vae2, [model2], clip)

        # Align temporally: each model trims `frames_to_trim` startup frames from the head.
        off1 = clip.shape[1] - dec1.shape[1]
        off2 = clip.shape[1] - dec2.shape[1]
        off = max(off1, off2)
        orig = clip[:, off:]
        dec1 = dec1[:, off - off1:]
        dec2 = dec2[:, off - off2:]
        T = min(orig.shape[1], dec1.shape[1], dec2.shape[1])
        orig, dec1, dec2 = orig[:, :T], dec1[:, :T], dec2[:, :T]

        sbs = torch.cat(
            [orig.clamp(0, 1), dec1.clamp(0, 1), dec2.clamp(0, 1)], dim=-1
        )  # [1, T, C, H, 3W]
        sbs = sbs[0].mul(255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()  # [T, H, 3W, C]

        # Stamp the labels on every frame so the output is unambiguous.
        sbs = np.concatenate([np.broadcast_to(header, (sbs.shape[0], *header.shape)), sbs], axis=1)

        out_path = os.path.join(
            OUTPUT_DIR,
            f"compare_{i:03d}_{MODEL1['label']}_vs_{MODEL2['label']}.mp4",
        )
        iio.imwrite(out_path, sbs, fps=FPS, codec="libx264", quality=8)
        print(f"  wrote {out_path}  ({sbs.shape[0]} frames @ {sbs.shape[2]}x{sbs.shape[1]})")

    print("Done.")


if __name__ == "__main__":
    main()
