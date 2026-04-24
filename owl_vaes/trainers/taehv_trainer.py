"""
Trainer for distilling a Tiny AutoEncoder (TAEHV) from a reference video VAE.

Trains encoder and/or decoder of a TAEHV model to match the latent space of a
teacher VAE (e.g. WAN 2.1), optionally using Seraena adversarial corrections
for the decoder.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data import get_loader
from ..data.video_dir_loader import RandomRGBFromMP4s
from ..models.taehv import TAEHV
from ..models.seraena import Seraena
from ..schedulers import get_scheduler_cls
from ..utils import Timer, freeze
from ..utils.logging import LogHelper, to_wandb_video_sidebyside
from .base import BaseTrainer


REF_VAE_SPECS = {
    "wan2.1": {"latent_channels": 16, "time_downscale": 4, "space_downscale": 8},
    "hy1.5":  {"latent_channels": 32, "time_downscale": 4, "space_downscale": 16},
}


def _get_ref_vae(ref_vae_id, ref_dtype):
    """Load a reference (teacher) video VAE for latent distillation."""
    if ref_vae_id not in REF_VAE_SPECS:
        raise ValueError(f"Unknown ref_vae_id: {ref_vae_id}")
    spec = REF_VAE_SPECS[ref_vae_id]
    latent_channels = spec["latent_channels"]
    time_downscale = spec["time_downscale"]
    space_downscale = spec["space_downscale"]

    if ref_vae_id == "wan2.1":
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=ref_dtype
        )
    elif ref_vae_id == "hy1.5":
        from diffusers import AutoencoderKLHunyuanVideo15
        # Patch diffusers 0.37.1 bug: prepare_causal_attention_mask returns a 3D
        # tensor but the attention forward unsqueezes QKV to 4D, so SDPA needs
        # a 4D mask. Inject unsqueeze(1) once, idempotently.
        from diffusers.models.autoencoders import autoencoder_kl_hunyuanvideo15 as _hv15
        _attn_cls = _hv15.HunyuanVideo15AttnBlock
        if not getattr(_attn_cls, "_mask_patched", False):
            _orig = _attn_cls.prepare_causal_attention_mask
            @staticmethod
            def _patched(*args, **kwargs):
                m = _orig(*args, **kwargs)
                return m.unsqueeze(1) if m.ndim == 3 else m
            _attn_cls.prepare_causal_attention_mask = _patched
            _attn_cls._mask_patched = True
        vae = AutoencoderKLHunyuanVideo15.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
            subfolder="vae",
            torch_dtype=ref_dtype,
        )
        # Tiling stays OFF by default — encode-tiling introduces seams in latents that
        # corrupt the distillation target. Turn it on only around long-frame sample encodes.

    class RefVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = vae
            self.latent_channels = latent_channels
            self.time_downscale = time_downscale
            self.space_downscale = space_downscale

        @torch.no_grad()
        def encode(self, x):
            """Encode NTCHW [0,1] RGB -> NTCHW latents (using mean for deterministic targets)."""
            # diffusers expects BCTHW with [-1,1] range
            y = x.transpose(1, 2).to(ref_dtype).mul(2).sub_(1)
            y = self.vae.encode(y).latent_dist.mode()
            return y.transpose(1, 2).to(x.dtype)

        def set_tiling(self, enabled: bool):
            """Toggle VAE spatial tiling + batch slicing. Use only for long sample clips."""
            self.vae.use_tiling = enabled
            self.vae.use_slicing = enabled

    return RefVAE()


class TAEHVTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Reference VAE id (determines latent_channels — must match ref VAE)
        self.ref_vae_id = getattr(self.train_cfg, "ref_vae_id", "wan2.1")
        if self.ref_vae_id not in REF_VAE_SPECS:
            raise ValueError(f"Unknown ref_vae_id: {self.ref_vae_id}")

        # Build TAEHV from model config
        mc = self.model_cfg
        taehv_kwargs = {}
        taehv_kwargs["checkpoint_path"] = getattr(mc, "checkpoint_path", None)
        taehv_kwargs["latent_channels"] = REF_VAE_SPECS[self.ref_vae_id]["latent_channels"]
        taehv_kwargs["patch_size"] = getattr(mc, "patch_size", 1)
        taehv_kwargs["encoder_channels"] = getattr(mc, "encoder_channels", 64)
        taehv_kwargs["blocks_per_stage"] = getattr(mc, "blocks_per_stage", 3)
        if getattr(mc, "decoder_channels", None) is not None:
            taehv_kwargs["decoder_channels"] = list(mc.decoder_channels)
        if getattr(mc, "encoder_time_downscale", None) is not None:
            taehv_kwargs["encoder_time_downscale"] = tuple(mc.encoder_time_downscale)
        if getattr(mc, "decoder_time_upscale", None) is not None:
            taehv_kwargs["decoder_time_upscale"] = tuple(mc.decoder_time_upscale)
        if getattr(mc, "decoder_space_upscale", None) is not None:
            taehv_kwargs["decoder_space_upscale"] = tuple(mc.decoder_space_upscale)

        self.model = TAEHV(**taehv_kwargs)
        # Only initialize decoder output bias to 0.5 when training from scratch;
        # when loading a pretrained checkpoint, keep its learned bias.
        if taehv_kwargs["checkpoint_path"] is None:
            nn.init.constant_(self.model.decoder[-1].bias, 0.5)

        self.train_encoder = getattr(self.train_cfg, "train_encoder", True)
        self.train_decoder = getattr(self.train_cfg, "train_decoder", True)
        self.n_frames = getattr(self.train_cfg, "n_frames", 12)
        self.n_seraena_frames = getattr(self.train_cfg, "n_seraena_frames", 3)

        self.ref_dtype = torch.bfloat16

        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"TAEHV parameters: {n_params:,}")

        self.ema = None
        self.opt = None
        self.scheduler = None
        self.scaler = None
        self.seraena = None
        self.total_step_counter = 0

    def save(self):
        save_dict = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "steps": self.total_step_counter,
        }
        if self.scheduler is not None:
            save_dict["scheduler"] = self.scheduler.state_dict()
        if self.seraena is not None:
            save_dict["seraena"] = {
                "model": self.seraena.state_dict(),
                "opt": self.seraena.opt.state_dict(),
                "scaler": self.seraena.scaler.state_dict(),
                "buff": self.seraena.buff,
            }
        super().save(save_dict)

    def load(self):
        if not hasattr(self.train_cfg, "resume_ckpt") or self.train_cfg.resume_ckpt is None:
            return {"resumed": False, "seraena_loaded": False}
        save_dict = super().load(self.train_cfg.resume_ckpt)
        self.model.load_state_dict(save_dict["model"])
        self.ema.load_state_dict(save_dict["ema"])
        self.opt.load_state_dict(save_dict["opt"])
        self.scaler.load_state_dict(save_dict["scaler"])
        if self.scheduler is not None and "scheduler" in save_dict:
            self.scheduler.load_state_dict(save_dict["scheduler"])
        self.total_step_counter = save_dict["steps"]
        seraena_loaded = False
        if self.seraena is not None and "seraena" in save_dict:
            s = save_dict["seraena"]
            self.seraena.load_state_dict(s["model"])
            self.seraena.opt.load_state_dict(s["opt"])
            self.seraena.scaler.load_state_dict(s["scaler"])
            self.seraena.buff = s["buff"]
            seraena_loaded = True
        return {"resumed": True, "seraena_loaded": seraena_loaded}

    def train(self):
        if "cuda" in self.device:
            torch.cuda.set_device(self.local_rank)

        # Loss weights
        l2_weight = self.train_cfg.loss_weights.get("l2", 1.0)
        gan_weight = self.train_cfg.loss_weights.get("gan", 1.0)

        # Load reference VAE
        if self.rank == 0:
            print(f"Loading reference VAE ({self.ref_vae_id})...")
        ref_vae = _get_ref_vae(self.ref_vae_id, self.ref_dtype).to(self.device).eval()
        freeze(ref_vae)
        if self.rank == 0:
            print("Reference VAE loaded.")

        # Move model to device
        self.model = self.model.to(self.device).train()
        if self.world_size > 1:
            # find_unused_parameters=True because train_encoder=False leaves
            # encoder params out of the backward graph; without this, DDP's reducer
            # can mishandle the unused params.
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=True,
            )

        # EMA wraps the inner (un-DDP'd) module so deepcopy doesn't pull in
        # DDP/ProcessGroup state.
        inner_model = self.model.module if self.world_size > 1 else self.model
        self.ema = EMA(inner_model, beta=0.995, update_after_step=0, update_every=1)

        # Seraena (adversarial corrector) for decoder training
        if self.train_decoder and gan_weight > 0.0:
            self.seraena = Seraena(
                3 * self.n_seraena_frames,
                ref_vae.latent_channels,
                max_buff_len=256,
            ).to(self.device)
            if self.rank == 0:
                s_params = sum(p.numel() for p in self.seraena.parameters())
                print(f"Seraena parameters: {s_params:,}")

        # Optimizer
        opt_cls = getattr(torch.optim, self.train_cfg.opt)
        self.opt = opt_cls(self.model.parameters(), **self.train_cfg.opt_kwargs)

        # Scheduler
        if getattr(self.train_cfg, "scheduler", None) is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(
                self.opt, **self.train_cfg.scheduler_kwargs
            )

        # Grad accum
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)

        # bf16 has plenty of range — no loss-scaling needed. A per-rank enabled
        # GradScaler will diverge in DDP: if one rank's initial unscaled grad hits
        # inf and another rank's doesn't, scalers drift, skip-patterns diverge,
        # some ranks get stuck perpetually skipping while others train. Disable.
        self.scaler = torch.amp.GradScaler(enabled=False)
        ctx = torch.amp.autocast(self.device, torch.bfloat16)

        load_status = self.load()

        timer = Timer()
        timer.reset()
        metrics = LogHelper()

        if self.rank == 0 and self.logging_cfg is not None:
            wandb.watch(self.get_module(), log="all")

        # Data loader — yields NTCHW bf16 in [-1,1]
        data_kwargs = dict(self.train_cfg.data_kwargs) if hasattr(self.train_cfg, "data_kwargs") else {}
        data_kwargs["window_length"] = self.n_frames
        loader = get_loader(
            self.train_cfg.data_id,
            self.train_cfg.batch_size,
            rank=self.rank,
            world_size=self.world_size,
            **data_kwargs,
        )

        frames_to_trim = self.get_module().frames_to_trim

        # Optional dedicated sample-time reader, always bs=1 to save VRAM (runs in
        # main process, reuses already-resolved paths). When set, sampling uses this
        # path even if sample_n_frames == n_frames.
        sample_n_frames = getattr(self.train_cfg, "sample_n_frames", None)
        sample_reader_iter = None
        if sample_n_frames is not None:
            sample_reader = RandomRGBFromMP4s(
                None,
                seed=self.rank + 12345,
                target_size=data_kwargs.get("target_size", (360, 640)),
                window_length=sample_n_frames,
                suppress_warnings=True,
                _resolved_paths=loader.dataset.paths,
            )
            sample_reader_iter = iter(sample_reader)

        def pad_and_group(x):
            """Group frames into chunks of n_seraena_frames. Pads (cyclically from start)
            to the next multiple of n_seraena_frames so no frames are silently dropped.
            Returns (grouped_tensor, t_padded, t_orig)."""
            n, t, c, h, w = x.shape
            pad_n = (-t) % self.n_seraena_frames
            if pad_n > 0:
                x = torch.cat([x, x[:, :pad_n]], 1)
            t2 = x.shape[1]
            return (
                x.reshape(n * t2 // self.n_seraena_frames, self.n_seraena_frames * c, h, w),
                t2,
                t,
            )

        def ungroup_and_unpad(x, t_padded, t_orig):
            _, _, h, w = x.shape
            n = x.shape[0] * self.n_seraena_frames // t_padded
            x = x.reshape(n, t_padded, 3, h, w)
            return x[:, :t_orig]

        # Warm up Seraena's critic when resuming a checkpoint that predates
        # Seraena persistence. Otherwise the fresh critic + empty replay buffer
        # produce correction targets ≈ fakes, so gan_loss sits at ~0 for a long
        # time while the decoder (already converged) doesn't move.
        warmup_steps = getattr(self.train_cfg, "seraena_warmup_steps", 0)
        if (
            self.seraena is not None
            and load_status["resumed"]
            and not load_status["seraena_loaded"]
            and warmup_steps > 0
        ):
            if self.rank == 0:
                print(f"Warming up Seraena critic for {warmup_steps} steps...")
            warmup_iter = iter(loader)
            for wi in range(warmup_steps):
                try:
                    batch = next(warmup_iter)
                except StopIteration:
                    warmup_iter = iter(loader)
                    batch = next(warmup_iter)
                batch = batch.to(self.device)
                ims = batch.mul(0.5).add_(0.5)
                with ctx, torch.no_grad():
                    ref_latent = ref_vae.encode(ims)
                    decoded = self.get_module().decode_video(
                        ref_latent, parallel=True, show_progress_bar=False
                    )
                    offset = ims.shape[1] - decoded.shape[1]
                    ims_target = ims[:, offset:]
                    grouped_real, t_padded, t_orig = pad_and_group(ims_target)
                    grouped_fake, _, _ = pad_and_group(decoded)
                    n_groups = t_padded // self.n_seraena_frames
                    lat_ctx = ref_latent.mean(1, keepdim=True).repeat_interleave(n_groups, dim=1).flatten(0, 1)
                # _disc_train_step: trains disc + fills replay buffer. Skips the
                # correction backward we'd otherwise discard.
                with ctx:
                    debug = self.seraena._disc_train_step(grouped_real, grouped_fake, lat_ctx)
                if self.rank == 0 and (wi + 1) % max(1, warmup_steps // 10) == 0:
                    print(f"  [seraena warmup] step {wi+1}/{warmup_steps}  disc_loss={debug['disc_loss']:.4f}")
            self.barrier()

        local_step = 0

        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                total_loss = 0.0
                # batch is [B, T, C, H, W] bf16 in [-1,1]
                batch = batch.to(self.device)
                # Convert from [-1,1] to [0,1] for TAEHV (it expects [0,1])
                ims = batch.mul(0.5).add_(0.5)

                with ctx:
                    with torch.no_grad():
                        ref_latent = ref_vae.encode(ims)

                    losses = {}

                    if self.train_encoder:
                        encoded = self.get_module().encode_video(ims, parallel=True, show_progress_bar=False)
                        enc_loss = F.mse_loss(encoded, ref_latent) / accum_steps
                        losses["encoder"] = enc_loss * l2_weight
                        metrics.log("enc_l2", enc_loss)

                    if self.train_decoder:
                        decoded = self.get_module().decode_video(ref_latent, parallel=True, show_progress_bar=False)
                        # Temporal alignment between TAEHV output and ims:
                        # The offset is whatever is needed to match decoded's length to the
                        # tail of ims. Works for both conventions:
                        #   - Wan 2.1 (4k → k latents): decoded shorter by frames_to_trim → offset=3
                        #   - HY 1.5 (4k+1 → k+1): decoded same length as ims → offset=0
                        offset = ims.shape[1] - decoded.shape[1]
                        ims_target = ims[:, offset:]
                        rec_loss = F.mse_loss(decoded, ims_target) / accum_steps
                        losses["dec_rec"] = rec_loss * l2_weight
                        metrics.log("dec_l2", rec_loss)

                        if self.seraena is not None and gan_weight > 0.0:
                            with torch.no_grad():
                                grouped_real, t_padded, t_orig = pad_and_group(ims_target)
                                grouped_fake, _, _ = pad_and_group(decoded.detach())
                                # Time-average latents for Seraena context — one ctx per group
                                n_groups = t_padded // self.n_seraena_frames
                                lat_ctx = ref_latent.mean(1, keepdim=True).repeat_interleave(n_groups, dim=1).flatten(0, 1)
                            target, _ = self.seraena.step_and_make_correction_targets(grouped_real, grouped_fake, lat_ctx)
                            target = ungroup_and_unpad(target, t_padded, t_orig)

                            gan_loss = F.mse_loss(decoded, target) / accum_steps
                            losses["dec_gan"] = gan_loss * gan_weight
                            metrics.log("dec_gan", gan_loss)

                    total_loss = sum(losses.values())

                self.scaler.scale(total_loss).backward()

                local_step += 1
                if local_step % accum_steps == 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.opt)
                    self.opt.zero_grad(set_to_none=True)
                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.ema.update()

                    with torch.no_grad():
                        wandb_dict = metrics.pop()
                        wandb_dict["time"] = timer.hit()
                        wandb_dict["lr"] = self.opt.param_groups[0]["lr"]
                        timer.reset()

                        if self.total_step_counter % self.train_cfg.sample_interval == 0:
                            with ctx:
                                ema_model = self.ema.ema_model
                                if sample_reader_iter is not None:
                                    sample_np = next(sample_reader_iter)  # [T, H, W, C] uint8
                                    sample_batch = torch.from_numpy(sample_np).permute(0, 3, 1, 2).contiguous()
                                    sample_batch = sample_batch.bfloat16().div_(127.5).sub_(1.0).unsqueeze(0).to(self.device)
                                    sample_ims = sample_batch.mul(0.5).add_(0.5)
                                    # Tiling introduces latent seams; only use it here to fit the long sample clip.
                                    if hasattr(ref_vae, "set_tiling"):
                                        ref_vae.set_tiling(True)
                                    try:
                                        sample_latent = ref_vae.encode(sample_ims)
                                    finally:
                                        if hasattr(ref_vae, "set_tiling"):
                                            ref_vae.set_tiling(False)
                                    ema_dec = ema_model.decode_video(sample_latent, parallel=True, show_progress_bar=False)
                                    offset_s = sample_ims.shape[1] - ema_dec.shape[1]
                                    ima_log_src = sample_ims[:, offset_s:]
                                else:
                                    ema_dec = ema_model.decode_video(ref_latent, parallel=True, show_progress_bar=False)
                                    offset_s = ims.shape[1] - ema_dec.shape[1]
                                    ima_log_src = ims[:, offset_s:]
                                ims_log = ima_log_src.mul(2).sub(1)
                            # Convert [0,1] back to [-1,1] for wandb logging
                            dec_log = ema_dec.clamp(0, 1).mul(2).sub(1)

                            # Gather samples from all ranks so rank 0 logs one video per GPU
                            ims_log = ims_log.detach().contiguous().bfloat16()
                            dec_log = dec_log.detach().contiguous().bfloat16()
                            if self.world_size > 1 and dist.is_initialized():
                                g_ims = [torch.empty_like(ims_log) for _ in range(self.world_size)]
                                g_dec = [torch.empty_like(dec_log) for _ in range(self.world_size)]
                                dist.all_gather(g_ims, ims_log)
                                dist.all_gather(g_dec, dec_log)
                                if self.rank == 0:
                                    ims_log = torch.cat(g_ims, dim=0)
                                    dec_log = torch.cat(g_dec, dim=0)

                            if self.rank == 0:
                                wandb_dict["samples"] = to_wandb_video_sidebyside(ims_log, dec_log)

                        if self.rank == 0 and self.logging_cfg is not None:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()
