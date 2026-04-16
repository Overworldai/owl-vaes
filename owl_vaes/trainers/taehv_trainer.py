"""
Trainer for distilling a Tiny AutoEncoder (TAEHV) from a reference video VAE.

Trains encoder and/or decoder of a TAEHV model to match the latent space of a
teacher VAE (e.g. WAN 2.1), optionally using Seraena adversarial corrections
for the decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data import get_loader
from ..models.taehv import TAEHV
from ..models.seraena import Seraena
from ..schedulers import get_scheduler_cls
from ..utils import Timer, freeze
from ..utils.logging import LogHelper, to_wandb_video_sidebyside
from .base import BaseTrainer


def _get_ref_vae(ref_vae_id, ref_dtype):
    """Load a reference (teacher) video VAE for latent distillation."""
    if ref_vae_id == "wan2.1":
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=ref_dtype
        )
        latent_channels = 16
        time_downscale = 4
        space_downscale = 8
    else:
        raise ValueError(f"Unknown ref_vae_id: {ref_vae_id}")

    class RefVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = vae
            self.latent_channels = latent_channels
            self.time_downscale = time_downscale
            self.space_downscale = space_downscale

        @torch.no_grad()
        def encode(self, x):
            """Encode NTCHW [0,1] RGB -> NTCHW latents."""
            # diffusers expects BCTHW with [-1,1] range
            y = x.transpose(1, 2).to(ref_dtype).mul(2).sub_(1)
            y = self.vae.encode(y).latent_dist.sample()
            return y.transpose(1, 2).to(x.dtype)

    return RefVAE()


class TAEHVTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Build TAEHV from model config
        mc = self.model_cfg
        taehv_kwargs = {}
        taehv_kwargs["checkpoint_path"] = getattr(mc, "checkpoint_path", None)
        taehv_kwargs["latent_channels"] = getattr(mc, "latent_channels", 16)
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
        # Initialize decoder output bias to 0.5 (as in the reference notebook)
        nn.init.constant_(self.model.decoder[-1].bias, 0.5)

        self.train_encoder = getattr(self.train_cfg, "train_encoder", True)
        self.train_decoder = getattr(self.train_cfg, "train_decoder", True)
        self.n_frames = getattr(self.train_cfg, "n_frames", 12)
        self.n_seraena_frames = getattr(self.train_cfg, "n_seraena_frames", 3)

        # Reference VAE id
        self.ref_vae_id = getattr(self.train_cfg, "ref_vae_id", "wan2.1")
        self.ref_dtype = torch.bfloat16

        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"TAEHV parameters: {n_params:,}")

        self.ema = None
        self.opt = None
        self.scheduler = None
        self.scaler = None
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
        super().save(save_dict)

    def load(self):
        if not hasattr(self.train_cfg, "resume_ckpt") or self.train_cfg.resume_ckpt is None:
            return
        save_dict = super().load(self.train_cfg.resume_ckpt)
        self.model.load_state_dict(save_dict["model"])
        self.ema.load_state_dict(save_dict["ema"])
        self.opt.load_state_dict(save_dict["opt"])
        self.scaler.load_state_dict(save_dict["scaler"])
        if self.scheduler is not None and "scheduler" in save_dict:
            self.scheduler.load_state_dict(save_dict["scheduler"])
        self.total_step_counter = save_dict["steps"]

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
            self.model = DDP(self.model, device_ids=[self.local_rank])

        # EMA
        self.ema = EMA(self.model, beta=0.995, update_after_step=0, update_every=1)

        # Seraena (adversarial corrector) for decoder training
        seraena = None
        if self.train_decoder and gan_weight > 0.0:
            seraena = Seraena(
                3 * self.n_seraena_frames,
                ref_vae.latent_channels,
            ).to(self.device)
            if self.rank == 0:
                s_params = sum(p.numel() for p in seraena.parameters())
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

        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast(self.device, torch.bfloat16)

        self.load()

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

        def pad_and_group(x):
            """Group decoded frames into chunks for Seraena."""
            n, t, c, h, w = x.shape
            x = torch.cat([x, x[:, :frames_to_trim]], 1)
            n, t2, c, h, w = x.shape
            return x.reshape(n * t2 // self.n_seraena_frames, self.n_seraena_frames * c, h, w)

        def ungroup_and_unpad(x):
            _, _, h, w = x.shape
            x = x.reshape(-1, self.n_frames, 3, h, w)
            return x[:, :-frames_to_trim]

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
                        encoded = self.model.encode_video(ims, parallel=True, show_progress_bar=False)
                        enc_loss = F.mse_loss(encoded, ref_latent) / accum_steps
                        losses["encoder"] = enc_loss * l2_weight
                        metrics.log("enc_l2", enc_loss)

                    if self.train_decoder:
                        decoded = self.model.decode_video(ref_latent, parallel=True, show_progress_bar=False)
                        ims_target = ims[:, :-frames_to_trim]
                        rec_loss = F.mse_loss(decoded, ims_target) / accum_steps
                        losses["dec_rec"] = rec_loss * l2_weight
                        metrics.log("dec_l2", rec_loss)

                        if seraena is not None and gan_weight > 0.0:
                            with torch.no_grad():
                                grouped_real = pad_and_group(ims_target)
                                grouped_fake = pad_and_group(decoded.detach())
                                # Time-average latents for Seraena context
                                n_groups = self.n_frames // self.n_seraena_frames
                                lat_ctx = ref_latent.mean(1, keepdim=True).repeat_interleave(n_groups, dim=1).flatten(0, 1)
                                target, _ = seraena.step_and_make_correction_targets(grouped_real, grouped_fake, lat_ctx)
                                target = ungroup_and_unpad(target)

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
                                if self.world_size > 1:
                                    ema_model = ema_model.module
                                ema_dec = ema_model.decode_video(ref_latent, parallel=True, show_progress_bar=False)
                            # Convert [0,1] back to [-1,1] for wandb logging
                            ims_log = ims[:, :-frames_to_trim].mul(2).sub(1)
                            dec_log = ema_dec.clamp(0, 1).mul(2).sub(1)
                            wandb_dict["samples"] = to_wandb_video_sidebyside(
                                ims_log.detach().contiguous().bfloat16(),
                                dec_log.detach().contiguous().bfloat16(),
                            )

                        if self.rank == 0 and self.logging_cfg is not None:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()
