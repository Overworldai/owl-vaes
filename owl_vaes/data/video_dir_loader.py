import os, glob, random
from pathlib import Path
from fractions import Fraction
import numpy as np
import av

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import torchvision.transforms.functional as TF


class RandomRGBFromMP4s:
    """
    Continuous iterator yielding random RGB frames (H,W,3 uint8) from MP4s.
    - Uniform over files; uniform over time within each file.
    - No persistent handles: each yield opens the chosen file once and closes it.
    - Duration/fps are computed lazily on first use and cached in memory.
    """
    def __init__(self, source, seed=None, target_size=(360,640), window_length=1, suppress_warnings = True, _resolved_paths=None):
        self.target_size = target_size  # (H, W)
        self.window_length = window_length
        # Use pre-resolved paths if provided, otherwise resolve from source
        if _resolved_paths is not None:
            self.paths = _resolved_paths
        else:
            if isinstance(source, str):
                source = [source]
            self.paths = self._find_mp4s(source)

        if not self.paths:
            raise RuntimeError("No videos found in the supplied source.")
        self.rng = random.Random(seed)
        self.meta = {}  # path -> (duration_s, fps, eps_end_s)
        self.suppress_warnings = suppress_warnings

    @staticmethod
    def _find_mp4s(spec):
        """Return sorted unique .mp4 Paths from globs/dirs/files (abs/rel OK)."""
        specs = [spec] if isinstance(spec, (str, Path)) else list(spec)
        out = []
        for s in specs:
            s = os.path.expanduser(str(s))
            p = Path(s)
            if p.exists() and p.is_dir():
                out += glob.glob(str(p / "**/*.mp4"), recursive=True)
            elif p.exists() and p.is_file() and p.suffix.lower() == ".mp4":
                out.append(str(p))
            else:
                # treat as (possibly absolute) glob pattern
                out += glob.glob(s, recursive=True)
        return [Path(x) for x in sorted({x for x in out if x.lower().endswith(".mp4")})]

    def __iter__(self):
        return self

    def __next__(self):
        max_attempts = 10  # Try up to 10 different videos before giving up
        for attempt in range(max_attempts):
            try:
                p = self.paths[self.rng.randrange(len(self.paths))]
                # If we already know (dur, fps, eps), use it; otherwise compute inside the same open.
                if p in self.meta:
                    dur, fps, eps = self.meta[p]
                    # Leave enough space at the end for window_length frames
                    max_t = max(0.0, dur - eps * self.window_length)
                    t = self.rng.random() * max_t
                    frames = self._decode_window_at_time(p, t, fps)
                else:
                    # First time for this file: open once, read metadata, pick t, decode, close.
                    with av.open(str(p)) as c:
                        v = next(s for s in c.streams if s.type == "video")
                        fps = float(v.average_rate) if v.average_rate else 30.0
                        dur = (c.duration / 1e6) if c.duration is not None else (
                            (float(v.frames) / fps) if (v.frames and fps) else 600.0
                        )
                        eps = 1.0 / max(1.0, fps)
                        self.meta[p] = (float(dur), float(fps), float(eps))
                        max_t = max(0.0, dur - eps * self.window_length)
                        t = self.rng.random() * max_t
                        frames = self._decode_window_from_open(c, v, t, fps)  # decode using this same open

                # Resize frames if needed and return
                if self.window_length == 1:
                    return self._resize_if_needed(frames[0])
                else:
                    return np.stack([self._resize_if_needed(f) for f in frames], axis=0)

            except Exception as e:
                if not self.suppress_warnings:
                    print(f"Error decoding {p}: {e}. Trying another video...")
                # Remove failed video from meta cache if it exists
                if p in self.meta:
                    del self.meta[p]
                continue

        raise RuntimeError(f"Failed to decode a video after {max_attempts} attempts")

    @staticmethod
    def _decode_from_open(
            container: av.container.input.InputContainer,
            vstream: av.video.stream.VideoStream,
            t_sec: float
    ) -> np.ndarray:
        """Seek+decode within an already-open container; returns RGB24 (H,W,3)."""
        tb: Fraction = vstream.time_base
        # Light speed knobs (don’t persist beyond this call anyway)
        try:
            vstream.thread_type = "FRAME"
            vstream.codec_context.thread_count = 1
            vstream.codec_context.skip_loop_filter = "ALL"
        except Exception:
            pass
        # Clamp t to known duration if available
        if container.duration is not None:
            t_sec = min(max(0.0, t_sec), max(0.0, container.duration / 1e6 - 1e-3))
        # Keyframe seek then decode forward to >= t
        try:
            vstream.codec_context.skip_frame = "NONKEY"
            container.seek(int(max(0.0, t_sec) / float(tb)), stream=vstream, backward=True, any_frame=False)
        finally:
            vstream.codec_context.skip_frame = "DEFAULT"

        last = None
        for pkt in container.demux(vstream):
            for fr in pkt.decode():
                last = fr
                if fr.time is not None and fr.time + 1e-6 >= t_sec:
                    return fr.to_ndarray(format="rgb24")
        # Fallbacks
        if last is not None:
            return last.to_ndarray(format="rgb24")
        container.seek(0, stream=vstream)
        for pkt in container.demux(vstream):
            for fr in pkt.decode():
                return fr.to_ndarray(format="rgb24")
        raise RuntimeError("Decode failed")

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target_size if needed. Uses AREA interpolation for downsampling."""
        h, w = frame.shape[:2]
        target_h, target_w = self.target_size

        # Skip if already at target size
        if h == target_h and w == target_w:
            return frame

        # Convert to torch tensor (HWC -> CHW), resize, then convert back
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)  # CHW
        # Use AREA interpolation (antialias=True) for best quality downsampling
        # This is the fastest high-quality method for downsampling
        resized = TF.resize(frame_tensor, [target_h, target_w], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        return resized.permute(1, 2, 0).numpy()  # CHW -> HWC

    def _decode_at_time(self, path: Path, t_sec: float) -> np.ndarray:
        """Open the file, decode at t, close; returns RGB24 (H,W,3)."""
        with av.open(str(path), options={"fflags": "fastseek+nobuffer"}) as c:
            v = next(s for s in c.streams if s.type == "video")
            return self._decode_from_open(c, v, t_sec)

    def _decode_window_from_open(
            self,
            container: av.container.input.InputContainer,
            vstream: av.video.stream.VideoStream,
            t_sec: float,
            fps: float
    ) -> list:
        """Decode window_length consecutive frames starting at t_sec; returns list of RGB24 (H,W,3)."""
        tb: Fraction = vstream.time_base
        # Light speed knobs
        try:
            vstream.thread_type = "FRAME"
            vstream.codec_context.thread_count = 1
            vstream.codec_context.skip_loop_filter = "ALL"
        except Exception:
            pass

        # Clamp t to known duration if available
        if container.duration is not None:
            t_sec = min(max(0.0, t_sec), max(0.0, container.duration / 1e6 - 1e-3))

        # Keyframe seek then decode forward to >= t
        try:
            vstream.codec_context.skip_frame = "NONKEY"
            container.seek(int(max(0.0, t_sec) / float(tb)), stream=vstream, backward=True, any_frame=False)
        finally:
            vstream.codec_context.skip_frame = "DEFAULT"

        frames = []
        started = False

        for pkt in container.demux(vstream):
            for fr in pkt.decode():
                # Check if we've reached the target time
                if not started and fr.time is not None and fr.time + 1e-6 >= t_sec:
                    started = True

                if started:
                    frames.append(fr.to_ndarray(format="rgb24"))
                    if len(frames) >= self.window_length:
                        return frames[:self.window_length]

        # If we didn't get enough frames, raise an error to trigger retry
        if len(frames) < self.window_length:
            raise RuntimeError(f"Could not decode {self.window_length} frames, only got {len(frames)}")

        return frames[:self.window_length]

    def _decode_window_at_time(self, path: Path, t_sec: float, fps: float) -> list:
        """Open the file, decode window at t, close; returns list of RGB24 (H,W,3)."""
        with av.open(str(path), options={"fflags": "fastseek+nobuffer"}) as c:
            v = next(s for s in c.streams if s.type == "video")
            return self._decode_window_from_open(c, v, t_sec, fps)


class RandomRGBDataset(IterableDataset):
    """
    Infinite stream of CHW uint8 frames. One independent generator per worker.
    Assumes frames share a common resolution so default collate can stack.
    If window_length > 1, yields [window_length, C, H, W] tensors.
    """
    def __init__(self, source, seed: int = 0, target_size = (360, 640), window_length = 1, rank: int = 0, world_size: int = 1, suppress_warnings = True):
        super().__init__()
        # Resolve globs eagerly in main process so workers don't repeat the work
        self.paths = RandomRGBFromMP4s._find_mp4s(source)
        if not self.paths:
            raise RuntimeError("No videos found in the supplied source.")
        self.seed = int(seed)
        self.target_size = target_size
        self.window_length = window_length
        self.rank = rank
        self.world_size = world_size
        self.suppress_warnings = suppress_warnings

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        # Derive a per-worker seed (works with persistent workers)
        # Incorporate rank to ensure different data across nodes
        wseed = (torch.initial_seed() + self.seed + wid + self.rank * 10000) % (2**32)
        rng = RandomRGBFromMP4s(None, seed=int(wseed), target_size = self.target_size, window_length = self.window_length, suppress_warnings = self.suppress_warnings, _resolved_paths=self.paths)
        for rgb in rng:
            # HWC uint8 -> CHW uint8 (or THWC -> TCHW for windows)
            # clone() gives the tensor its own resizable storage, preventing rare 'resize_ not allowed' errors.
            if self.window_length == 1:
                # Single frame: [H, W, C] -> [C, H, W]
                yield torch.from_numpy(rgb).permute(2, 0, 1).contiguous().clone().bfloat16() / 127.5 - 1.0
            else:
                # Window: [T, H, W, C] -> [T, C, H, W]
                yield torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous().clone().bfloat16() / 127.5 - 1.0

def get_loader(batch_size, rank=0, world_size=1, **data_kwargs):
    if "seed" not in data_kwargs:
        data_kwargs["seed"] = 123
    # Add rank and world_size to ensure distributed training works correctly
    data_kwargs["rank"] = rank
    data_kwargs["world_size"] = world_size
    ds = RandomRGBDataset(**data_kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        multiprocessing_context="spawn",
    )


if __name__ == "__main__":
    import time
    from PIL import Image

    print("Testing window_length=120 on ./data")
    loader = get_loader(
        1,
        source="./data",
        target_size=(360, 640),
        window_length=120
    )

    loader_iter = iter(loader)
    t0 = time.time()
    batch = next(loader_iter)
    t1 = time.time()

    print(f"Batch shape: {tuple(batch.shape)}, dtype: {batch.dtype}, load_time: {t1-t0:.4f}s")
    print(f"Value range: [{batch.min().item():.3f}, {batch.max().item():.3f}]")

    # Take first sample: [T, C, H, W]
    sample = batch[0]  # [120, 3, 360, 640]

    # Convert from [-1, 1] bfloat16 to [0, 255] uint8
    sample = ((sample.float() + 1.0) * 127.5).clamp(0, 255).byte()

    # Convert to [T, H, W, C] for PIL
    sample = sample.permute(0, 2, 3, 1).cpu().numpy()  # [120, 360, 640, 3]

    # Create PIL images and save as gif
    frames = [Image.fromarray(frame) for frame in sample]
    frames[0].save(
        "test_window.gif",
        save_all=True,
        append_images=frames[1:],
        duration=33,  # ~30 fps
        loop=0
    )
    print("Saved test_window.gif")