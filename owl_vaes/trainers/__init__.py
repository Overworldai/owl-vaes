from typing import Literal

def get_trainer_cls(trainer_id: Literal["rec", "proxy", "audio_rec", "taehv"]):
    match trainer_id:
        case "rec":
            from .rec import RecTrainer
            return RecTrainer
        case "diff_dec":
            from .diffdec_trainer import DiffusionDecoderTrainer
            return DiffusionDecoderTrainer
        case "distill_dec":
            from .distill_dec import DistillDecTrainer
            return DistillDecTrainer
        case "distill_dec_seraena":
            from .distill_dec_seraena import SerDistillDecTrainer
            return SerDistillDecTrainer
        case "distill_enc":
            from .distill_enc import DistillEncTrainer
            return DistillEncTrainer
        case "distill_pretrained_enc":
            from .distill_pretrained_enc import DistillPretrainedEncTrainer
            return DistillPretrainedEncTrainer
        case "proxy_diffdec":
            from .proxy_diffdec import DiffDecLiveDepthTrainer
            return DiffDecLiveDepthTrainer
        case "video_rec":
            from .video_rec import VideoRecTrainer
            return VideoRecTrainer
        case "dito":
            from .dito import DiToTrainer
            return DiToTrainer
        case "video_dito":
            from .video_dito import VideoDiToTrainer
            return VideoDiToTrainer
        case "taehv":
            from .taehv_trainer import TAEHVTrainer
            return TAEHVTrainer
        case _:
            raise NotImplementedError
