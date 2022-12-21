import dataclasses


@dataclasses.dataclass
class Config:
    model_path: str = "./lightning_logs/checkpoint.ckpt"
    audio_path: str = "./audio.mp3"
    src_seq_len: int = 512
    tgt_seq_len: int = 256
    frame_size: int = 128
    sample_rate: int = 16000
    batch_size: int = 32


@dataclasses.dataclass
class BeatmapMetadata:
    audio_filename: str = "audio.mp3"
    title: str = "osu_beatmap"
    artist: str = "osu_transformer"
    title_unicode: str = "osu_beatmap"
    artist_unicode: str = "osu_transformer"


@dataclasses.dataclass
class BeatmapDifficulty:
    hp_drain_rate: float = 5
    circle_size: float = 4
    overall_difficulty: float = 8
    approach_rate: float = 9
    slider_multiplier: float = 1.8


@dataclasses.dataclass
class BeatmapTiming:
    offset: int = 0
    beat_length: float = 500
