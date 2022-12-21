from __future__ import annotations

import argparse

import torch

from inference.config import Config, BeatmapMetadata, BeatmapDifficulty, BeatmapTiming
from inference.preprocessor import Preprocessor
from inference.pipeline import Pipeline
from inference.postprocessor import Postprocessor
from utils.tokenizer import Tokenizer
from model import OsuTransformer

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "-m",
    "--model_path",
    help="path to trained model",
    type=str,
)
argparser.add_argument(
    "-a",
    "--audio_path",
    help="path to input audio",
    type=str,
)
argparser.add_argument(
    "-b",
    "--bpm",
    help="beats per minute of input audio",
    type=float,
)
argparser.add_argument(
    "-o",
    "--offset",
    help="start of beat, in miliseconds, from the beginning of input audio",
    type=int,
)
args = argparser.parse_args()

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config(audio_path=args.audio_path, model_path=args.model_path)

    checkpoint = torch.load(config.model_path, map_location=device)

    model = OsuTransformer()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    tokenizer = Tokenizer()
    pipeline = Pipeline(config, tokenizer)
    preprocessor = Preprocessor(config)
    postprocessor = Postprocessor(config)

    audio = preprocessor.load(config.audio_path)
    sequences = preprocessor.segment(audio)
    events, event_times = pipeline.generate(model, sequences)

    beat_length = 60000 / args.bpm
    beatmap_metadata = BeatmapMetadata()
    beatmap_difficulty = BeatmapDifficulty()
    beatmap_timing = BeatmapTiming(offset=args.offset, beat_length=beat_length)
    postprocessor.generate(
        events,
        event_times,
        beatmap_metadata,
        beatmap_difficulty,
        beatmap_timing,
    )
