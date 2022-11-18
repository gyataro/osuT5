from __future__ import annotations
import logging

import torch
import numpy as np
from torch import nn
from pydub import AudioSegment
from scipy.signal import resample

from train import OsuTransformer
from data.tokenizer import Tokenizer
from data.event import EventType
from config.config import Config


class Preprocessor(object):
    def __init__(self, config: Config):
        self.frame_seq_len = config.dataset.src_seq_len - 1
        self.frame_size = config.spectrogram.hop_length
        self.sample_rate = config.spectrogram.sample_rate

    def preprocess(self):
        # 1. Convert stereo to mono, normalize.
        audio = AudioSegment.from_file("./audio.mp3", format="mp3")
        audio = audio.set_frame_rate(self.sample_rate)
        audio = audio.set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        # samples = samples.mean(axis=1).astype(np.float32)
        samples *= 1.0 / np.max(np.abs(samples))

        # 2. Segment audio samples into frames.
        samples = np.pad(samples, [0, self.frame_size - len(samples) % self.frame_size])
        frames = np.reshape(samples, (-1, self.frame_size))

        # 3. Create frame sequences.
        sequences = []
        n_frames = len(frames)
        for split_start_idx in range(0, n_frames, self.frame_seq_len):
            split_end_idx = min(split_start_idx + self.frame_seq_len, n_frames)
            split_frames = frames[split_start_idx:split_end_idx]

            split_frames = torch.from_numpy(split_frames).to(torch.float32)

            # 4. Pad each frame sequence with zeros until `frame_seq_len`.
            if len(split_frames) != self.frame_seq_len:
                n = min(self.frame_seq_len, len(split_frames))
                padded_frames = torch.zeros(
                    self.frame_seq_len,
                    split_frames.shape[-1],
                    dtype=split_frames.dtype,
                    device=split_frames.device,
                )
                padded_frames[:n] = split_frames[:n]
                sequences.append(padded_frames)
            else:
                sequences.append(split_frames)

        return sequences


if __name__ == "__main__":
    config = Config()

    checkpoint = torch.load(
        "./lightning_logs/version_4/checkpoints/last.ckpt",
        map_location=torch.device("cpu"),
    )

    model = OsuTransformer(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    preprocessor = Preprocessor(config)
    sequences = preprocessor.preprocess()
    tokenizer = Tokenizer()

    for index, frames in enumerate(sequences):
        logging.info(f"processing frame: {index+1}/{len(frames)}")

        with torch.no_grad():
            tgt = torch.tensor([[tokenizer.sos_id]])

            for i in range(255):
                src = torch.unsqueeze(frames, 0)
                logits = model.forward(src, tgt)
                token = torch.argmax(logits, dim=1)
                token = token[0, -1]

                if token == tokenizer.eos_id:
                    break

                token = torch.tensor([[token]])
                tgt = torch.cat([tgt, token], dim=1)

        events = []
        hit_object = []
        for token in tgt[0, :]:
            if token == tokenizer.eos_id or token == tokenizer.sos_id:
                continue

            event = tokenizer.decode(token)

            if event.type == EventType.TIME_SHIFT and len(hit_object) > 0:
                events.append(hit_object)
                hit_object = []

            hit_object.append(event)

        events.append(hit_object)
