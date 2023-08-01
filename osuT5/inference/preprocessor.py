from __future__ import annotations
import torch
import numpy as np
import numpy.typing as npt
from pydub import AudioSegment
from omegaconf import DictConfig


class Preprocessor(object):
    def __init__(self, args: DictConfig):
        """Preprocess audio data into sequences."""
        self.frame_seq_len = args.model.max_seq_len - 1
        self.frame_size = args.model.spectrogram.hop_length
        self.sample_rate = args.model.spectrogram.sample_rate
        self.samples_per_sequence = self.frame_seq_len * self.frame_size
        self.batch_size = args.batch_size

    def load(self, path: str) -> npt.ArrayLike:
        """Load an audio file as audio frames. Convert stereo to mono, normalize.

        Args:
            path: Path to audio file.

        Returns:
            samples: Audio time-series.
        """
        audio = AudioSegment.from_file(path)
        audio = audio.set_frame_rate(self.sample_rate)
        audio = audio.set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples *= 1.0 / np.max(np.abs(samples))
        return samples

    def segment(self, samples: npt.ArrayLike) -> tuple[torch.Tensor]:
        """Segment audio samples into sequences. Sequences are flattened frames.

        Args:
            samples: Audio time-series.

        Returns:
            sequences: A list of sequences of shape (batch size, samples per sequence).
        """
        samples = np.pad(
            samples,
            [0, self.samples_per_sequence - len(samples) % self.samples_per_sequence],
        )
        sequences = np.reshape(samples, (-1, self.samples_per_sequence))
        sequences = torch.from_numpy(sequences).to(torch.float32)
        return torch.split(sequences, self.batch_size)
