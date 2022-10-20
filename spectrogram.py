from lib2to3.pgen2.token import NT_OFFSET
from random import sample
import torch
import torch.nn as nn
from nnAudio import features


class MelSpectrogram(nn.Module):
    def __init__(self, sample_rate: int, n_ftt: int, n_mels: int, hop_length: int):
        """
        melspectrogram transformation layer, supports on-the-fly processing on GPU
        :param sample_rate: the sampling rate for the input audio
        :param n_ftt: the window size for the STFT
        :param n_mels: the number of Mel filter banks
        :param hop_length: the hop (or stride) size
        """
        super().__init__()
        self.transform = features.MelSpectrogram(
            sr=sample_rate,
            n_fft=n_ftt,
            n_mels=n_mels,
            hop_length=hop_length,
            center=True,
            fmin=0,
            fmax=sample_rate // 2,
            pad_mode="constant",
        )

    def forward(self, samples) -> torch.tensor:
        """
        converts a batch of audio samples into a batch of Mel spectrogram frames
        for each audio in batch:
            1. pad left and right ends of audio by n_fft // 2
            2. run STFT with window size of |n_ftt| and stride of |hop_length|
            3. convert result into mel-scale
            4. therefore, n_frames = n_samples // hop_length + 1
        :param samples: audio time-series (batch size, n_samples)
        :return: batch of Mel spectrograms of size (batch size, n_frames, n_mels)
        """
        spectrogram = self.transform(samples)
        spectrogram = spectrogram.permute(0, 2, 1)
        return spectrogram
