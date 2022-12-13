from __future__ import annotations

import torch
import numpy as np
import numpy.typing as npt
import torch.nn.functional as F
from pydub import AudioSegment
from tqdm import tqdm

from model import OsuTransformer
from data.tokenizer import Tokenizer
from data.event import Event, EventType
from config.config import Config

MILISECONDS_PER_STEP = 10
MILISECONDS_PER_SECOND = 1000


class Preprocessor(object):
    def __init__(self, config):
        """Preprocess audio data into sequences."""
        self.frame_seq_len = config.dataset.src_seq_len - 1
        self.frame_size = config.spectrogram.hop_length
        self.sample_rate = config.spectrogram.sample_rate
        self.samples_per_sequence = self.frame_seq_len * self.frame_size

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

    def segment(self, samples: npt.ArrayLike) -> torch.Tensor:
        """Segment audio samples into sequences. Sequences are flattened frames.

        Args:
            samples: Audio time-series.

        Returns:
            sequences: Sequences each with `samples_per_sequence` samples.
        """
        samples = np.pad(
            samples,
            [0, self.samples_per_sequence - len(samples) % self.samples_per_sequence],
        )
        sequences = np.reshape(samples, (-1, self.samples_per_sequence))
        return torch.from_numpy(sequences).to(torch.float32)


class Pipeline(object):
    def __init__(self, config):
        """Model inference stage that processes sequences."""
        self.tokenizer = Tokenizer()
        self.tgt_seq_len = config.dataset.tgt_seq_len
        self.frame_seq_len = config.dataset.src_seq_len - 1
        self.frame_size = config.spectrogram.hop_length
        self.sample_rate = config.spectrogram.sample_rate
        self.samples_per_sequence = self.frame_seq_len * self.frame_size
        self.miliseconds_per_sequence = (
            self.samples_per_sequence * MILISECONDS_PER_SECOND / self.sample_rate
        )

    def generate(self, model: OsuTransformer, sequences: torch.Tensor):
        """Generate a list of Event object lists and their timestamps given source sequences.

        Args:
            model: Trained model to use for inference.
            sequences: Input source sequences.

        Returns:
            events: List of Event object lists.
            event_times: Corresponding event times of Event object lists in miliseconds.
        """
        events, event_times = [], []

        for index, sequence in enumerate(tqdm(sequences)):
            src = torch.unsqueeze(sequence, 0)
            tgt = torch.tensor([[self.tokenizer.sos_id]])

            for _ in tqdm(range(self.tgt_seq_len - 1), leave=False):
                logits = model.forward(src, tgt)
                logits = logits[0, :, -1]
                logits = self._filter(logits, 0.9)
                probabilities = F.softmax(logits, dim=-1)
                token = torch.multinomial(probabilities, 1)

                if token == self.tokenizer.eos_id:
                    break

                tgt = torch.cat([tgt, token.unsqueeze(-1)], dim=-1)

            result = self._decode(tgt[0], index)
            events.append(result[0])
            event_times.append(result[1])

        return events, event_times

    def _decode(
        self, tokens: torch.Tensor, index: int
    ) -> tuple[list[list[Event]], list[int]]:
        """Convers a list of tokens into Event object lists and their timestamps.

        Args:
            tokens: List of tokens.
            index: Index of current source sequence.

        Returns:
            events: List of Event object lists.
            event_times: Corresponding event times of Event object lists in miliseconds.
        """
        events, event_times = [], []
        for token in tokens:
            if token == self.tokenizer.sos_id:
                continue
            elif token == self.tokenizer.eos_id:
                break

            event = self.tokenizer.decode(token)

            if event.type == EventType.TIME_SHIFT:
                timestamp = index * self.miliseconds_per_sequence + event.value
                events.append([])
                event_times.append(timestamp)
            else:
                events[-1].append(event)

        return events, event_times

    def _filter(
        self, logits: torch.Tensor, top_p: float, filter_value: float = -float("Inf")
    ) -> torch.Tensor:
        """Filter a distribution of logits using nucleus (top-p) filtering.

        Source: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

        Args:
            logits: logits distribution shape (vocabulary size).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).

        Returns:
            logits of top tokens.
        """
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        return logits


class Postprocessor(object):
    def __init__(self):
        """Generate a beatmap from a list of Event objects."""


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    config = Config()

    checkpoint = torch.load(
        "./lightning_logs/checkpoint.ckpt",
        map_location=torch.device("cpu"),
    )

    model = OsuTransformer(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    preprocessor = Preprocessor(config)
    pipeline = Pipeline(config)

    audio = preprocessor.load("./audio.mp3")
    sequences = preprocessor.segment(audio)
    events, event_times = pipeline.generate(model, sequences)

    print(events)
    print(event_times)
