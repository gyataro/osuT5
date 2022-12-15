from __future__ import annotations

import uuid
import dataclasses
from string import Template

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

OSU_FILE_EXTENSION = ".osu"
MILISECONDS_PER_STEP = 10
MILISECONDS_PER_SECOND = 1000


@dataclasses.dataclass
class InferenceConfig:
    src_seq_len: int = 512
    tgt_seq_len: int = 256
    frame_size: int = 128
    sample_rate: int = 16000
    batch_size: int = 4


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
    circle_size: float = 5
    overall_difficulty: float = 5
    approach_rate: float = 5
    slider_multiplier: float = 1.4


@dataclasses.dataclass
class BeatmapTiming:
    offset: int = 0
    beat_length: float = 120


class Preprocessor(object):
    def __init__(self, config: InferenceConfig):
        """Preprocess audio data into sequences."""
        self.frame_seq_len = config.src_seq_len - 1
        self.frame_size = config.frame_size
        self.sample_rate = config.sample_rate
        self.samples_per_sequence = self.frame_seq_len * self.frame_size
        self.batch_size = config.batch_size

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


class Pipeline(object):
    def __init__(self, config: InferenceConfig):
        """Model inference stage that processes sequences."""
        self.tokenizer = Tokenizer()
        self.batch_size = config.batch_size
        self.tgt_seq_len = config.tgt_seq_len
        self.frame_seq_len = config.src_seq_len - 1
        self.frame_size = config.frame_size
        self.sample_rate = config.sample_rate
        self.samples_per_sequence = self.frame_seq_len * self.frame_size
        self.miliseconds_per_sequence = (
            self.samples_per_sequence * MILISECONDS_PER_SECOND / self.sample_rate
        )

    def generate(
        self, model: OsuTransformer, sequences: tuple[torch.Tensor]
    ) -> tuple[list[list[Event]], list[int]]:
        """Generate a list of Event object lists and their timestamps given source sequences.

        Args:
            model: Trained model to use for inference.
            sequences: A list of batched source sequences.

        Returns:
            events: List of Event object lists.
            event_times: Corresponding event times of Event object lists in miliseconds.
        """
        events, event_times = [], []

        for batch_index, sources in enumerate(tqdm(sequences)):
            targets = torch.LongTensor(
                [[self.tokenizer.sos_id] for _ in range(self.batch_size)]
            )

            for _ in tqdm(range(self.tgt_seq_len - 1), leave=False):
                logits = model.forward(sources, targets)
                logits = logits[:, :, -1]
                logits = self._filter(logits, 0.9)
                probabilities = F.softmax(logits, dim=-1)
                token = torch.multinomial(probabilities, 1)

                targets = torch.cat([targets, token], dim=-1)

            for seq_index, target in enumerate(targets):
                index = batch_index * self.batch_size + seq_index
                print(index)
                result = self._decode(target, index)
                events += result[0]
                event_times += result[1]

        return events, event_times

    def _decode(
        self, tokens: torch.Tensor, index: int
    ) -> tuple[list[list[Event]], list[int]]:
        """Converts a list of tokens into Event object lists and their timestamps.

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

            event = self.tokenizer.decode(token.item())

            if event.type == EventType.TIME_SHIFT:
                timestamp = (
                    index * self.miliseconds_per_sequence
                    + event.value * MILISECONDS_PER_STEP
                )
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
            logits: logits distribution of shape (batch size, vocabulary size).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).

        Returns:
            logits of top tokens.
        """
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p

            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value

        return logits


class Postprocessor(object):
    def __init__(self, config: InferenceConfig):
        """Postprocessing stage that converts a list of Event objects to a beatmap file."""
        self.curve_types = {
            EventType.SLIDER_BEZIER: "B",
            EventType.SLIDER_CATMULI: "C",
            EventType.SLIDER_LINEAR: "L",
            EventType.SLIDER_PERFECT_CIRCLE: "P",
        }

    def generate(
        self,
        events: list[list[Event]],
        event_times: list[int],
        metadata: BeatmapMetadata,
        difficulty: BeatmapDifficulty,
        timing: BeatmapTiming,
    ):
        """Generate a beatmap file.

        Args:
            events: List of Event object lists.
            event_times: Corresponding event times of Event object lists in miliseconds.
            metadata: Beatmap metadata
            difficulty: Beatmap difficulty configuration
            timing: Beatmap timing configuration

        Returns:
            None. An .osu file with a random UUID filename will be generated.
        """

        hit_object_strings = []

        for hit_object, timestamp in zip(events, event_times):
            print(hit_object)
            x = hit_object[0].value
            y = hit_object[1].value
            hit_type = hit_object[2].type

            if hit_type == EventType.CIRCLE:
                hit_object_strings.append(f"{x},{y},{timestamp},1,0")

            elif hit_type in self.curve_types:
                curve_type = self.curve_types[hit_type]
                slides = 1
                control_points = ""

                if hit_object[-1].type == EventType.SLIDES:
                    slides = hit_object[-1].value

                for i in range(3, len(hit_object) - 1, 2):
                    if (
                        hit_object[i].type == EventType.CONTROL_POINT
                        and hit_object[i + 1].type == EventType.CONTROL_POINT
                    ):
                        control_points += (
                            f"|{hit_object[i].value}:{hit_object[i+1].value}"
                        )

                hit_object_strings.append(
                    f"{x},{y},{timestamp},2,0,{curve_type}{control_points},{slides}"
                )

        with open("template.osu", "r") as tf:
            template = Template(tf.read())
            metadata = dataclasses.asdict(metadata)
            difficulty = dataclasses.asdict(difficulty)
            timing = dataclasses.asdict(timing)
            hit_objects = {"hit_objects": "\n".join(hit_object_strings)}

            result = template.safe_substitute(
                {**metadata, **difficulty, **timing, **hit_objects}
            )

            with open(f"{uuid.uuid4().hex}{OSU_FILE_EXTENSION}", "w") as f:
                f.write(result)
                f.close()

            tf.close()


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    config = InferenceConfig()

    checkpoint = torch.load("./checkpoint.ckpt")

    model = OsuTransformer(Config())
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    pipeline = Pipeline(config)
    preprocessor = Preprocessor(config)
    postprocessor = Postprocessor(config)

    audio = preprocessor.load("./audio.mp3")
    sequences = preprocessor.segment(audio)
    events, event_times = pipeline.generate(model, sequences)

    beatmap_metadata = BeatmapMetadata()
    beatmap_difficulty = BeatmapDifficulty()
    beatmap_timing = BeatmapTiming()
    postprocessor.generate(
        events, event_times, beatmap_metadata, beatmap_difficulty, beatmap_timing
    )
