from __future__ import annotations

import sys
import random
import logging
from glob import glob

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import IterableDataset

from .osu_parser import OsuParser
from .osz_loader import OszLoader
from osuT5.tokenizer import Event, EventType

OSZ_FILE_EXTENSION = ".osz"
MILISECONDS_PER_SECOND = 1000
STEPS_PER_MILLISECOND = 0.1


class OszDataset(IterableDataset):
    def __init__(
        self,
        path: str,
        tokenizer: type,
        sample_rate: int = 16000,
        frame_size: int = 16000,
        src_seq_len: int = 128,
        tgt_seq_len: int = 256,
    ):
        """Manage and process .osz archives.

        Attributes:
            path: Location of .osz files to load.
            tokenizer: Instance of Tokenizer class.
            sample_rate: Sampling rate of audio file (samples/second).
            frame_size: Samples per audio frame (samples/frame).
            src_seq_len: Maximum length of source sequence.
            tgt_seq_len: Maximum length of target sequence.
        """
        super().__init__()
        self.dataset = np.array(
            glob(f"{path}/**/*{OSZ_FILE_EXTENSION}", recursive=True), dtype=np.string_
        )
        np.random.shuffle(self.dataset)
        self.loader = OszLoader()
        self.parser = OsuParser()
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        # let N = |src_seq_len|
        # N-1 frames creates N mel-spectrogram frames
        self.frame_seq_len = src_seq_len - 1
        # let N = |tgt_seq_len|
        # [SOS] token + event_tokens + [EOS] token creates N+1 tokens
        # [SOS] token + event_tokens[:-1] creates N target sequence
        # event_tokens[1:] + [EOS] token creates N label sequence
        self.token_seq_len = tgt_seq_len + 1
        self.encoding = sys.getfilesystemencoding()

    def _get_audio_and_osu(self, osz_path: str) -> tuple[npt.NDArray, list[str]]:
        """Load an .osz archive and get its audio samples and .osu beatmap.

        An .osz archive may have multiple .osu beatmaps, only one is selected based
        on OszLoader's criteria.

        If an archive is corrupted (bad audio, bad metadata, missing files etc.),
        we index the selection result as `None`, which will be skipped on subsequent queries.

        Args:
            osz_path: Path to the .osz archive.

        Returns:
            audio_samples: Audio time series.
            osu_beatmap: A list of strings (osu beatmap data).
        """
        try:
            audio_samples, osu_beatmap, _, _ = self.loader.load_osz(
                osz_path,
            )
        except Exception as e:
            logging.warn(f"skipped: {osz_path}")
            logging.warn(f"reason: {e}")
            return None, None

        return audio_samples, osu_beatmap

    def _get_frames(self, samples: npt.NDArray) -> list[float]:
        """Segment audio samples into frames.

        Each frame has `frame_size` audio samples.
        It will also calculate and return the time of each audio frame, in miliseconds.

        Args:
            samples: Audio time-series.

        Returns:
            frames: Audio frames.
            frame_times: Audio frame times.
        """
        samples = np.pad(samples, [0, self.frame_size - len(samples) % self.frame_size])
        frames = np.reshape(samples, (-1, self.frame_size))
        frames_per_milisecond = (
            self.sample_rate / self.frame_size / MILISECONDS_PER_SECOND
        )
        frame_times = np.arange(len(frames)) / frames_per_milisecond
        return frames, frame_times

    def _tokenize_and_index_events(
        self,
        events: list[list[Event]],
        event_times: list[int],
        frame_times: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        """Tokenize Event objects and index them to audio frame times.

        Tokenize every time shift as multiple single time steps.
        It should always be true that event_end_indices[i] = event_start_indices[i + 1].

        Args:
            events: List of Event object lists
            event_times: Time of each Event object list, in miliseconds
            frame_times: Audio frame times, in miliseconds

        Returns:
            event_tokens: Tokenized Events and time shifts
            event_start_indices: Corresponding start event index for every audio frame
            event_end_indices: Corresponding end event index for every audio frame
        """
        TIME_STEP_TOKEN = self.tokenizer.time_step_id
        event_steps = [round(t * STEPS_PER_MILLISECOND) for t in event_times]

        cur_step = 0
        cur_event_idx = 0

        event_tokens = []
        event_start_indices = []
        event_end_indices = []

        def fill_event_start_indices_to_cur_step():
            while (
                len(event_start_indices) < len(frame_times)
                and frame_times[len(event_start_indices)]
                < cur_step / STEPS_PER_MILLISECOND
            ):
                event_start_indices.append(cur_event_idx)

        for event_step, event_list in zip(event_steps, events):
            while event_step > cur_step:
                event_tokens.append(TIME_STEP_TOKEN)
                cur_step += 1
                fill_event_start_indices_to_cur_step()
                cur_event_idx = len(event_tokens)

            try:
                new_tokens = []
                for event in event_list:
                    token = self.tokenizer.encode(event)
                    new_tokens.append(token)
                event_tokens += new_tokens
            except Exception as e:
                logging.warn(f"tokenization failed: {e}")

        while cur_step / STEPS_PER_MILLISECOND <= frame_times[-1]:
            event_tokens.append(TIME_STEP_TOKEN)
            cur_step += 1
            fill_event_start_indices_to_cur_step()
            cur_event_idx = len(event_tokens)

        event_end_indices = event_start_indices[1:] + [len(event_tokens)]

        return event_tokens, event_start_indices, event_end_indices

    def _create_sequences(
        self,
        event_tokens: list[int],
        event_start_indices: list[int],
        event_end_indices: list[int],
        frames: npt.NDArray,
        frames_per_split: int = 1024,
    ) -> list[dict[npt.NDArray, list[int]]]:
        """Create frame and token sequences for training/testing.

        Args:
            event_tokens: Tokenized Events and time shifts.
            event_start_indices: Corresponding start event index for every audio frame.
            event_end_indices: Corresponding end event index for every audio frame.
            frames: Audio frames.
            frames_per_split: Maximum number of frames in each split.

        Returns:
            A list of source and target sequences.
        """
        sequences = []
        n_frames = len(frames)
        # Divide audio frames into splits
        for split_start_idx in range(0, n_frames, frames_per_split):
            split_end_idx = min(split_start_idx + frames_per_split, n_frames)
            split_frames = frames[split_start_idx:split_end_idx]
            split_event_starts = event_start_indices[split_start_idx:split_end_idx]
            split_event_ends = event_end_indices[split_start_idx:split_end_idx]

            # For each split, randomly select a contiguous sequence of frames and events
            max_offset = len(split_frames) - self.frame_seq_len
            if max_offset < 1:
                sequence_start_idx = 0
                sequence_end_idx = len(split_frames)
            else:
                sequence_start_idx = random.randint(0, max_offset)
                sequence_end_idx = sequence_start_idx + self.frame_seq_len

            # Create the sequence
            sequence = {}
            sequence_event_starts = split_event_starts[
                sequence_start_idx:sequence_end_idx
            ]
            sequence_event_ends = split_event_ends[sequence_start_idx:sequence_end_idx]
            target_start_idx = sequence_event_starts[0]
            target_end_idx = sequence_event_ends[-1]
            sequence["frames"] = split_frames[sequence_start_idx:sequence_end_idx]
            sequence["tokens"] = event_tokens[target_start_idx:target_end_idx]
            sequences.append(sequence)

        return sequences

    def _merge_time_step_tokens(self, sequence):
        """Merge time steps into time shifts.

        Convert relative time shifts into absolute time shifts.
        Each time shift token now indicates the amount of time from the beginning of the sequence.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with relative time steps converted into absolute time shifts.
        """
        total_time_shift = 0
        tokens = []
        is_redundant = False

        for token in sequence["tokens"]:
            if token == self.tokenizer.time_step_id:
                total_time_shift += 1
                is_redundant = False
            else:
                if not is_redundant:
                    shift_event = Event(EventType.TIME_SHIFT, total_time_shift)
                    shift_token = self.tokenizer.encode(shift_event)
                    tokens.append(shift_token)
                    is_redundant = True
                tokens.append(token)

        sequence["tokens"] = tokens
        return sequence

    def _pad_token_sequence(self, sequence):
        """Pad token sequence to a fixed length.

        Begin token sequence with `[PAD]` token (start-of-sequence).
        End token sequence with `[EOS]` token (end-of-sequence).
        Then, pad with `[PAD]` tokens until `token_seq_len`.

        Token sequence (w/o last token) is the input to the transformer decoder,
        token sequence (w/o first token) is the label, a.k.a decoder ground truth.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with padded tokens.
        """
        tokens = torch.tensor(sequence["tokens"], dtype=torch.long)
        n = min(self.tgt_seq_len - 1, len(tokens))
        sos = (
            torch.ones(1, dtype=tokens.dtype, device=tokens.device)
            * self.tokenizer.sos_id
        )
        eos = (
            torch.ones(1, dtype=tokens.dtype, device=tokens.device)
            * self.tokenizer.eos_id
        )
        padded_tokens = (
            torch.ones(self.token_seq_len, dtype=tokens.dtype, device=tokens.device)
            * self.tokenizer.pad_id
        )
        padded_tokens[0] = sos
        padded_tokens[1 : n + 1] = tokens[:n]
        padded_tokens[n + 1 : n + 2] = eos
        sequence["tokens"] = padded_tokens
        return sequence

    def _pad_frame_sequence(self, sequence):
        """Pad frame sequence with zeros until `frame_seq_len`.

        Frame sequence can be further processed into Mel spectrogram frames,
        which is the input to the transformer encoder.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with padded frames.
        """
        frames = torch.from_numpy(sequence["frames"]).to(torch.float32)

        if frames.shape[0] != self.frame_seq_len:
            n = min(self.frame_seq_len, len(frames))
            padded_frames = torch.zeros(
                self.frame_seq_len,
                frames.shape[-1],
                dtype=frames.dtype,
                device=frames.device,
            )
            padded_frames[:n] = frames[:n]
            sequence["frames"] = padded_frames
        else:
            sequence["frames"] = frames

        return sequence

    def _get_next(self) -> dict[int, int]:
        """Generate samples.

        A single .osz archive may yield multiple samples.

        This is a generator that provides unlimited samples. When all .osz archives are
        read, it will cycle to the beginning and repeat the samples.

        Yields:
            A sample, which contains a source sequence of `frame_seq_len` audio frames
            and target sequence of `token_seq_len` event tokens.
        """
        while True:
            for osz_path in np.nditer(self.dataset):
                osz_path = str(osz_path, encoding=self.encoding)
                audio_samples, osu_beatmap = self._get_audio_and_osu(osz_path)

                if audio_samples is None or osu_beatmap is None:
                    continue

                frames, frame_times = self._get_frames(audio_samples)
                events, event_times = self.parser.parse(osu_beatmap)

                (
                    event_tokens,
                    event_start_indices,
                    event_end_indices,
                ) = self._tokenize_and_index_events(events, event_times, frame_times)

                sequences = self._create_sequences(
                    event_tokens,
                    event_start_indices,
                    event_end_indices,
                    frames,
                )

                for sequence in sequences:
                    sequence = self._merge_time_step_tokens(sequence)
                    sequence = self._pad_frame_sequence(sequence)
                    sequence = self._pad_token_sequence(sequence)
                    if sequence["tokens"][1] == self.tokenizer.eos_id:
                        continue
                    yield sequence

    def __iter__(self):
        """Get a single sample from the dataset."""
        return self._get_next()
