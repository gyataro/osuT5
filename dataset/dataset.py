"""
Testing dataloaders and datasets
"""
import random
from glob import glob

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.sampler import RandomSampler
from itertools import cycle

import dataset.parser as parser
from dataset.loader import OszLoader
from tokenizer import Tokenizer
from event import Event, EventType

OSZ_FILE_EXTENSION = ".osz"
MILISECONDS_PER_SECOND = 1000
STEPS_PER_MILLISECOND = 0.1


class OszDataset(IterableDataset):
    def __init__(
        self,
        dataset_directory: str,
        loader: OszLoader,
        tokenizer: Tokenizer,
        sample_rate: int,
        frame_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
    ):
        """
        dataset class, manage and process .osz archives
        :param dataset_directory: location of .osz files to load
        :param sample_rate: sampling rate of audio file (samples/second)
        :param frame_size: samples per audio frame (samples/frame)
        :param loader: instance of Loader class
        :param src_seq_len: maximum length of source sequence
        :param tgt_seq_len: maximum length of target sequence
        """
        super().__init__()
        self.dataset = glob(f"{dataset_directory}/*{OSZ_FILE_EXTENSION}")
        self.dataset_index = {}
        self.loader = loader
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        random.shuffle(self.dataset)

    def _get_audio_and_osu(self, osz_path: str) -> tuple[npt.NDArray, list[str]]:
        """
        load an .osz archive and get its audio samples and .osu beatmap
        :param osz_path: path to the .osz archive
        :return audio_samples: audio time series
        :return osu_beatmap: a list of strings (osu beatmap data)
        """
        if osz_path in self.dataset_index:
            audio_samples, osu_beatmap = self.loader.load_osz_indexed(
                osz_path,
                self.dataset_index[osz_path],
            )
        else:
            audio_samples, osu_beatmap, osu_filename, _ = self.loader.load_osz(
                osz_path,
            )
            self.dataset_index[osz_path] = osu_filename

        return audio_samples, osu_beatmap

    def _get_frames(self, samples: npt.NDArray) -> list[float]:
        """
        segment audio samples into frames.
        calculate the time of each audio frame, in miliseconds.
        :param samples: audio time-series
        :return: audio frame times
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
        """
        tokenize Event objects and index them to audio frame times.
        tokenize every time shift as multiple single time steps.
        it should always be true that event_end_indices[i] = event_start_indices[i + 1].
        :param events: list of Event object lists
        :param event_times: time of each Event object list, in miliseconds
        :param frame_times: audio frame times, in miliseconds
        :return event_tokens: tokenized Events and time shifts
        :return event_start_indices: corresponding start event index for every audio frame
        :return event_end_indices: corresponding end event index for every audio frame
        """
        event_steps = [round(t * STEPS_PER_MILLISECOND) for t in event_times]
        time_step_token = self.tokenizer.time_step_id

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
                event_tokens.append(time_step_token)
                cur_step += 1
                fill_event_start_indices_to_cur_step()
                cur_event_idx = len(event_tokens)

            for event in event_list:
                token = self.tokenizer.encode(event)
                event_tokens.append(token)

        while cur_step / STEPS_PER_MILLISECOND <= frame_times[-1]:
            event_tokens.append(time_step_token)
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
        frames_per_split: int = 6,
    ) -> list[dict[npt.NDArray, list[int]]]:
        """
        create source and target sequences for training/testing.
        source: input to the transformer encoder
        target: input to the transformer decoder, ground truth
        :param event_tokens: tokenized Events and time shifts
        :param event_start_indices: corresponding start event index for every audio frame
        :param event_end_indices: corresponding end event index for every audio frame
        :param frames: audio frames
        :param frames_per_split: maximum number of frames in each split
        :return sequences: list of source and target sequences
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
            max_offset = len(split_frames) - self.src_seq_len
            if max_offset < 1:
                sequence_start_idx = 0
                sequence_end_idx = len(split_frames)
            else:
                sequence_start_idx = random.randint(0, max_offset)
                sequence_end_idx = sequence_start_idx + self.src_seq_len

            # Create the sequence
            sequence = {}
            sequence_event_starts = split_event_starts[
                sequence_start_idx:sequence_end_idx
            ]
            sequence_event_ends = split_event_ends[sequence_start_idx:sequence_end_idx]
            target_start_idx = sequence_event_starts[0]
            target_end_idx = sequence_event_ends[-1]
            sequence["source"] = split_frames[sequence_start_idx:sequence_end_idx]
            sequence["target"] = event_tokens[target_start_idx:target_end_idx]
            sequences.append(sequence)

        return sequences

    def _merge_time_step_tokens(self, sequence):
        total_time_shift = 0
        tokens = []
        is_redundant = False

        for token in sequence["target"]:
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

        sequence["target"] = tokens
        return sequence

    def _pad_sequence(self, sequence):
        source = torch.from_numpy(sequence["source"]).to(torch.float32)
        target = torch.tensor(sequence["target"], dtype=torch.long)
        if source.shape[0] < self.src_seq_len:
            pad = torch.zeros(
                self.src_seq_len - source.shape[0],
                source.shape[1],
                dtype=source.dtype,
                device=source.device,
            )
            source = torch.cat([source, pad], dim=0)
        if target.shape[0] < self.tgt_seq_len:
            eos = (
                torch.ones(1, dtype=target.dtype, device=target.device)
                * self.tokenizer.eos_id
            )
            if self.tgt_seq_len - target.shape[0] - 1 > 0:
                pad = (
                    torch.ones(
                        self.tgt_seq_len - target.shape[0] - 1,
                        dtype=target.dtype,
                        device=target.device,
                    )
                    * self.tokenizer.pad_id
                )
                target = torch.cat([target, eos, pad], dim=0)
            else:
                target = torch.cat([target, eos], dim=0)

        sequence["source"] = source
        sequence["target"] = target
        return sequence

    def _get_next(self) -> dict[int, int]:
        for osz_path in self.dataset:
            audio_samples, osu_beatmap = self._get_audio_and_osu(osz_path)

            frames, frame_times = self._get_frames(audio_samples)
            events, event_times = parser.parse_osu(osu_beatmap)

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
                sequence = self._pad_sequence(sequence)
                yield sequence

    def __iter__(self):
        """
        get a single sample from the dataset
        :return: 'src_seq_len' audio frames and 'tgt_seq_len' event tokens
        """
        return cycle(self._get_next())


class CustomLoader(object):
    """
    Here we iterate over the dataset, after drawing an index and loading the associated object,
    we concatenate it to the tensors we drew before (batch). We keep doing this until we reach the desired size,
    such that we can cut out and yield a batch. We retain the rows in batch, which we did not yield.
    Because it may be the case that a single instance exceeds the desired batch_size, we use a while loop.

    You could modify this minimal CustomDataloader to add more features in the style of PyTorch's dataloader.
    There is also no need to use a RandomSampler to draw in indices, others would work equally well.
    It would also be possible to avoid repeated concats, in case your data is large by using for example a list and keeping track of the cumulative length of its tensors.
    """

    def __init__(self, dataset, batch_size, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = RandomSampler(dataset)

    def __iter__(self):
        batch = torch.Tensor()
        for index in self.sampler:
            batch = torch.cat([batch, self.dataset[index]])
            while batch.size(0) >= self.batch_size:
                if batch.size(0) == self.batch_size:
                    yield batch
                    batch = torch.Tensor()
                else:
                    return_batch, batch = batch.split(
                        [self.batch_size, batch.size(0) - self.batch_size]
                    )
                    yield return_batch
        if batch.size(0) > 0 and not self.drop_last:
            yield batch


"""
mel_bins = 3
dim0sizes = torch.LongTensor(100).random_(1, 100)
print(dim0sizes)
print(dim0sizes.sum())
data = torch.randn(size=(dim0sizes.sum(), mel_bins))
tensors = torch.split(data, list(dim0sizes))
for x in tensors:
    print(x.shape)

ds = TaikoDataset(tensors)
train_size = int(0.8 * len(ds))
test_size = len(ds) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, test_size])

train_dataloader = CustomLoader(train_dataset, batch_size=256, drop_last=False)
test_dataloader = CustomLoader(test_dataset, batch_size=256, drop_last=False)
sum = 0
for c, i in enumerate(train_dataloader):
    print(c, i.size(0))
    sum += i.size(0)
for c, i in enumerate(test_dataloader):
    print(c, i.size(0))
    sum += i.size(0)

print(sum)
"""
