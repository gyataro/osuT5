from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from omegaconf import DictConfig
from osuT5.tokenizer import Event, EventType, Tokenizer

MILISECONDS_PER_SECOND = 1000
MILISECONDS_PER_STEP = 10


class Pipeline(object):
    def __init__(self, args: DictConfig, tokenizer: Tokenizer):
        """Model inference stage that processes sequences."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.batch_size = args.batch_size
        self.tgt_seq_len = args.model.max_target_len
        self.frame_seq_len = args.model.max_seq_len - 1
        self.frame_size = args.model.spectrogram.hop_length
        self.sample_rate = args.model.spectrogram.sample_rate
        self.samples_per_sequence = self.frame_seq_len * self.frame_size
        self.miliseconds_per_sequence = (
            self.samples_per_sequence * MILISECONDS_PER_SECOND / self.sample_rate
        )

    def generate(
        self, model: nn.Module, sequences: tuple[torch.Tensor]
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
            n = len(sources)
            unfinished = torch.LongTensor([[1] for _ in range(n)])
            targets = torch.LongTensor([[self.tokenizer.sos_id] for _ in range(n)])
            targets = targets.to(self.device)
            sources = sources.to(self.device)
            encoder_outputs = None

            for _ in tqdm(range(self.tgt_seq_len - 1), leave=False):
                out = model.forward(
                    frames=sources,
                    decoder_input_ids=targets,
                    encoder_outputs=encoder_outputs,
                )
                encoder_outputs = out.encoder_outputs
                logits = out.logits
                logits = logits[:, -1, :]
                logits = self._filter(logits, 0.9)
                probabilities = F.softmax(logits, dim=-1)
                tokens = torch.multinomial(probabilities, 1)

                # change next tokens of finished sentences to PAD token
                tokens = tokens.cpu() * unfinished + 0 * (1 - unfinished)
                targets = torch.cat([targets, tokens.to(self.device)], dim=-1)

                # check if any sentence in batch has reached EOS, mark as finished
                eos_in_sentence = tokens == self.tokenizer.eos_id
                unfinished.mul_((~eos_in_sentence).long())

                # stop preemptively when all sentences have finished
                if unfinished.max() == 0:
                    break

            for seq_index, target in enumerate(targets):
                index = batch_index * self.batch_size + seq_index
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

            try:
                event = self.tokenizer.decode(token.item())
            except:
                continue

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
