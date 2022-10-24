import dataclasses
import math
from typing import Tuple
from enum import Enum


class EventType(Enum):
    TIME_SHIFT = "t"
    POS_X = "x"
    POS_Y = "y"
    CIRCLE = "circle"
    SLIDER = "slider"
    BEZIER = "bezier"
    CATMULI = "catmuli"
    LINEAR = "linear"
    PERFECT_CIRCLE = "perfect_circle"
    SLIDES = "slides"
    SPINNER_START = "spinner_start"
    SPINNER_END = "spinner_end"


@dataclasses.dataclass
class EventRange:
    type: EventType
    min_value: int
    max_value: int


@dataclasses.dataclass
class Event:
    type: EventType
    value: int = 0

    def __repr__(self) -> str:
        return f"{self.type.value}{self.value}"

    def __str__(self) -> str:
        return f"{self.type.value}{self.value}"


class Tokenizer:
    def __init__(self):
        """
        fixed vocabulary tokenizer
        maps Event objects into corresponding token ids and vice versa
        manages special tokens (EOS, PAD)
        """
        self._offset = 2
        self._event_ranges = [
            EventRange(EventType.TIME_SHIFT, 0, 800),
            EventRange(EventType.POS_X, -24, 536),
            EventRange(EventType.POS_Y, -24, 408),
            EventRange(EventType.CIRCLE, 0, 0),
            EventRange(EventType.SLIDER, 0, 0),
            EventRange(EventType.BEZIER, 0, 0),
            EventRange(EventType.CATMULI, 0, 0),
            EventRange(EventType.LINEAR, 0, 0),
            EventRange(EventType.PERFECT_CIRCLE, 0, 0),
            EventRange(EventType.SLIDES, 0, 20),
            EventRange(EventType.SPINNER_START, 0, 0),
            EventRange(EventType.SPINNER_END, 0, 0),
        ]

    @property
    def pad_id(self) -> int:
        """[PAD] token for padding"""
        return 0

    @property
    def eos_id(self) -> int:
        """[EOS] token for end-of-sequence"""
        return 1

    def decode(self, id: int) -> Event:
        offset = self._offset
        for er in self._event_ranges:
            if offset <= id <= offset + er.max_value - er.min_value:
                return Event(type=er.type, value=er.min_value + id - offset)
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"id {id} is not mapped to any event")

    def encode(self, event: Event) -> int:
        offset = self._offset
        for er in self._event_ranges:
            if event.type is er.type:
                if not er.min_value <= event.value <= er.max_value:
                    raise ValueError(
                        f"event value {event.value} is not within range "
                        f"[{er.min_value}, {er.max_value}] for event type {event.type}"
                    )
                return offset + event.value - er.min_value
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"unknown event type: {event.type}")

    def event_type_range(self, event_type: EventType) -> Tuple[int, int]:
        offset = self._offset
        for er in self._event_ranges:
            if event_type is er.type:
                return offset, offset + (er.max_value - er.min_value)
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"Unknown event type: {event_type}")

    def vocab_size(self) -> int:
        return self._offset + sum(
            er.max_value - er.min_value + 1 for er in self._event_ranges
        )


def n_tokens(tokenizer: Tokenizer) -> int:
    """Vocabulary size as a multiple of 128 for TPU efficiency."""
    return 128 * math.ceil(tokenizer.vocab_size() / 128)
