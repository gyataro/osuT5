import dataclasses
import logging
from typing import Tuple
from enum import Enum


class EventType(Enum):
    TIME_SHIFT = "t"
    POS_X = "x"
    POS_Y = "y"
    COMBO = "combo"
    CIRCLE = "circle"
    SLIDER = "slider"
    BEZIER = "bezier"
    CATMULI = "catmuli"
    LINEAR = "linear"
    PERFECT_CIRCLE = "perfectcircle"
    SLIDES = "slides"
    SPINNER = "spinner"


@dataclasses.dataclass
class EventRange:
    type: EventType
    min_value: int
    max_value: int


@dataclasses.dataclass
class Event:
    type: EventType
    value: int


class Tokenizer:
    def __init__(self):
        """
        fixed vocabulary tokenizer
        converts Event objects into corresponding token ids and vice versa
        manages special tokens (EOS, PAD)
        """
        self._event_ranges = [
            EventRange(EventType.TIME_SHIFT, 0, 799),
            EventRange(EventType.POS_X, 0, 511),
            EventRange(EventType.POS_Y, 0, 383),
            EventRange(EventType.COMBO, 0, 0),
            EventRange(EventType.CIRCLE, 0, 0),
            EventRange(EventType.SLIDER, 0, 0),
            EventRange(EventType.BEZIER, 0, 0),
            EventRange(EventType.CATMULI, 0, 0),
            EventRange(EventType.LINEAR, 0, 0),
            EventRange(EventType.PERFECT_CIRCLE, 0, 0),
            EventRange(EventType.SLIDES, 0, 19),
            EventRange(EventType.SPINNER, 0, 0),
        ]

    @property
    def pad_id(self) -> int:
        "[PAD] token for padding"
        return 0

    @property
    def eos_id(self) -> int:
        "[EOS] token for end-of-sequence"
        return 1

    def decode(self, id: int) -> Event:
        offset = 2
        for er in self._event_ranges:
            if offset <= id <= offset + er.max_value - er.min_value:
                return Event(type=er.type, value=er.min_value + id - offset)
            offset += er.max_value - er.min_value + 1

    def encode(self, event: Event) -> int:
        offset = 2
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
        offset = 2
        for er in self._event_ranges:
            if event_type is er.type:
                return offset, offset + (er.max_value - er.min_value)
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"Unknown event type: {event_type}")
