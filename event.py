from __future__ import annotations

import dataclasses
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
