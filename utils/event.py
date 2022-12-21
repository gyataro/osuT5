from __future__ import annotations

import dataclasses
from enum import Enum


class EventType(Enum):
    TIME_SHIFT = "t"
    POINT = "p"
    CIRCLE = "circle"
    SLIDER_BEZIER = "slide_b"
    SLIDER_CATMULI = "slide_c"
    SLIDER_LINEAR = "slide_l"
    SLIDER_PERFECT_CIRCLE = "slide_p"
    CONTROL_POINT = "cp"
    SLIDES = "slides"


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
