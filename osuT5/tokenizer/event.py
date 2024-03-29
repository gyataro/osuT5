from __future__ import annotations

import dataclasses
from enum import Enum


class EventType(Enum):
    TIME_SHIFT = "t"
    POINT = "p"
    NEW_COMBO = "new_combo"
    CIRCLE = "circle"
    SLIDER_BEZIER = "slide_b"
    SLIDER_CATMULI = "slide_c"
    SLIDER_LINEAR = "slide_l"
    SLIDER_PERFECT_CIRCLE = "slide_p"
    CONTROL_POINT = "cp"
    SLIDES = "slides"
    SPINNER = "spinner"
    SPINNER_LENGTH = "spinner_length"


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


def get_event_ranges() -> list[EventRange]:
    return [
        EventRange(EventType.TIME_SHIFT, 0, 512),
        EventRange(EventType.POINT, 0, 512),
        EventRange(EventType.CIRCLE, 0, 0),
        EventRange(EventType.NEW_COMBO, 0, 1),
        EventRange(EventType.SLIDER_BEZIER, 0, 0),
        EventRange(EventType.SLIDER_CATMULI, 0, 0),
        EventRange(EventType.SLIDER_LINEAR, 0, 0),
        EventRange(EventType.SLIDER_PERFECT_CIRCLE, 0, 0),
        EventRange(EventType.CONTROL_POINT, -256, 768),
        EventRange(EventType.SLIDES, 0, 100),
        EventRange(EventType.SPINNER, 0, 0),
        EventRange(EventType.SPINNER_LENGTH, 0, 1024),
    ]
