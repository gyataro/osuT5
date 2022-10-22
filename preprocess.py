import numpy as np
import numpy.typing as npt

import config
from tokenizer import Event, EventType


def parse_osu(path: str) -> tuple[list[list[Event]], list[int]]:
    """
    parse an .osu beatmap, each hit object is parsed into a list of Event objects
    :param chart: path to an .osu file.
    :return: list Event object lists, and their corresponding event times

    example:
    64,80,11000,1,0                                       -> circle
    256,192,11200,8,0,12000                               -> spinner
    100,100,12600,2,0,B|200:200|250:200|250:200|300:150,2 -> slider

    parsed events:
    [
        [x64, y80, circle],
        [spinner_start],
        [spinner_end],
        [x100, y100, slider, bezier, x200, y200, x250, y200, x260, y200, x300, y150, slides2]
    ]

    parsed event times:
    [
        11000,
        11200,
        12000
        12600
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    parsing = False
    events = []
    event_times = []

    for line in lines:
        if line == "[HitObjects]":
            parsing = True
            continue
        if not parsing:
            continue
        else:
            elements = line.split(",")
            type = int(elements[3])
            if type & 1:
                circle, time = parse_circle(elements)
                events.append(circle)
                event_times.append(time)
            elif type & 2:
                slider, time = parse_slider(elements)
                events.append(slider)
                event_times.append(time)
            elif type & 8:
                start_spinner, end_spinner, start_time, end_time = parse_spinner(
                    elements
                )
                events.append(start_spinner)
                events.append(end_spinner)
                event_times.append(start_time)
                event_times.append(end_time)

    return events, event_times


def parse_circle(elements: list[str]) -> tuple[list[Event], int]:
    """
    parse a circle hit object
    :param elements: list of strings extracted from .osu file
    :return events: a list of events, format: [x, y, circle]
    :return time: time when circle is to be hit, in miliseconds from the beginning of the beatmap's audio
    """
    pos_x = int(elements[0])
    pos_y = int(elements[1])
    time = int(elements[2])

    events = [
        Event(EventType.POS_X, pos_x),
        Event(EventType.POS_Y, pos_y),
        Event(EventType.CIRCLE),
    ]

    return events, time


def parse_slider(elements: list[str]) -> tuple[list[Event], int]:
    """
    parse a slider hit object
    :param elements: list of strings extracted from .osu file
    :return events: a list of events, format: [x, y, slider, curve_type, curve_points, slides]
    :return time: time when slider is to be dragged, in miliseconds from the beginning of the beatmap's audio
    """
    curve_types = {
        "B": EventType.BEZIER,
        "C": EventType.CATMULI,
        "L": EventType.LINEAR,
        "P": EventType.PERFECT_CIRCLE,
    }

    pos_x = int(elements[0])
    pos_y = int(elements[1])
    time = int(elements[2])

    events = [
        Event(EventType.POS_X, pos_x),
        Event(EventType.POS_Y, pos_y),
        Event(EventType.SLIDER),
    ]

    curve = elements[5].split("|")
    curve_type = curve_types[curve[0]]
    events.append(Event(curve_type))

    for curve_point in curve[1:]:
        curve_point = curve_point.split(":")
        curve_x = curve_point[0]
        curve_y = curve_point[1]
        events.append(Event(EventType.POS_X, curve_x))
        events.append(Event(EventType.POS_Y, curve_y))

    slides = int(elements[6])
    events.append(Event(EventType.SLIDES, slides))

    return events, time


def parse_spinner(elements: list[str]) -> tuple[list[Event], list[Event], int, int]:
    """
    parse a spinner hit object
    :param elements: list of strings extracted from .osu file
    :return start_event: a list of events, format: []
    :return time: time when slider is to start, in miliseconds from the beginning of the beatmap's audio
    """
    start_time = int(elements[2])
    end_time = int(elements[5])
    start_event = [Event(EventType.SPINNER_START)]
    end_event = [Event(EventType.SPINNER_END)]

    return start_event, end_event, start_time, end_time
