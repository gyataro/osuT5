from __future__ import annotations

from osuT5.tokenizer import Event, EventType

STEPS_PER_MILLISECOND = 0.1


class OsuParser:
    def __init__(self) -> None:
        pass

    def parse(self, beatmap: list[str]) -> tuple[list[list[Event]], list[int]]:
        """Parse an .osu beatmap.

        Each hit object is parsed into a list of Event objects.
        Each list of Event objects is appended into a list, in order of its
        appearance in the beatmap. In other words, in ascending order of time.

        Args:
            beatmap: List of strings parsed from an .osu file.

        Returns:
            events: List of Event object lists.
            event_times: Corresponding event times of Event object lists in miliseconds.

        Example::
            >>> beatmap = [
                "64,80,11000,1,0",
                "100,100,12600,2,0,B|200:200|250:200|250:200|300:150,2"
            ]
            >>> event, event_times = parse_osu(beatmap)
            >>> print(event)
            [
                [p64, p80, circle],
                [p100, p100, slider_b, cp200, cp200, cp250, cp200, cp260, cp200, cp300, cp150, slides2]
            ]
            >>> print(event_times)
            [
                11000,
                12600
            ]
        """
        parsing = False
        events = []
        event_times = []

        for line in beatmap:
            if line == "[HitObjects]":
                parsing = True
                continue
            if not parsing:
                continue
            else:
                elements = line.split(",")
                type = int(elements[3])
                if type & 1:
                    circle, time = self._parse_circle(elements)
                    events.append(circle)
                    event_times.append(time)
                elif type & 2:
                    slider, time = self._parse_slider(elements)
                    events.append(slider)
                    event_times.append(time)
                elif type & 8:
                    spinner, time = self._parse_spinner(elements)
                    events.append(spinner)
                    event_times.append(time)

        return events, event_times

    def _parse_circle(self, elements: list[str]) -> tuple[list[Event], int]:
        """Parse a circle hit object.

        Args:
            elements: List of strings extracted from .osu file.

        Returns:
            events: List of events, format: [x, y, new_combo, circle].
            time: Time when circle is to be hit, in miliseconds from the beginning of the beatmap's audio.
        """
        new_combo = min(1, int(elements[3]) & 4)
        pos_x = int(elements[0])
        pos_y = int(elements[1])
        time = int(elements[2])

        events = [
            Event(EventType.POINT, pos_x),
            Event(EventType.POINT, pos_y),
            Event(EventType.NEW_COMBO, new_combo),
            Event(EventType.CIRCLE),
        ]

        return events, time

    def _parse_slider(self, elements: list[str]) -> tuple[list[Event], int]:
        """Parse a slider hit object.

        Args:
            elements: List of strings extracted from .osu file.

        Returns:
            events: A list of events, format: [x, y, new_combo, slider_type, curve_points, slides].
            time: Time when slider is to be dragged, in miliseconds from the beginning of the beatmap's audio.
        """
        curve_types = {
            "B": EventType.SLIDER_BEZIER,
            "C": EventType.SLIDER_CATMULI,
            "L": EventType.SLIDER_LINEAR,
            "P": EventType.SLIDER_PERFECT_CIRCLE,
        }

        new_combo = min(1, int(elements[3]) & 4)
        pos_x = int(elements[0])
        pos_y = int(elements[1])
        time = int(elements[2])

        events = [
            Event(EventType.POINT, pos_x),
            Event(EventType.POINT, pos_y),
            Event(EventType.NEW_COMBO, new_combo),
        ]

        curve = elements[5].split("|")
        curve_type = curve_types[curve[0].capitalize()]
        events.append(Event(curve_type))

        for curve_point in curve[1:]:
            curve_point = curve_point.split(":")
            curve_x = abs(int(curve_point[0]))
            curve_y = abs(int(curve_point[1]))
            events.append(Event(EventType.CONTROL_POINT, curve_x))
            events.append(Event(EventType.CONTROL_POINT, curve_y))

        slides = int(elements[6])
        events.append(Event(EventType.SLIDES, slides))

        return events, time

    def _parse_spinner(self, elements: list[str]) -> tuple[list[Event], int]:
        """Parse a slider hit object.

        Args:
            elements: List of strings extracted from .osu file.

        Returns:
            events: A list of events, format: [x, y, new_combo, spinner].
            time: Time when slider is to be dragged, in miliseconds from the beginning of the beatmap's audio.
        """
        new_combo = min(1, int(elements[3]) & 4)
        pos_x = 256
        pos_y = 192
        start_time = int(elements[2])
        end_time = int(elements[5])
        length = round((end_time - start_time) * STEPS_PER_MILLISECOND)

        events = [
            Event(EventType.POINT, pos_x),
            Event(EventType.POINT, pos_y),
            Event(EventType.NEW_COMBO, new_combo),
            Event(EventType.SPINNER),
            Event(EventType.SPINNER_LENGTH, length),
        ]

        return events, start_time
