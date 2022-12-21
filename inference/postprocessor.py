from __future__ import annotations
import uuid
import dataclasses
from string import Template

from inference.config import Config, BeatmapMetadata, BeatmapDifficulty, BeatmapTiming
from utils.event import Event, EventType

OSU_FILE_EXTENSION = ".osu"
OSU_TEMPLATE_PATH = "./template.osu"


class Postprocessor(object):
    def __init__(self, config: Config):
        """Postprocessing stage that converts a list of Event objects to a beatmap file."""
        self.curve_types = {
            EventType.SLIDER_BEZIER: "B",
            EventType.SLIDER_CATMULI: "C",
            EventType.SLIDER_LINEAR: "L",
            EventType.SLIDER_PERFECT_CIRCLE: "P",
        }

    def generate(
        self,
        events: list[list[Event]],
        event_times: list[int],
        metadata: BeatmapMetadata,
        difficulty: BeatmapDifficulty,
        timing: BeatmapTiming,
    ):
        """Generate a beatmap file.

        Args:
            events: List of Event object lists.
            event_times: Corresponding event times of Event object lists in miliseconds.
            metadata: Beatmap metadata
            difficulty: Beatmap difficulty configuration
            timing: Beatmap timing configuration

        Returns:
            None. An .osu file with a random UUID filename will be generated.
        """

        hit_object_strings = []

        for hit_object, timestamp in zip(events, event_times):
            x = hit_object[0].value
            y = hit_object[1].value
            hit_type = hit_object[2].type

            if hit_type == EventType.CIRCLE:
                hit_object_strings.append(f"{x},{y},{timestamp},1,0")

            elif hit_type in self.curve_types:
                curve_type = self.curve_types[hit_type]
                slides = 1
                control_points = ""

                if hit_object[-1].type == EventType.SLIDES:
                    slides = hit_object[-1].value

                for i in range(3, len(hit_object) - 1, 2):
                    if (
                        hit_object[i].type == EventType.CONTROL_POINT
                        and hit_object[i + 1].type == EventType.CONTROL_POINT
                    ):
                        control_points += (
                            f"|{hit_object[i].value}:{hit_object[i+1].value}"
                        )

                hit_object_strings.append(
                    f"{x},{y},{timestamp},2,0,{curve_type}{control_points},{slides}"
                )

        with open(OSU_TEMPLATE_PATH, "r") as tf:
            template = Template(tf.read())
            metadata = dataclasses.asdict(metadata)
            difficulty = dataclasses.asdict(difficulty)
            timing = dataclasses.asdict(timing)
            hit_objects = {"hit_objects": "\n".join(hit_object_strings)}

            result = template.safe_substitute(
                {**metadata, **difficulty, **timing, **hit_objects}
            )

            with open(f"{uuid.uuid4().hex}{OSU_FILE_EXTENSION}", "w") as f:
                f.write(result)
                f.close()

            tf.close()
