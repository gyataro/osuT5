from __future__ import annotations
import os
import uuid
import shutil
import pathlib
import tempfile
import zipfile
import dataclasses
from string import Template

from omegaconf import DictConfig

from osuT5.tokenizer import Event, EventType

OSZ_FILE_EXTENSION = ".osz"
OSU_FILE_EXTENSION = ".osu"
OSU_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "template.osu")


@dataclasses.dataclass
class BeatmapConfig:
    # General
    audio_filename: str = ""

    # Metadata
    title: str = ""
    title_unicode: str = ""
    artist: str = ""
    artist_unicode: str = ""

    # Difficulty
    hp_drain_rate: float = 5
    circle_size: float = 4
    overall_difficulty: float = 8
    approach_rate: float = 9
    slider_multiplier: float = 1.8

    # Timing
    offset: int = 0
    beat_length: int = 500


class Postprocessor(object):
    def __init__(self, args: DictConfig):
        """Postprocessing stage that converts a list of Event objects to a beatmap file."""
        self.curve_types = {
            EventType.SLIDER_BEZIER: "B",
            EventType.SLIDER_CATMULI: "C",
            EventType.SLIDER_LINEAR: "L",
            EventType.SLIDER_PERFECT_CIRCLE: "P",
        }

        self.output_path = args.output_path
        self.audio_path = args.audio_path
        self.beatmap_config = BeatmapConfig(
            title=str(args.title).encode("ascii", "ignore"),
            artist=str(args.artist).encode("ascii", "ignore"),
            title_unicode=str(args.title),
            artist_unicode=str(args.artist),
            beat_length=float(60000 / args.bpm),
            audio_filename=f"audio{pathlib.Path(args.audio_path).suffix}",
        )

    def generate(self, events: list[list[Event]], event_times: list[int], output_path):
        """Generate a beatmap file.

        Args:
            events: List of Event object lists.
            event_times: Corresponding event times of Event object lists in miliseconds.
            output_path: Generated .osz file location

        Returns:
            None. An .osz file will be generated.
        """

        hit_object_strings = []

        # Convert to .osu format
        for hit_object, timestamp in zip(events, event_times):
            if len(hit_object) < 3:
                continue

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

        # Write .osu file
        with open(OSU_TEMPLATE_PATH, "r") as tf:
            template = Template(tf.read())
            hit_objects = {"hit_objects": "\n".join(hit_object_strings)}
            beatmap_config = dataclasses.asdict(self.beatmap_config)
            result = template.safe_substitute({**beatmap_config, **hit_objects})

            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy audio file to directory
                osz_audio_path = os.path.join(
                    temp_dir, self.beatmap_config.audio_filename
                )
                shutil.copy(self.audio_path, osz_audio_path)

                # Write .osu file to directory
                osz_osu_path = os.path.join(temp_dir, f"beatmap{OSU_FILE_EXTENSION}")
                osu_file = open(osz_osu_path, "w")
                osu_file.write(result)
                osu_file.close()

                # Compress directory and create final .osz file
                osz_output_path = os.path.join(
                    self.output_path, f"{str(uuid.uuid4().hex)}{OSZ_FILE_EXTENSION}"
                )
                osz_file = zipfile.ZipFile(osz_output_path, "w")
                osz_file.write(osz_audio_path)
                osz_file.write(osz_osu_path)
                osz_file.close()
