from __future__ import annotations

import os
import io
import zipfile
from typing import BinaryIO

import numpy as np
import numpy.typing as npt
from pydub import AudioSegment

OSU_FILE_EXTENSION = ".osu"
OSZ_AUDIO_FILENAME = "audio.mp3"
MIN_DIFFICULTY = 0
MAX_DIFFICULY = 10


class OszLoader(object):
    def __init__(
        self,
        sample_rate: int = 16000,
        min_difficulty: float = 0,
        max_difficulty: float = 10,
        mode: str = "center",
        **kwargs,
    ):
        """Load .osz archives.

        Attributes:
            sample_rate: Sampling rate of audio file (samples/second).
            min_difficulty: Consider all .osu files above and equal to this difficulty.
            max_difficulty: Consider all .osu files below and equal to this difficulty.
            mode: The criteria on which an .osu file is selected.

                `min`: pick the .osu file with lowest difficulty, within range.
                `max`: pick the .osu file with highest difficulty, within range.
                `center`: pick the .osu file with a difficulty value closest to (max + min)/2, within range.
                `keyword`: pick the first .osu file with matching keyword in filename, within range.

            keywords: Used in 'keyword', list of keywords to consider.
        """
        self.sample_rate = sample_rate
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.mode = mode
        self.keywords = kwargs.get("keywords", [""])

        if not self.min_difficulty <= self.max_difficulty:
            raise ValueError(f"max_range must be larger or equal to min_range")

        if not MIN_DIFFICULTY <= self.min_difficulty <= MAX_DIFFICULY:
            raise ValueError(
                f"min_range must be in range [{MIN_DIFFICULTY}, {MAX_DIFFICULY}]"
            )

        if not MIN_DIFFICULTY <= self.max_difficulty <= MAX_DIFFICULY:
            raise ValueError(
                f"max_range must be in range [{MIN_DIFFICULTY}, {MAX_DIFFICULY}]"
            )

        if self.mode not in ["min", "max", "center", "keyword"]:
            raise ValueError(f"mode {self.mode} is not supported")

    def load_osz(self, path: str) -> tuple[npt.NDArray, list[str], str, float]:
        """Load an .osz archive and extract its audio and the target .osu file.

        Args:
            path: Path to the .osz archive.

        Returns:
            audio_data: Audio time series.
            osu_data: The .osu beatmap data as a list of strings.
            osu_filename: The selected .osu file.
            osu_difficulty: The difficulty value of .osu file.
        """
        audio_data = None
        osu_file = None
        osu_candidates = []

        with zipfile.ZipFile(path) as z:
            for filename in z.namelist():
                if filename == OSZ_AUDIO_FILENAME:
                    with z.open(filename) as f:
                        audio_data = self._load_mp3(f)
                        f.close()

                _, extension = os.path.splitext(filename)

                if extension == OSU_FILE_EXTENSION:
                    with io.TextIOWrapper(z.open(filename), encoding="utf-8") as f:
                        lines = f.read().splitlines()
                        for line in lines:
                            if line.startswith("OverallDifficulty"):
                                values = line.split(":")
                                difficulty = float(values[-1])
                                if (
                                    self.min_difficulty
                                    <= difficulty
                                    <= self.max_difficulty
                                ):
                                    candidate = {
                                        "data": lines,
                                        "filename": filename,
                                        "difficulty": difficulty,
                                    }
                                    osu_candidates.append(candidate)

                                break

                        f.close()

            z.close()

        if len(osu_candidates) <= 0:
            raise ValueError(
                f"no .osu files in difficulty range [{self.min_difficulty}, {self.max_difficulty}]"
            )

        if self.mode == "min":
            osu_file = min(osu_candidates, key=lambda x: x["difficulty"])

        elif self.mode == "max":
            osu_file = max(osu_candidates, key=lambda x: x["difficulty"])

        elif self.mode == "center":
            center = (self.min_difficulty + self.max_difficulty) / 2
            osu_file = min(osu_candidates, key=lambda x: abs(x["difficulty"] - center))

        elif self.mode == "keyword":

            def match_any(filename, keywords):
                for keyword in keywords:
                    if keyword in filename:
                        return True
                return False

            for osu_candidate in osu_candidates:
                if match_any(osu_candidate["filename"], self.keywords):
                    osu_file = osu_candidate
                    break

        if audio_data is None or osu_file is None:
            raise ValueError(f"no audio or .osu file matching criteria")

        return (
            audio_data,
            osu_file["data"],
            osu_file["filename"],
            osu_file["difficulty"],
        )

    def _load_mp3(self, file: BinaryIO) -> npt.ArrayLike:
        """Load .mp3 file as a numpy time-series array

        The signals are resampled, converted to mono channel, and normalized.

        Args:
            file: Python file object.
        Returns:
            samples: Audio time series.
        """
        audio = AudioSegment.from_file(file, format="mp3")
        audio = audio.set_frame_rate(self.sample_rate)
        audio = audio.set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples *= 1.0 / np.max(np.abs(samples))
        return samples
