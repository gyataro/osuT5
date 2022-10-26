import os
import io
import zipfile

import numpy.typing as npt
import librosa

OSU_FILE_EXTENSION = ".osu"
MIN_DIFFICULTY = 0
MAX_DIFFICULY = 10


def load_osz(
    path: str,
    audio_filename: str,
    sample_rate: int,
    min_range: float,
    max_range: float,
    mode: str,
    **kwargs,
) -> tuple[npt.NDArray, list[str], str, float]:
    """
    load an .osz archive and extract its audio and the target .osu file
    :param path: path to the .osz archive
    :param audio_filename: filename of audio file
    :param sample_rate: the sampling rate of input audio (samples/second)
    :param min_range: consider all .osu files above and equal to this difficulty
    :param max_range: consider all .osu files below and equal to this difficulty
    :param mode: the criteria on which an .osu file is selected
        - 'min': pick the .osu file with lowest difficulty, within range
        - 'max': pick the .osu file with highest difficulty, within range
        - 'center': pick the .osu file with a difficulty value closest to (max + min)/2, within range
        - 'keyword': pick the first .osu file with matching keyword in filename, within range

    :param keywords: used in 'keyword', list of keywords to consider
    :return audio_data: audio time series
    :return osu_data: .osu beatmap data as a list of strings
    :return osu_filename: the selected .osu file
    :return osu_difficulty: the difficulty value of .osu file
    """
    audio_data = None
    osu_file = None
    osu_candidates = []

    if not min_range <= max_range:
        raise ValueError(f"max_range must be larger or equal to min_range")

    if not MIN_DIFFICULTY <= min_range <= MAX_DIFFICULY:
        raise ValueError(
            f"min_range must be in range [{MIN_DIFFICULTY}, {MAX_DIFFICULY}]"
        )

    if not MIN_DIFFICULTY <= max_range <= MAX_DIFFICULY:
        raise ValueError(
            f"max_range must be in range [{MIN_DIFFICULTY}, {MAX_DIFFICULY}]"
        )

    if mode not in ["min", "max", "center", "keyword"]:
        raise ValueError(f"mode {mode} is not supported")

    with zipfile.ZipFile(path) as z:
        for filename in z.namelist():
            if filename == audio_filename:
                with z.open(filename) as f:
                    audio_data, _ = librosa.load(f, sr=sample_rate, mono=True)
                    f.close()

            _, extension = os.path.splitext(filename)

            if extension == OSU_FILE_EXTENSION:
                with io.TextIOWrapper(z.open(filename), encoding="utf-8") as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        if line.startswith("OverallDifficulty"):
                            values = line.split(":")
                            difficulty = float(values[-1])
                            if min_range <= difficulty <= max_range:
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
            f"no .osu files in difficulty range [{min_range}, {max_range}]"
        )

    if mode == "min":
        osu_file = min(osu_candidates, key=lambda x: x["difficulty"])

    elif mode == "max":
        osu_file = max(osu_candidates, key=lambda x: x["difficulty"])

    elif mode == "center":
        center = (max_range + min_range) / 2
        osu_file = min(osu_candidates, key=lambda x: abs(x["difficulty"] - center))

    elif mode == "keyword":

        def match_any(filename, keywords):
            for keyword in keywords:
                if keyword in filename:
                    return True
            return False

        keywords = kwargs.get("keywords", [""])
        for osu_candidate in osu_candidates:
            if match_any(osu_candidate["filename"], keywords):
                osu_file = osu_candidate
                break

    if audio_data is None or osu_file is None:
        raise ValueError(f"no audio or .osu file matching criteria")

    return audio_data, osu_file["data"], osu_file["filename"], osu_file["difficulty"]


def load_osz_indexed(
    path: str, audio_filename: str, sample_rate: int, osu_filename: str
) -> tuple[npt.NDArray, list[str]]:
    """
    load an .osz archive and extract its audio and the target .osu file by filename
    :param path: path to the .osz archive
    :param audio_filename: filename of audio file
    :param sample_rate: the sampling rate of input audio (samples/second)
    :return audio_data: audio time series
    :return osu_data: a list of strings (osu beatmap data)
    """
    audio_data = None
    osu_data = None

    with zipfile.ZipFile(path) as z:
        for filename in z.namelist():
            if filename == audio_filename:
                with z.open(filename) as f:
                    audio_data, _ = librosa.load(f, sr=sample_rate, mono=True)
                    f.close()

            elif filename == osu_filename:
                with io.TextIOWrapper(z.open(filename), encoding="utf-8") as f:
                    osu_data = f.read().splitlines()
                    f.close()

    if audio_data is None or osu_data is None:
        raise ValueError(f"no audio or .osu file matching criteria")

    return audio_data, osu_data
