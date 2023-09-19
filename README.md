# osuT5

Try the model [here](https://colab.research.google.com/drive/1HJhyPwhf4uBJt4zbk2BwfXoVU_6mpmfW?usp=sharing). Check out a video showcase [here](https://youtu.be/HNlEVQiAvCA).

osuT5 is a transformer-based encoder-decoder that uses spectrogram inputs to generate osu! hit-object output events. The goal of this project is to automatically generate osu! beatmaps from any song.

This project is heavily inspired by Google Magenta's [MT3](https://github.com/magenta/mt3), quoting their [paper](https://magenta.tensorflow.org/transcription-with-transformers):

> This sequence-to-sequence approach simplifies transcription by jointly modeling audio features and language-like output dependencies, thus removing the need for task-specific architectures.

## Overview

The high-level overview of the model's input-output is as follow:

![Picture2](https://user-images.githubusercontent.com/28675590/201044116-1384ad72-c540-44db-a285-7319dd01caad.svg)

The model uses Mel spectrogram frames as encoder input, with one frame per input position. The model decoder output at each step is a softmax distribution over a discrete, predefined, vocabulary of events. Outputs are sparse, events are only needed when a hit-object occurs, instead of annotating every single audio frame.

## Inference

The instruction below allows you to generate beatmaps on your local machine.

### 1. Clone the Repository

Clone the repo and create a Python virtual environment. Activate the virtual environment.

```sh
git clone https://github.com/gyataro/osuTransformer.git
cd osuTransformer
python -m venv .venv
```

### 2. Install Dependencies

Install [ffmpeg](http://www.ffmpeg.org/), [PyTorch](https://pytorch.org/get-started/locally/), and the remaining Python dependencies.

```sh
pip install -r requirements.txt
```

### 3. Download Model

Download the latest model from the [release](https://github.com/gyataro/osuT5/releases) section.

### 4. Begin Inference

Run `inference.py` and pass in some arguments to generate beatmaps.
```
python -m inference \
  model_path   [PATH TO DOWNLOADED MODEL] \
  audio_path   [PATH TO INPUT AUDIO] \
  output_path  [PATH TO OUTPUT DIRECTORY] \
  bpm          [BEATS PER MINUTE OF INPUT AUDIO] \
  offset       [START OF BEAT, IN MILISECONDS, FROM THE BEGINNING OF INPUT AUDIO] \
  title        [SONG TITLE] \
  artist       [SONG ARTIST]
```

Example:
```
python -m inference 
  model_path="./osuT5_model.bin" \ 
  audio_path="./song.mp3" \
  output_path="./output" \
  bpm=120 \
  offset=0 \
  title="A Great Song" \
  artist="A Great Artist"
```

## Training

The instruction below creates a training environment on your local machine.

### 1. Clone the Repository

Clone the repo and create a Python virtual environment. Activate the virtual environment.

```sh
git clone https://github.com/gyataro/osuTransformer.git
cd osuTransformer
python -m venv .venv
```

### 2. Install Dependencies

Install [ffmpeg](http://www.ffmpeg.org/), [PyTorch](https://pytorch.org/get-started/locally/), and the remaining Python dependencies.

```sh
pip install -r requirements.txt
```

### 3. Download Dataset

The [dataset](https://www.kaggle.com/datasets/gernyataro/osu-beatmap-dataset) is available on Kaggle. You can also prepare your own dataset.

```sh
kaggle datasets download -d gernyataro/osu-beatmap-dataset
```

### 4. Configure Parameters and Begin Training

All configurations are located in `./configs/train.yaml`. Begin training by calling `train.py`.

```sh
python train.py
```

## Credits

Special thanks to:
1. The authors of [nanoT5](https://github.com/PiotrNawrot/nanoT5/tree/main) for their T5 training code.
2. Hugging Face team for their [tools](https://huggingface.co/docs/transformers/index). 
3. The osu! community for the beatmaps.

## Related Works

1. [osu! Beatmap Generator](https://github.com/Syps/osu_beatmap_generator) by Syps (Nick Sypteras)
2. [osumapper](https://github.com/kotritrona/osumapper) by kotritrona, jyvden, Yoyolick (Ryan Zmuda)
3. [osu! Diffusion](https://github.com/OliBomby/osu-diffusion) by OliBomby (Olivier Schipper), NiceAesth (Andrei Baciu)
