# osuTransformer

osuTransformer is a transformer-based encoder-decoder that can learn to convert spectrogram inputs to osu! hit-object output events. The goal of this project is to create a model that automatically generates beatmaps for any song.

This project is heavily inspired by Google Magenta's [MT3](https://github.com/magenta/mt3), quoting their [paper](https://magenta.tensorflow.org/transcription-with-transformers):

> This sequence-to-sequence approach simplifies transcription by jointly modeling audio features and language-like output dependencies, thus removing the need for task-specific architectures.

## Overview

The high-level overview of the model's input-output is as follow:

![Picture2](https://user-images.githubusercontent.com/28675590/201044116-1384ad72-c540-44db-a285-7319dd01caad.svg)

The model uses Mel spectrogram frames as encoder input, with one frame per input position. The model decoder output at each step is a softmax distribution over a discrete, predefined, vocabulary of events. Outputs are sparse, events are only needed when a hit-object occurs, instead of annotating every single audio frame.

## Usage

Coming soon.

## Training

The instructions below creates a training environment on your local machine. Alternatively, [Kaggle](https://www.kaggle.com/code/gernyataro/osutransformer-public/notebook) and [Colab](https://colab.research.google.com/drive/1V4WwZKlzQfqznFiEgw4lR04mjpriFKCC?usp=sharing) notebooks are provided.

### 1. Clone the Repository

Clone the repo and create a Python virtual environment.

```sh
git clone https://github.com/gyataro/osuTransformer.git
cd osuTransformer
python -m venv .venv
```

### 2. Install Dependencies

1. Install [ffmpeg](http://www.ffmpeg.org/). This is required for reading audio files.
2. Install [PyTorch](https://pytorch.org/get-started/locally/). Pick the latest stable version based on your operating system, your package manager. Select any of the proposed CUDA versions if GPU is present, otherwise select CPU. Run the given command.
3. Install the remaining Python dependencies.

```sh
pip install -r requirements.txt
```

### 3. Download Dataset

The [dataset](https://www.kaggle.com/datasets/gernyataro/osu-beatmap-dataset) is available on Kaggle. You can also prepare your own dataset.

```sh
kaggle datasets download -d gernyataro/osu-beatmap-dataset
```

### 4. Configure Parameters and Begin Training

All configurations are located in `./config/config.yaml`. Begin training by calling `train.py`.

```sh
python train.py
```

## Related Works

1. [osu! Beatmap Generator](https://github.com/Syps/osu_beatmap_generator) by Syps (Nick Sypteras)
2. [osumapper](https://github.com/kotritrona/osumapper) by kotritrona, jyvden, Yoyolick (Ryan Zmuda)
