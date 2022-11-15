# osuTransformer

osuTransformer is a genertic encoder-decoder Transformer model that can learn to translate spectrogram inputs to osu! hit-object output events. The goal of this project is to create a model that automatically generates beatmaps for any song.

This project is heavily inspired by Google Magenta's [MT3](https://github.com/magenta/mt3), quoting their [paper](https://magenta.tensorflow.org/transcription-with-transformers):

> This sequence-to-sequence approach simplifies transcription by jointly modeling audio features and language-like output dependencies, thus removing the need for task-specific architectures.

## Overview
The high-level overview of the model's input-output is as follow:

![Picture2](https://user-images.githubusercontent.com/28675590/201044116-1384ad72-c540-44db-a285-7319dd01caad.svg)

## Usage
Coming soon.

## Training
### Training on Kaggle/Colab
1. Kaggle: coming soon.
2. Colab: coming soon.

### Training Locally
1. Clone the repository.
   ```sh
   git clone https://github.com/gyataro/osuTransformer.git
   ```
2. Setup a Python virtual environment (optional, but recommended).
   ```sh
   python -m venv .venv
   ```
3. Install dependencies. 
   ```sh
   pip install -r requirements.txt
   ```
4. Modify training configurations and begin training.
   ```sh
   python train.py
   ```

## Related Works

1. [osu! Beatmap Generator](https://github.com/Syps/osu_beatmap_generator) by Syps (Nick Sypteras)
2. [osumapper](https://github.com/Syps/osu_beatmap_generator) by kotritrona, jyvden, Yoyolick (Ryan Zmuda)
