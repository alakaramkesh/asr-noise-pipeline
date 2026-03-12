# ASR Pipeline

This project implements an experimental pipeline to evaluate the robustness of an Automatic Speech Recognition (ASR) phoneme model under different noise conditions.

The pipeline processes a speech dataset, adds synthetic noise at several signal-to-noise ratios (SNR), performs phoneme prediction using a pretrained Wav2Vec2 model, and evaluates performance using the Phoneme Error Rate (PER).

## Pipeline Overview

The workflow is implemented as a DVC pipeline with the following stages:

1. **prepare_subset**
   - Extract a small subset of the dataset.
   - Convert audio files to mono WAV with 16 kHz sampling rate.

2. **build_manifest**
   - Create a manifest describing each utterance.
   - Each entry contains metadata such as text, phonemes, and checksum.

3. **add_noise**
   - Generate noisy versions of each audio file for multiple SNR levels.

4. **predict_phonemes**
   - Run inference with the pretrained model:
   - `facebook/wav2vec2-lv-60-espeak-cv-ft`

5. **evaluate_per**
   - Compute Phoneme Error Rate (PER) using edit distance.

6. **plot_per_curve**
   - Generate a performance curve showing PER as a function of noise level.

## Pipeline Structure

## Prerequisites

Before running the pipeline, make sure you have:

- DVC installed
- the `python` command available in your PATH

## Running the Pipeline

This project requires the `espeak-ng` phonemizer to generate reference phoneme sequences.
On Ubuntu / WSL:

```bash
sudo apt install espeak-ng
```
Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
To run the pipeline:
```bash
dvc repro
```

## Results

The final output includes:

per.json – PER metrics for each SNR level


per_curve.png – performance curve showing PER vs noise level

