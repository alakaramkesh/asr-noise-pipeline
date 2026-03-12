import numpy as np
import soundfile as sf


def add_noise(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),
        size=signal.shape,
    )
    return signal + noise


def add_noise_to_file(
    input_wav: str,
    output_wav: str,
    snr_db: float,
    seed: int | None = None,
) -> None:
    signal, sr = sf.read(input_wav)
    if signal.ndim != 1:
        raise ValueError("Only mono audio is supported")

    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)

    sf.write(output_wav, noisy_signal, sr)

import json
from pathlib import Path

INPUT_MANIFEST = "data/manifests/clean.jsonl"
OUTPUT_MANIFEST = "data/manifests/noisy.jsonl"

# SNR level for the noise we add to the signal
SNR_LEVEL = 10
# read all entries from the clean manifest
with open(INPUT_MANIFEST) as f:
    lines = [json.loads(l) for l in f]

with open(OUTPUT_MANIFEST, "w") as out:
    # process each entry from the clean manifest
    for entry in lines:
        # path to the original wav file
        wav_path = entry["wav_path"]
        # extract the wav filename 
        wav_name = Path(wav_path).name
        # create the path where the noisy wav will be stored
        noisy_path = f"data/noisy/{wav_name}"
        # make sure the noisy directory exists
        Path("data/noisy").mkdir(parents=True, exist_ok=True)
        # create a noisy version of the audio file
        add_noise_to_file(
            wav_path,
            noisy_path,
            snr_db=SNR_LEVEL,
            seed=42  # fixed seed for reproducibility
        )
        # copy the original manifest entry
        new_entry = entry.copy()
        # update the wav path to point to the noisy audio
        new_entry["wav_path"] = noisy_path
        # store the SNR value used to generate the noise
        new_entry["snr_db"] = SNR_LEVEL
        # write the updated entry into the new manifest
        out.write(json.dumps(new_entry) + "\n")

# small message to confirm the script finished
print("Noisy manifest created")