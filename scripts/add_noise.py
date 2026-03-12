import json
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
# project root
PROJECT_ROOT = SCRIPT_DIR.parent
# load parameters from params.yaml
with open(PROJECT_ROOT / "params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)
# input clean manifest
INPUT_MANIFEST = PROJECT_ROOT / "data/manifests/clean.jsonl"
# output directories
NOISY_AUDIO_DIR = PROJECT_ROOT / "data/noisy"
NOISY_MANIFEST_DIR = PROJECT_ROOT / "data/manifests"

# noise parameters
SNR_LEVELS = params["noise"]["snr_levels"]
SEED = params["noise"]["seed"]


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
    input_wav: Path,
    output_wav: Path,
    snr_db: float,
    seed: int,
) -> None:
    signal, sr = sf.read(input_wav)
    if signal.ndim != 1:
        raise ValueError(f"{input_wav} is not mono")
    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, noisy_signal, sr)


def main():
    # create output directories 
    NOISY_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    NOISY_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(INPUT_MANIFEST, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    # create one noisy manifest per SNR level
    for snr_db in SNR_LEVELS:
        output_manifest = NOISY_MANIFEST_DIR / f"noisy_{snr_db}db.jsonl"
        # atomic writing
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"noisy_{snr_db}db_",
            suffix=".jsonl.tmp",
            dir=str(NOISY_MANIFEST_DIR),
        )
        os.close(fd)

        try:
            with open(tmp_name, "w", encoding="utf-8") as out_file:
                for entry in entries:
                    input_wav = PROJECT_ROOT / entry["wav_path"]
                    # keep the same wav filename
                    wav_name = Path(entry["wav_path"]).name
                    # store noisy files in an SNR-specific directory
                    output_wav = NOISY_AUDIO_DIR / f"snr_{snr_db}" / wav_name
                    add_noise_to_file(
                        input_wav=input_wav,
                        output_wav=output_wav,
                        snr_db=snr_db,
                        seed=SEED,
                    )
                    # copy original manifest entry
                    new_entry = dict(entry)
                    # update wav path to point to the noisy file
                    new_entry["wav_path"] = str(output_wav.relative_to(PROJECT_ROOT))
                    # store SNR value used for this noisy variant
                    new_entry["snr_db"] = snr_db
                    # write updated entry into the noisy manifest
                    out_file.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
            # replace temp manifest with final manifest
            os.replace(tmp_name, output_manifest)
            print(f"Noisy manifest created: {output_manifest}")
        except Exception:
            # remove temporary manifest if something goes wrong
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise


if __name__ == "__main__":
    main()