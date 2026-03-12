import csv
import subprocess
from pathlib import Path

from pathlib import Path

# path of this script
SCRIPT_DIR = Path(__file__).resolve().parent

# project root (one level above scripts)
PROJECT_ROOT = SCRIPT_DIR.parent

DATASET_DIR = PROJECT_ROOT / "sps-corpus-1.0-2025-11-25-en"
AUDIO_DIR = DATASET_DIR / "audios"

OUT_DIR = PROJECT_ROOT / "data/raw/en"
WAV_DIR = OUT_DIR / "wav"
WAV_DIR.mkdir(parents=True, exist_ok=True)

TSV_PATH = DATASET_DIR / "ss-corpus-en.tsv"
# we only take a small subset of the dataset for the pipeline
N = 20
saved = 0
# open the metadata file
with open(TSV_PATH) as f:
    # read the tsv file using tab as separator
    reader = csv.DictReader(f, delimiter="\t")
    # go through the rows of the dataset
    for row in reader:
        if saved >= N:
            break
        text = row["transcription"].strip()
        # skip rows with empty transcription
        if not text:
            continue
        # name of the mp3 file
        mp3_name = row["audio_file"].strip()
        # full path to the mp3 file
        mp3_path = AUDIO_DIR / mp3_name

        # names for the output files
        wav_name = f"utt_{saved:04d}.wav"
        txt_name = f"utt_{saved:04d}.txt"

        wav_path = WAV_DIR / wav_name
        txt_path = WAV_DIR / txt_name

        # convert mp3 to wav using ffmpeg
        # -ac 1 makes it mono
        # -ar 16000 sets the sample rate to 16kHz
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i",
            str(mp3_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path)
        ])
        # save the transcription in a text file
        with open(txt_path, "w") as out:
            out.write(text)
        saved += 1

print("subset created")
