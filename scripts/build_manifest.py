import json
import hashlib
from pathlib import Path
import subprocess

# directory containing wav and txt files
DATA_DIR = Path("data/raw/en/wav")
# output manifest file
MANIFEST_PATH = Path("data/manifests/clean.jsonl")
# language code
LANG = "en"
# create directory if needed
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
# collect all wav files
wav_files = sorted(DATA_DIR.glob("*.wav"))

def compute_md5(file_path):
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()

def phonemize_text(text):
    """Convert text to phonemes using espeak-ng."""
    result = subprocess.run(
        ["espeak-ng", "-q", "--ipa=3", text],
        capture_output=True,
        text=True
    )
    return result.stdout.strip() #result.stdout.replace("\n", " ").strip()


with open(MANIFEST_PATH, "w") as manifest:
    for wav_path in wav_files:
        # filename without extension
        stem = wav_path.stem
        # stable utterance id
        utt_id = f"{LANG}_{stem}"
        # corresponding text file
        txt_path = wav_path.with_suffix(".txt")
        with open(txt_path) as f:
            text = f.read().strip()
        # phoneme transcription using espeak-ng
        phonemes = phonemize_text(text)
        # md5 checksum of the audio file
        audio_md5 = compute_md5(wav_path)
        entry = {
            "utt_id": utt_id,
            "lang": LANG,
            "wav_path": str(wav_path),
            "ref_text": text,
            "ref_phon": phonemes,
            "audio_md5": audio_md5
        }
        manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
print("Manifest created.")