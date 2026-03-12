import json
import os
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import yaml
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# load parameters from params.yaml
with open(PROJECT_ROOT / "params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

MANIFEST_DIR = PROJECT_ROOT / "data/manifests"
MODEL_NAME = params["model"]["name"]


def read_and_validate_audio(wav_path: Path):
    # read waveform and sampling rate
    signal, sr = sf.read(wav_path)
    # expected parameters from params.yaml
    expected_sr = params["audio"]["sample_rate"]
    expected_channels = params["audio"]["channels"]
    if expected_channels == 1 and signal.ndim != 1:
        raise ValueError(f"{wav_path} is not mono")
    # check sample rate
    if sr != expected_sr:
        raise ValueError(
            f"{wav_path} has sample rate {sr}, expected {expected_sr}"
        )
    return signal


def predict_phonemes(signal, processor, model, device):
    """
    Run inference on a single waveform and return the predicted phoneme string.
    """
    # prepare the input for the wav2vec2 model
    inputs = processor(
        signal,
        sampling_rate=params["audio"]["sample_rate"],
        return_tensors="pt"
    )
    # move tensor to device
    input_values = inputs.input_values.to(device)
    # run the model without computing gradients
    with torch.no_grad():
        logits = model(input_values).logits
    # take the most probable token at each timestep
    predicted_ids = torch.argmax(logits, dim=-1)
    # convert token IDs to phoneme string
    pred_phon = processor.batch_decode(predicted_ids)[0].strip()

    return pred_phon


def main():
    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load processor and model once
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    noisy_manifests = sorted(MANIFEST_DIR.glob("noisy_*db.jsonl"))
    if not noisy_manifests:
        raise FileNotFoundError("No noisy manifests found in data/manifests")
    for input_manifest in noisy_manifests:
        # create matching output name
        output_name = input_manifest.name.replace("noisy_", "predictions_")
        output_manifest = MANIFEST_DIR / output_name
        # create a temporary file for atomic writing
        fd, tmp_name = tempfile.mkstemp(
            prefix=output_manifest.stem + "_",
            suffix=".jsonl.tmp",
            dir=str(MANIFEST_DIR),
        )
        os.close(fd)

        try:
            with open(input_manifest, "r", encoding="utf-8") as in_file, \
                 open(tmp_name, "w", encoding="utf-8") as out_file:
                for line in in_file:
                    entry = json.loads(line)
                    # build full path to the wav file
                    wav_path = PROJECT_ROOT / entry["wav_path"]
                    # load and validate audio (security check)
                    signal = read_and_validate_audio(wav_path)
                    pred_phon = predict_phonemes(
                        signal=signal,
                        processor=processor,
                        model=model,
                        device=device,
                    )
                    # keep all original fields
                    pred_entry = dict(entry)
                    # add predicted phoneme sequence
                    pred_entry["pred_phon"] = pred_phon
                    # write updated entry to output manifest
                    out_file.write(json.dumps(pred_entry, ensure_ascii=False) + "\n")
            os.replace(tmp_name, output_manifest)
            print(f"Prediction manifest created: {output_manifest}")
        except Exception:
            # if an error occurs, remove temporary file
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise


if __name__ == "__main__":
    main()