import json
from pathlib import Path
import editdistance
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
with open(PROJECT_ROOT / "params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)
MANIFEST_DIR = PROJECT_ROOT / "data/manifests"
OUTPUT_METRICS = PROJECT_ROOT / "data/results/per.json"

def main():
    # find all prediction manifests
    prediction_manifests = sorted(MANIFEST_DIR.glob("predictions_*db.jsonl"))
    if not prediction_manifests:
        raise FileNotFoundError("No prediction manifests found in data/manifests")
    # store one result per noise level
    results = []
    for manifest_path in prediction_manifests:
        # total number of edit operations (S + D + I)
        total_distance = 0
        # total number of reference phonemes (N)
        total_ref = 0
        # open prediction manifest
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                # reference phoneme sequence
                ref = entry["ref_phon"].replace("\n", " ").split()
                # predicted phoneme sequence
                pred = entry["pred_phon"].split()
                # compute edit distance between reference and prediction
                dist = editdistance.eval(ref, pred)
                # accumulate total edit operations
                total_distance += dist
                # accumulate total number of reference phonemes
                total_ref += len(ref)
        # compute PER for this noise level
        per = total_distance / total_ref
        # extract snr value from the first entry of the file name
        snr_db = manifest_path.stem.replace("predictions_", "").replace("db", "")
        snr_db = int(snr_db)
        results.append({
            "snr_db": snr_db,
            "total_phonemes": total_ref,
            "total_edits": total_distance,
            "per": per
        })
        print(f"{manifest_path.name}")
        print("Total phonemes:", total_ref)
        print("Total edits:", total_distance)
        print("PER:", per)
        print()

    # sort results by SNR
    results.sort(key=lambda x: x["snr_db"])
    OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"PER metrics saved to: {OUTPUT_METRICS}")

if __name__ == "__main__":
    main()