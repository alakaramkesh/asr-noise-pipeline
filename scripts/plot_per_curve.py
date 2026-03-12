import json
from pathlib import Path
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_METRICS = PROJECT_ROOT / "data/results/per.json"
OUTPUT_FIGURE = PROJECT_ROOT / "data/results/per_curve.png"


def main():
    with open(INPUT_METRICS, "r", encoding="utf-8") as f:
        results = json.load(f)

    snr_values = [entry["snr_db"] for entry in results]
    per_values = [entry["per"] for entry in results]

    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(snr_values, per_values, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("PER")
    plt.title("PER as a function of noise level")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=200)

    print(f"Figure saved to: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()