import csv
from pathlib import Path
import numpy as np

RANDOM_SEED = 42
N_SAMPLES = 1200
LABEL_NOISE_RATE = 0.04  # fraction of labels randomly flipped

# Resolve paths relative to the project root (parent of this file's src/ dir),
# so the script works regardless of the terminal's current directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

rng = np.random.default_rng(RANDOM_SEED)


def clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate():
    # Latent "student ability/discipline" factor per student (drives correlations)
    ability = rng.normal(loc=0.55, scale=0.2, size=N_SAMPLES)
    ability = clip(ability, 0, 1)

    # Previous semester GPA (0-4 scale, typical university scale)
    previous_semester_gpa = clip(
        ability * 4.0 + rng.normal(0, 0.4, N_SAMPLES), 0, 4.0
    )

    # Attendance (0-10) - correlated with ability, own noise
    attendance = clip(
        ability * 7 + 2 + rng.normal(0, 1.0, N_SAMPLES), 0, 10
    )

    # CT marks (0-10) - correlated with ability AND attendance
    ct_marks = clip(
        ability * 6 + (attendance / 10) * 2 + rng.normal(0, 1.2, N_SAMPLES),
        0, 10,
    )

    # Mid marks (0-30) - correlated with ability, ct performance, own noise
    mid_marks = clip(
        ability * 18 + (ct_marks / 10) * 8 + rng.normal(0, 3.0, N_SAMPLES),
        0, 30,
    )

    # Study hours per week (0-40) - another realistic, semi-independent feature
    study_hours = clip(
        ability * 15 + rng.normal(8, 4, N_SAMPLES), 0, 40
    )

    # Weighted "true" score with its own noise term (unseen effort/luck factor)
    raw_score = (
        2.5 * attendance                   # attendance out of 25 (10 * 2.5)
        + 1.5 * ct_marks                   # ct out of 15
        + 1.0 * mid_marks                  # mid out of 30
        + 6.0 * previous_semester_gpa      # gpa out of 24
        + 0.4 * study_hours                # study hours out of 16
        + rng.normal(0, 6, N_SAMPLES)      # unexplained variance
    )

    threshold = np.percentile(raw_score, 42)  # roughly ~58% pass, realistic-ish
    result = (raw_score >= threshold).astype(int)

    # Inject label noise to simulate real-world inconsistency
    flip_mask = rng.random(N_SAMPLES) < LABEL_NOISE_RATE
    result[flip_mask] = 1 - result[flip_mask]

    rows = []
    for i in range(N_SAMPLES):
        rows.append([
            round(float(attendance[i]), 1),
            round(float(ct_marks[i]), 1),
            round(float(mid_marks[i]), 1),
            round(float(previous_semester_gpa[i]), 2),
            round(float(study_hours[i]), 1),
            int(result[i]),
        ])
    return rows


def main():
    rows = generate()
    header = [
        "attendance",            # 0-10
        "ct_marks",              # 0-10
        "mid_marks",             # 0-30
        "previous_semester_gpa", # 0-4.0
        "study_hours_per_week",  # 0-40
        "result",                # 0 = Fail, 1 = Pass
    ]
    out_path = DATA_DIR / "dataset.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    passes = sum(r[-1] for r in rows)
    print(f"Dataset written to {out_path}")
    print(f"Total rows: {len(rows)} | Pass: {passes} ({passes/len(rows)*100:.1f}%) "
          f"| Fail: {len(rows)-passes} ({(len(rows)-passes)/len(rows)*100:.1f}%)")


if __name__ == "__main__":
    main()
