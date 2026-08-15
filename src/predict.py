import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd

FEATURES = [
    "attendance", "ct_marks", "mid_marks",
    "previous_semester_gpa", "study_hours_per_week",
]

RANGES = {
    "attendance": (0, 10, "Attendance (0-10)"),
    "ct_marks": (0, 10, "CT Marks (0-10)"),
    "mid_marks": (0, 30, "Mid Marks (0-30)"),
    "previous_semester_gpa": (0, 4.0, "Previous Semester GPA (0-4.0)"),
    "study_hours_per_week": (0, 40, "Study Hours per Week (0-40)"),
}

# Resolve paths relative to the project root, so this works regardless of
# the terminal's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def load_artifacts():
    try:
        model = joblib.load(MODELS_DIR / "best_model.joblib")
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    except FileNotFoundError:
        print("Model not found. Run `python src/train_model.py` first.")
        sys.exit(1)
    return model, scaler


def prompt_value(key):
    lo, hi, label = RANGES[key]
    while True:
        raw = input(f"{label}: ").strip()
        try:
            val = float(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if not (lo <= val <= hi):
            print(f"  Value must be between {lo} and {hi}.")
            continue
        return val


def get_inputs_interactive():
    print("Enter student data:")
    return {k: prompt_value(k) for k in FEATURES}


def get_inputs_from_args(args):
    values = {
        "attendance": args.attendance,
        "ct_marks": args.ct,
        "mid_marks": args.mid,
        "previous_semester_gpa": args.gpa,
        "study_hours_per_week": args.study,
    }
    for key, val in values.items():
        lo, hi, label = RANGES[key]
        if not (lo <= val <= hi):
            print(f"Error: {label} must be between {lo} and {hi} (got {val}).")
            sys.exit(1)
    return values


def predict(model, scaler, values: dict):
    X = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURES)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1] if hasattr(model, "predict_proba") else None
    return pred, proba


def main():
    parser = argparse.ArgumentParser(description="Predict student Pass/Fail.")
    parser.add_argument("--attendance", type=float)
    parser.add_argument("--ct", type=float)
    parser.add_argument("--mid", type=float)
    parser.add_argument("--gpa", type=float)
    parser.add_argument("--study", type=float)
    args = parser.parse_args()

    model, scaler = load_artifacts()

    if all(v is not None for v in [args.attendance, args.ct, args.mid, args.gpa, args.study]):
        values = get_inputs_from_args(args)
    else:
        values = get_inputs_interactive()

    pred, proba = predict(model, scaler, values)

    print("\n" + "=" * 40)
    if pred == 1:
        print(f"Prediction: PASS ✅  (confidence: {proba*100:.1f}%)")
    else:
        print(f"Prediction: FAIL ❌  (confidence: {(1-proba)*100:.1f}%)")
    print("=" * 40)


if __name__ == "__main__":
    main()
