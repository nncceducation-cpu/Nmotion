"""
Convert the trained Nmotion classifier (XGBoost bundle) to a Core ML model
for on-device inference in the iOS app.

Input : models/nmotion_model.joblib   (produced by train.py — a dict with
        'model' (xgboost.XGBClassifier), 'classes' (list[str]), and
        'feature_cols' (list[str] in the exact order the app must build).
Output: build/NmotionClassifier.mlmodel   +   build/nmotion_feature_cols.json

Run on a Mac (or any machine) with:
    pip install coremltools xgboost scikit-learn joblib
    python ios_convert_classifier_to_coreml.py --model models/nmotion_model.joblib

Then drag build/NmotionClassifier.mlmodel into the Xcode project and copy the
feature-column order from nmotion_feature_cols.json into the Swift classifier so
the on-device feature vector is assembled in the identical order.

NOTE: the iOS feature values MUST match the desktop feature battery (validate
with ios_export_feature_reference.py) or the model's predictions are meaningless.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="XGBoost Nmotion model -> Core ML")
    ap.add_argument("--model", type=Path, default=Path("models/nmotion_model.joblib"))
    ap.add_argument("--out-dir", type=Path, default=Path("build"))
    args = ap.parse_args()

    import joblib
    import coremltools as ct

    if not args.model.exists():
        raise SystemExit(
            f"No trained model at {args.model}. Train one first:\n"
            f"    python train.py --video-dir data/videos"
        )

    bundle = joblib.load(args.model)
    model = bundle["model"]
    classes = [str(c) for c in bundle["classes"]]
    feature_cols = list(bundle["feature_cols"])

    print(f"Classes      : {classes}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # coremltools converts the fitted XGBoost booster into a Core ML tree
    # ensemble classifier. feature_names must match how we feed inputs on-device.
    #
    # We expose a SINGLE multi-array input named "features" of length N so the
    # Swift side just fills one MLMultiArray in feature_cols order. To do that we
    # convert with per-feature names, then it is trivial to marshal in Swift.
    try:
        mlmodel = ct.converters.xgboost.convert(
            model.get_booster(),
            mode="classifier",
            feature_names=feature_cols,
            class_labels=classes,
            n_classes=len(classes),
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "coremltools XGBoost conversion failed. Common fixes:\n"
            "  - pip install 'coremltools>=7.0' 'xgboost>=1.7'\n"
            "  - ensure the model was fit (train.py) before export\n"
            f"Original error: {exc}"
        )

    mlmodel.short_description = "Nmotion neonatal movement classifier (research)"
    mlmodel.author = "Nmotion"
    mlmodel.license = "Research use only - not a medical device"
    # Surface class probabilities to the app.
    mlmodel.user_defined_metadata["classes"] = ",".join(classes)
    mlmodel.user_defined_metadata["feature_cols"] = ",".join(feature_cols)

    out_model = args.out_dir / "NmotionClassifier.mlmodel"
    mlmodel.save(str(out_model))

    cols_json = args.out_dir / "nmotion_feature_cols.json"
    cols_json.write_text(json.dumps(
        {"classes": classes, "feature_cols": feature_cols}, indent=2
    ))

    print(f"\nSaved: {out_model}")
    print(f"Saved: {cols_json}")
    print("\nNext: add the .mlmodel to Xcode, and paste feature_cols into")
    print("NmotionClassifier.swift so the on-device vector matches this order.")


if __name__ == "__main__":
    main()
