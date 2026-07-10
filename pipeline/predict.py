"""Train a final classifier and run per-video prediction.

The CV routine in ``pipeline.classify`` measures accuracy but never persists
a model. This module adds the two missing pieces for a live web UI:

  * ``train_and_save`` — fit ONE XGBoost on all clip features and save a
    self-contained bundle (model + label list + feature columns) with joblib.
  * ``load_model`` / ``predict_video`` — load that bundle and predict on a
    single uploaded video: the video's flow is cut into clips, each clip is
    scored, and the per-clip probabilities are averaged into a video-level
    result.

NOTE (clinical): output is a research classification, not a medical
diagnosis. It is only as good as the labeled data the model was trained on.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline.classify import _get_feature_cols, compute_class_weights
from pipeline.clip_extract import (
    DEFAULT_OVERLAP,
    DEFAULT_WINDOW_SECONDS,
    extract_clips,
)
from pipeline.features import extract_clip_features

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/nmotion_model.joblib")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_save(
    clip_df: pd.DataFrame,
    out_path: Path = DEFAULT_MODEL_PATH,
    feature_cols: Optional[List[str]] = None,
) -> Dict:
    """Fit a final XGBoost on ALL clip features and save the bundle.

    Args:
        clip_df: One row per clip, with a ``group`` label column (and ideally
            ``video_id``). Produced by ``run.py --classify`` at
            ``output/dataframes/clip_features.csv``.
        out_path: Where to write the joblib bundle.
        feature_cols: Override feature columns; defaults to the baseline set
            plus any ``mse_scale_*`` columns present.

    Returns:
        Dict summary: class_names, n_features, n_samples, feature_cols.
    """
    import joblib
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder

    if "group" not in clip_df.columns:
        raise ValueError("clip_df must contain a 'group' label column.")
    if feature_cols is None:
        feature_cols = _get_feature_cols(clip_df)
    feature_cols = [c for c in feature_cols if c in clip_df.columns]
    if not feature_cols:
        raise ValueError("No usable feature columns found in clip_df.")

    df = clip_df[clip_df[feature_cols].notna().any(axis=1)].copy()
    X = df[feature_cols].fillna(0).values

    le = LabelEncoder()
    y = le.fit_transform(df["group"])

    weight_map = compute_class_weights(df["group"])
    sample_weight = np.array([weight_map[g] for g in df["group"]])

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42, verbosity=0,
    )
    model.fit(X, y, sample_weight=sample_weight)

    bundle = {
        "model": model,
        "classes": list(le.classes_),
        "feature_cols": feature_cols,
        "window_seconds": DEFAULT_WINDOW_SECONDS,
        "overlap": DEFAULT_OVERLAP,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    logger.info(
        "Saved model to %s — %d classes %s, %d features, %d clips",
        out_path, len(le.classes_), list(le.classes_), len(feature_cols), len(df),
    )
    return {
        "class_names": list(le.classes_),
        "n_features": len(feature_cols),
        "n_samples": int(len(df)),
        "feature_cols": feature_cols,
        "path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def load_model(path: Path = DEFAULT_MODEL_PATH):
    """Load a saved bundle, or return None if the file does not exist."""
    import joblib
    path = Path(path)
    if not path.exists():
        return None
    return joblib.load(path)


def predict_video(flow: np.ndarray, fps: float, bundle: Dict) -> Dict:
    """Predict a single video's class from its dense optical flow.

    Cuts the flow into clips, scores each, and averages per-clip class
    probabilities into a video-level result.

    Returns:
        Dict: ``label`` (str), ``probabilities`` ({class: prob}),
        ``n_clips`` (int), ``per_clip_labels`` (list[str]).
    """
    feature_cols = bundle["feature_cols"]
    classes = bundle["classes"]
    model = bundle["model"]
    window_seconds = bundle.get("window_seconds", DEFAULT_WINDOW_SECONDS)
    overlap = bundle.get("overlap", DEFAULT_OVERLAP)

    window_frames = max(1, int(window_seconds * (fps or 30.0)))
    clips = extract_clips(np.asarray(flow), window_frames, overlap)

    n = len(clips)
    clip_df = extract_clip_features(
        clips,
        labels=["uploaded"] * n,
        video_ids=["uploaded"] * n,
        fps_values=[fps] * n,
    )
    if clip_df.empty:
        raise ValueError("Could not extract clip features from this video.")

    X = clip_df.reindex(columns=feature_cols).fillna(0).values
    proba = model.predict_proba(X)              # [n_clips, n_classes]
    mean_proba = proba.mean(axis=0)             # video-level

    order = list(model.classes_)                # encoded ints in model order
    prob_map = {classes[i]: float(mean_proba[j]) for j, i in enumerate(order)}
    label = max(prob_map, key=prob_map.get)
    per_clip = [classes[order[j]] for j in proba.argmax(axis=1)]

    return {
        "label": label,
        "probabilities": prob_map,
        "n_clips": int(n),
        "per_clip_labels": per_clip,
    }
