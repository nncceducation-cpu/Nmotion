"""Train and save the Nmotion classifier for live web prediction.

Two ways to run:

  # A) You already ran `python run.py --video-dir data/videos --classify`,
  #    which wrote output/dataframes/clip_features.csv:
  python train.py --clip-features output/dataframes/clip_features.csv

  # B) Start from raw labeled videos in data/videos/{class}/ — this runs
  #    flow extraction + clip feature extraction, then trains:
  python train.py --video-dir data/videos

The model is saved to models/nmotion_model.joblib. The web app (webapp/app.py)
auto-loads it on startup, so after training just restart the server and each
uploaded video gets a predicted class + probabilities.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from pipeline.predict import DEFAULT_MODEL_PATH, train_and_save

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nmotion.train")


def _clip_df_from_videos(video_dir: Path, output_dir: Path, device: str | None):
    """Run flow + clip feature extraction to build a clip DataFrame."""
    import torch
    from pipeline.augment import apply_augmentations
    from pipeline.clip_extract import extract_all_clips
    from pipeline.features import extract_clip_features
    from pipeline.flow_extract import extract_all_flows

    flow_dir = output_dir / "flows"
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Extracting optical flow on %s ...", dev)
    extract_all_flows(video_dir, output_dir, device=dev)

    logger.info("Extracting clips ...")
    clips, labels, video_ids, fps_values = extract_all_clips(flow_dir)

    logger.info("Augmenting clips ...")
    aug_c, aug_l, aug_v, aug_f = [], [], [], []
    for clip, label, vid, fps in zip(clips, labels, video_ids, fps_values):
        for aug in apply_augmentations(clip):
            aug_c.append(aug); aug_l.append(label); aug_v.append(vid); aug_f.append(fps)
    clips, labels, video_ids, fps_values = aug_c, aug_l, aug_v, aug_f

    logger.info("Extracting clip features from %d clips ...", len(clips))
    return extract_clip_features(clips, labels, video_ids, fps_values)


def main() -> None:
    p = argparse.ArgumentParser(description="Train + save the Nmotion classifier")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip-features", type=Path,
                     help="Path to a clip_features.csv produced by run.py --classify")
    src.add_argument("--video-dir", type=Path,
                     help="Root dir with labeled subfolders: {class}/*.mp4")
    p.add_argument("--output-dir", type=Path, default=Path("output"),
                   help="Working dir for flows (used with --video-dir)")
    p.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH,
                   help="Where to save the model bundle")
    p.add_argument("--device", default=None, help="cpu|cuda (with --video-dir)")
    args = p.parse_args()

    if args.clip_features:
        logger.info("Loading clip features from %s", args.clip_features)
        clip_df = pd.read_csv(args.clip_features)
    else:
        clip_df = _clip_df_from_videos(args.video_dir, args.output_dir, args.device)

    if clip_df.empty:
        raise SystemExit("No clip features available — nothing to train on.")

    n_classes = clip_df["group"].nunique()
    if n_classes < 2:
        raise SystemExit(
            f"Need >=2 classes to train; found {n_classes}. "
            "Add labeled videos to at least two class folders."
        )

    summary = train_and_save(clip_df, out_path=args.out)
    logger.info("Done. Classes: %s", summary["class_names"])
    logger.info("Model saved to %s", summary["path"])
    logger.info("Restart the web app to enable live predictions.")


if __name__ == "__main__":
    main()
