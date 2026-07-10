"""
Export the desktop feature battery for a set of clips so the iOS Swift port can
be unit-tested for numerical parity.

For each video it runs the SAME desktop pipeline the model is trained on
(flow -> compact -> feature battery) and writes one JSON row of feature name ->
value. In Xcode, feed the identical clip through the Swift FeatureExtractor and
assert each value matches within tolerance (~1e-3). If they match, the on-device
classifier is trustworthy; if not, fix the Swift math before shipping.

Usage (in the Nmotion project, same env as the desktop app):
    python ios_export_feature_reference.py --clips data/videos/normal/*.mp4 \
        --out build/feature_reference.json --device cpu
"""

from __future__ import annotations

import argparse
import glob
import json
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump desktop features for parity")
    ap.add_argument("--clips", nargs="+", required=True,
                    help="video paths or globs")
    ap.add_argument("--out", type=Path, default=Path("build/feature_reference.json"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import numpy as np
    from pipeline.flow_extract import extract_flow, _load_raft
    from pipeline.features import extract_features_single
    import torch

    paths: list[str] = []
    for pat in args.clips:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        raise SystemExit("No clips matched.")

    model = _load_raft(torch.device(args.device))
    rows = []
    for p in paths:
        p = Path(p)
        with tempfile.TemporaryDirectory() as td:
            flow_npy = Path(td) / "flow.npy"
            _, fps = extract_flow(p, flow_npy, device=args.device, model=model)
            flow = np.load(flow_npy)
            feats = extract_features_single(flow, fps=fps,
                                            video_name=p.stem, group="ref")
        clean = {k: (float(v) if isinstance(v, (int, float)) else v)
                 for k, v in feats.items()}
        rows.append({"clip": p.name, "features": clean})
        print(f"  {p.name}: {len(clean)} features")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {len(rows)} reference rows to {args.out}")
    print("Use these in an XCTest to assert Swift feature parity (tol ~1e-3).")


if __name__ == "__main__":
    main()
