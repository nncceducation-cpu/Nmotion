"""
OPTION A ONLY — convert RAFT-Large optical flow to Core ML so on-device features
match the desktop RAFT pipeline exactly.

Skip this if the app uses Apple's Vision optical flow (Option B in the spec).

Run on a Mac with a recent PyTorch + coremltools:
    pip install torch torchvision coremltools
    python ios_convert_flow_to_coreml.py --size 384 640 --iters 6

Output: build/RaftFlow.mlpackage   (input: two RGB frames; output: [2,H,W] flow)

Caveats (read the spec):
  * RAFT uses iterative refinement and correlation volumes. We wrap it in a
    fixed-iteration, fixed-size module so the graph traces cleanly. Some torch
    ops may still need coremltools op fallbacks; if conversion fails on an op,
    lower --iters and/or --size, or use Option B (Vision) instead.
  * Fewer iterations + smaller size = faster on-device but slightly different
    flow; re-check feature parity after any change.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="RAFT -> Core ML")
    ap.add_argument("--size", type=int, nargs=2, default=[384, 640],
                    metavar=("H", "W"), help="fixed inference size (mult of 8)")
    ap.add_argument("--iters", type=int, default=6, help="RAFT refinement iters")
    ap.add_argument("--out", type=Path, default=Path("build/RaftFlow.mlpackage"))
    args = ap.parse_args()

    import torch
    import coremltools as ct
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

    H, W = args.size
    if H % 8 or W % 8:
        raise SystemExit("H and W must be multiples of 8 for RAFT.")

    weights = Raft_Large_Weights.C_T_SKHT_V2
    raft = raft_large(weights=weights).eval()

    class FixedRaft(torch.nn.Module):
        """Fixed-size, fixed-iteration wrapper returning the final flow field."""
        def __init__(self, model: torch.nn.Module, iters: int):
            super().__init__()
            self.model = model
            self.iters = iters

        def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
            # inputs: [1,3,H,W] in [0,1]; RAFT wants [-1,1]
            a = img1 * 2.0 - 1.0
            b = img2 * 2.0 - 1.0
            flows = self.model(a, b, num_flow_updates=self.iters)
            return flows[-1]  # [1,2,H,W]

    wrapper = FixedRaft(raft, args.iters).eval()
    ex1 = torch.rand(1, 3, H, W)
    ex2 = torch.rand(1, 3, H, W)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (ex1, ex2))

    print(f"Tracing OK. Converting to Core ML at {H}x{W}, iters={args.iters} ...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="frame1", shape=(1, 3, H, W)),
            ct.TensorType(name="frame2", shape=(1, 3, H, W)),
        ],
        compute_units=ct.ComputeUnit.ALL,   # CPU+GPU+ANE where possible
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.short_description = f"RAFT-Large optical flow ({H}x{W}, {args.iters} iters)"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.out))
    print(f"Saved: {args.out}")
    print("Feed frames resized to exactly this HxW; output is dense [1,2,H,W] flow.")


if __name__ == "__main__":
    main()
