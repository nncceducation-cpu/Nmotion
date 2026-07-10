# Nmotion — Deploy & process video (web upload)

This adds a **web upload UI** on top of the existing `pipeline/` package.
Drop in a neonatal video → it runs RAFT optical-flow extraction → computes the
~90 movement features → shows a movement-intensity plot and the feature table
in your browser. Everything runs **locally**; video never leaves your machine.

## Recommended: run locally with Docker

Local Docker is the recommendation here. The pipeline has heavy ML dependencies
(PyTorch/RAFT, OpenCV, XGBoost) and Docker bundles them + `ffmpeg` so it "just
runs" the same way on any machine. Running locally also keeps sensitive medical
video off the cloud (see Privacy note below).

**Prerequisite:** Docker Desktop installed and running.

```bash
cd Nmotion
docker compose up --build
```

Then open http://localhost:8000 and drop in a video.

- First run downloads the RAFT model weights (~20 MB) into a cached volume, so
  later runs are faster.
- Processed jobs (flow, plot, features) persist in the `nmotion_runs` volume.
- Stop with `Ctrl+C`, or `docker compose down`.

### Tuning (docker-compose.yml → environment)

| Var | Default | Meaning |
|-----|---------|---------|
| `NMOTION_MAX_FRAMES` | `240` | Frames processed per video. RAFT on CPU is slow, so the app trims to this many frames. Set `0` for the full video. |
| `NMOTION_MAX_WIDTH` | `640` | Downscale width in px before flow extraction. `0` keeps native resolution. |
| `NMOTION_DEVICE` | auto | Force `cpu` or `cuda`. Defaults to auto-detect. |

> **CPU speed:** on CPU, RAFT processes roughly a few frames/sec, so the 240-frame
> default (~8–10 s of 30 fps video) takes ~1–3 min. Raise the caps once you know it
> works; use a GPU for full clips.

### GPU (optional, much faster)

The image installs **CPU-only** PyTorch by default to keep it small. For an
NVIDIA GPU, rebuild against the CUDA wheels and expose the GPU:

```bash
docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
# then add to the nmotion service in docker-compose.yml:
#   deploy:
#     resources:
#       reservations:
#         devices: [{capabilities: ["gpu"]}]
# and set NMOTION_DEVICE=cuda
```
(Requires the NVIDIA Container Toolkit on the host.)

## Run without Docker (Python 3.11)

```bash
cd Nmotion
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r webapp/requirements-web.txt
uvicorn webapp.app:app --host 0.0.0.0 --port 8000
```
On Windows you also need `ffmpeg` on PATH (e.g. `winget install Gyan.FFmpeg`).

## Deploying to the cloud (optional)

The same container runs on any host that accepts a Dockerfile — Render,
Railway, Fly.io, or a cloud VM. Two caveats:

1. **Cost/latency:** CPU-only cloud instances are slow for RAFT; pick a GPU
   instance or keep the frame caps low.
2. **Privacy:** neonatal clinical video is sensitive personal health data.
   Only upload to infrastructure that is authorized for it (BAA / DPA, access
   controls, encryption). When in doubt, keep it local.

## Enabling live diagnosis prediction (optional)

Out of the box the web app runs in **features-only** mode — it shows movement
features + a plot, but no class label, because the repo ships without a trained
model or labeled training videos.

To turn on per-video prediction:

1. **Put labeled videos** in `data/videos/<class>/`, e.g. `normal/`,
   `hypertonia/`, `spasms/` (at least two classes, several clips each).
2. **Train and save a model:**
   ```bash
   python train.py --video-dir data/videos
   # or, if you already ran run.py --classify:
   python train.py --clip-features output/dataframes/clip_features.csv
   ```
   This writes `models/nmotion_model.joblib`.
3. **Restart the web app** (`docker compose up` — the `models/` folder is
   mounted into the container). Uploaded videos now show a **Predicted class**
   card with per-class probabilities, averaged over the video's clips.

> **Not a medical device.** The prediction is a research-grade statistical
> classification, only as reliable as the labeled data behind it, and must
> never drive clinical decisions. The UI states this on every result.

## What the web app does vs. the CLI

The CLI (`python run.py --video-dir data/videos --classify`) trains and
evaluates an XGBoost **classifier** across a labeled dataset (normal /
hypertonia / spasms) with grouped cross-validation. The web app handles the
**single-video** case: it extracts flow + the full feature battery + a plot for
one uploaded clip, and — once you've trained a model (see above) — a predicted
class with per-class probabilities.

## Files added

```
Dockerfile               # CPU image: python3.11 + ffmpeg + pipeline + web deps
docker-compose.yml       # one-command local run, with weight/run caches
.dockerignore
webapp/app.py            # FastAPI upload UI + background processing + prediction
webapp/requirements-web.txt
train.py                 # train + save the classifier (models/nmotion_model.joblib)
pipeline/predict.py      # train_and_save + per-video predict (added, non-destructive)
models/                  # drop a trained model here; mounted into the container
run.bat / stop.bat       # one-click start/stop on Windows
DEPLOY.md                # this file
```
