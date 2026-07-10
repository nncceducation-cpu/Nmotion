# Nmotion — Project Handoff & Continuation Guide

Read this first on a new computer (or paste the "Continuation brief" at the
bottom into a fresh Claude Cowork session). It captures the whole project so you
can pick up exactly where things were left.

Owner: Kiaksar Mohammad (neonatologist). Purpose: analyze neonatal videos for
movement features and (eventually) a movement-class prediction.

---

## 1. Where everything lives

- **GitHub repo (source of truth):** `https://github.com/nncceducation-cpu/Nmotion`
  (signed in as the `nncceducation-cpu` GitHub account). All code + docs are here.
- **Local run folder (this computer):** `C:\Users\khors\nmotion` (short path — git
  and Docker both work there; the deep Cowork "outputs" path breaks git).
- **Docker image/container:** built locally as `nmotion:latest`, container
  `nmotion-nmotion-1`, served on port 8000.

To use on another computer: install **Docker Desktop** + **Git**, then
`git clone https://github.com/nncceducation-cpu/Nmotion` and follow
`START-HERE.txt` / `DEPLOY.md`.

---

## 2. What Nmotion is (the pipeline)

Python pipeline (in `pipeline/`): dense **optical flow (RAFT, PyTorch)** →
**compact summaries** → **~90 movement features** (entropy, DFA, spectral,
symmetry, etc.) → optional **XGBoost classifier** (normal / hypertonia / spasms).
Originally CLI-only (`run.py`); we wrapped it in a web app.

---

## 3. What's built and its status

| Piece | Where | Status |
|-------|-------|--------|
| Desktop Docker web app (upload a video, get plot + features) | `webapp/app.py`, `Dockerfile`, `docker-compose.yml` | ✅ Works locally |
| Live **progress UI** (4-step tracker, %, frame counter, timer) | `webapp/app.py` | ✅ Works |
| Per-video **prediction** wiring (shows class + probabilities, with disclaimer) | `pipeline/predict.py`, `train.py` | ✅ Wired, but **no trained model yet** → runs features-only |
| **Mobile PWA** (installable on iPhone, Record/Choose buttons, manifest, icons) | `webapp/app.py` | ✅ Works locally |
| **Cloud deploy** guide (Render, 2 GB RAM, HTTPS, iPhone install) | `DEPLOY-CLOUD-MOBILE.md` | ⏸️ Not deployed (would cost ~$25/mo) |
| **iOS native app** (on-device Core ML) | `NMOTION-iOS-SPEC.md`, `NMOTION-iOS-APP-SCAFFOLD.md`, `ios_*.py` | 📄 Design + scaffold only — not built |

---

## 4. How to run it (recap)

Local, on this or any machine with Docker:
1. Open the project folder, in the File Explorer address bar type:
   `cmd /k docker compose up --build -d`  (first build downloads ~2–3 GB).
2. Open `http://localhost:8000`.
3. To stop: `cmd /k docker compose down`. To restart: `cmd /k docker compose up -d`.

`run.bat` exists but `.bat` opens in an editor on these machines — use the
address-bar command instead.

---

## 5. Key constraints & decisions (important context)

- **No trained classifier exists yet.** The app shows features only until a model
  is trained on labeled clips (`python train.py --video-dir data/videos`, needs
  videos in `data/videos/<class>/`). The classifier must be trained on the SAME
  optical-flow method used at inference.
- **CPU is slow.** RAFT on CPU is minutes per clip. Frame caps
  (`NMOTION_MAX_FRAMES`, `NMOTION_MAX_WIDTH`) keep it usable; a GPU makes it
  seconds (see GPU notes in `DEPLOY.md`).
- **Phone access needs the right network.** The hospital Wi-Fi (`healthy.bewell.ca`)
  isolates devices + firewalls the port, so phones can't reach the PC there. Works
  on a home network / phone hotspot (the PC's IP changes per network), or via the
  paid cloud (works anywhere).
- **Windows Firewall** may block inbound port 8000 (needs an admin rule to allow).
- **Privacy / regulatory (flagged, not resolved):** neonatal video is sensitive
  PHI. Cloud hosting sends it off-device → needs REB/data-governance sign-off,
  consent, ideally de-identified clips. A public "analyzes clinical video" app
  likely becomes a regulated medical device (SaMD). The UI always shows a
  "research — not a medical diagnosis" disclaimer. Confirm the pathway with your
  institution before real patient use.

---

## 6. Open next steps (pick up here)

1. **Train + validate the classifier** on labeled clips so predictions turn on.
2. **Get it onto a phone**: easiest = run locally on a **home network / hotspot**
   and open the PC's IP:8000; or deploy to the **cloud** (Render, ~$25/mo, 2 GB
   RAM) for access anywhere — repo is ready, just connect it on render.com.
3. **iOS native app** (optional, bigger effort): follow `NMOTION-iOS-SPEC.md`;
   the SwiftUI scaffold + Core ML conversion scripts are in the repo.
4. **Speed**: rebuild with GPU support if a machine with an NVIDIA card is used.

---

## 7. File map (in the repo)

```
pipeline/            the analysis pipeline (flow, features, predict, classify)
webapp/app.py        FastAPI web app: upload UI, progress, mobile PWA, prediction
Dockerfile           CPU image (reads $PORT for cloud); GPU notes in DEPLOY.md
docker-compose.yml   local run
train.py             train + save the classifier -> models/nmotion_model.joblib
DEPLOY.md            local + GPU + cloud deploy details
DEPLOY-CLOUD-MOBILE.md  deploy to Render + install on iPhone (PWA)
START-HERE.txt       quick run guide (travels with the folder)
NMOTION-iOS-SPEC.md  + NMOTION-iOS-APP-SCAFFOLD.md + ios_*.py  iOS design/scaffold
PROJECT-HANDOFF.md   this file
```

---

## 8. Continuation brief — paste this into a new Claude Cowork session

> I'm continuing a project called **Nmotion** — a neonatal video movement-analysis
> app. The code is on GitHub at `github.com/nncceducation-cpu/Nmotion` (I'll clone
> it locally). It's a Python pipeline (RAFT optical flow → ~90 features → optional
> XGBoost classifier) wrapped in a Dockerized FastAPI web app with a live progress
> UI and a mobile PWA (installable on iPhone, camera-record or upload). It runs
> locally with `docker compose up --build -d` on port 8000. Status: web app +
> progress + mobile PWA work; the classifier is wired but **not trained yet**
> (features-only mode); cloud deploy to Render is documented but not done; an
> on-device iOS app is designed/scaffolded but not built. Constraints: CPU is slow
> (GPU helps a lot); phone access needs a non-isolated network or cloud hosting;
> neonatal video is PHI so keep the "research, not a diagnosis" disclaimer and mind
> consent/REB/regulatory. Read `PROJECT-HANDOFF.md` in the repo for full details.
> Next I want to: [state your goal — e.g., train the classifier / deploy to cloud /
> build the iOS app / get it on a phone].
