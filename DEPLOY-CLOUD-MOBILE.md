# Nmotion — Deploy to the cloud & install on any iPhone

This turns Nmotion into an **installable mobile app** without any Xcode, Swift,
or Apple Developer account. It is the same web app (your validated pipeline),
now made mobile-friendly and installable: on an iPhone you open the site in
Safari, tap Share → **Add to Home Screen**, and it launches full-screen with an
icon like a native app. From it you can **record a clip with the camera** or
**choose an existing video**, and it analyzes.

---

## ⚠️ READ FIRST — privacy & compliance
Cloud hosting means **the video leaves the phone and is uploaded to your server**.
For neonatal patient video that carries consent, privacy (PHI), and likely
medical-device/regulatory obligations. Only do this with your institution's
data-governance / REB approval, proper consent, ideally de-identified clips, and
a hosting region/agreement that permits health data. The app keeps a
"research — not a medical diagnosis" disclaimer visible; the compliance side is
yours to clear before real patient use.

---

## What you need
1. The `Nmotion` folder (with the latest changes — it now serves the mobile PWA).
2. A **GitHub account** (free) — the easiest way to hand your code to a host.
3. A **cloud host account**. Recommended for a non-developer: **Render**
   (render.com). Alternatives: Railway, Fly.io.
4. ~15–30 minutes, and a small monthly cost (see "Sizing & cost").

> HTTPS is required for the "Add to Home Screen" app behaviour. Render, Railway,
> and Fly all give you HTTPS automatically — do **not** self-host on plain HTTP.

---

## Sizing & cost (important — the app needs real memory)
This app loads PyTorch + the RAFT model, so a tiny free instance (512 MB) will
crash on startup. Pick an instance with **at least 2 GB RAM** (4 GB is safer).
- Render "Standard" (~2 GB) or larger.
- Processing runs on CPU and is **slow** (minutes per clip). To keep it usable,
  lower the frame cap (env var below). A GPU cloud instance is far faster but
  more expensive/complex — a later optimisation.

---

## Path A — Deploy on Render via GitHub (recommended, all clicks)

### 1. Put the code on GitHub (easiest with GitHub Desktop)
- Install **GitHub Desktop** (desktop.github.com), sign in.
- File → **New repository** → name it `nmotion` → set the local path, Create.
- Copy the **contents of your `Nmotion` folder** into that new repository folder.
- In GitHub Desktop: it shows the files → write a summary → **Commit to main** →
  **Publish repository** (you can keep it **Private**).

### 2. Create the web service on Render
- render.com → New → **Web Service** → connect your GitHub → pick the `nmotion` repo.
- Render detects the **Dockerfile** automatically. Settings:
  - **Instance type:** Standard (≥ 2 GB RAM).
  - **Environment variables** (Add):
    - `NMOTION_MAX_FRAMES` = `120`   (lower = faster; 240 is the default)
    - `NMOTION_MAX_WIDTH` = `512`    (lower = faster)
  - Leave the start command empty — the Dockerfile already handles it and reads
    Render's `$PORT`.
- Click **Create Web Service**. First build takes several minutes (downloads
  PyTorch). When it's live you get a URL like `https://nmotion-xxxx.onrender.com`.

### 3. Test it
- Open that HTTPS URL in any browser — you should see the Nmotion page.

---

## Path B — Deploy on Fly.io (no GitHub, uses the command line)
If you'd rather not use GitHub:
1. Install the Fly CLI (`flyctl`) and sign in (`fly auth signup`).
2. In the `Nmotion` folder, run `fly launch` — it detects the Dockerfile, asks a
   few questions, and deploys. Choose a machine with **≥ 2 GB RAM**
   (`fly scale memory 2048`). It gives you an `https://<app>.fly.dev` URL.
3. Set env vars: `fly secrets set NMOTION_MAX_FRAMES=120 NMOTION_MAX_WIDTH=512`.

---

## Install it on an iPhone
1. Open the **https://** URL in **Safari** (must be Safari for install, not Chrome).
2. Tap the **Share** button (square with an up arrow).
3. Tap **Add to Home Screen** → **Add**.
4. You now have an **Nmotion** icon on the home screen. Tapping it opens
   full-screen. Inside:
   - **Record video** → opens the camera to record a clip.
   - **Choose video** → pick an existing clip from the library/files.
   - Then **Process video** → the 4-step progress screen → results.

Any iPhone can do this from the same URL — just share the link.

---

## Predictions (class label)
Out of the box the cloud app runs in **features-only** mode (no diagnosis label),
because there is no trained model yet. When you have one:
1. Train + validate on the desktop app (`python train.py --video-dir data/videos`)
   → produces `models/nmotion_model.joblib`.
2. Put that file in the repo's `models/` folder, commit/push (Render redeploys),
   or `fly deploy` again. The Dockerfile now copies `models/`, so the model ships
   in the image and predictions turn on automatically. (Still labeled
   "research — not a diagnosis".)

---

## Everyday use & upkeep
- **Update the app:** change code → commit/push (Render) or `fly deploy` → it
  redeploys automatically.
- **Cost control:** some hosts can sleep the service when idle (slower first hit)
  or you can scale down between studies.
- **Big uploads:** long/high-res clips can be large; the frame cap keeps
  processing sane, but if uploads fail, trim clips shorter before uploading.

## Troubleshooting
- **App crashes on start / "out of memory":** instance too small — use ≥ 2 GB RAM.
- **"Add to Home Screen" missing camera/app feel:** make sure you opened the
  **https://** URL in **Safari** (not http, not Chrome).
- **Very slow analysis:** expected on CPU. Lower `NMOTION_MAX_FRAMES` /
  `NMOTION_MAX_WIDTH`, or move to a GPU instance later.
- **Camera didn't open on "Record":** some iOS versions open the file picker with
  a "Take Video" option instead — that also works.

---

## Files that make this work (already in your folder)
- `webapp/app.py` — mobile UI, camera/choose buttons, PWA manifest + service
  worker + app icons.
- `Dockerfile` — now reads `$PORT` and bundles `models/`.
- `docker-compose.yml` — still used for running locally.
