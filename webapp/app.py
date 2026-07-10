"""
Nmotion web app — upload a neonatal video, run the optical-flow feature
pipeline, and view the extracted movement features + a plot in the browser.

Wraps the existing `pipeline` package without modifying it:
  1. The uploaded video is trimmed + downscaled with OpenCV so RAFT stays
     tractable on CPU.
  2. `pipeline.flow_extract.extract_flow` computes dense optical flow (RAFT).
  3. `pipeline.features.extract_features_single` computes the ~100 features.
  4. A magnitude-over-time plot + a feature table are rendered.

Processing runs in a background thread; the browser polls /status/{job_id}.

Env knobs (optional):
  NMOTION_DEVICE      "cpu" | "cuda"       (default: auto-detect)
  NMOTION_MAX_FRAMES  cap frames processed (default: 240; 0 = no cap)
  NMOTION_MAX_WIDTH   downscale width px   (default: 640; 0 = keep native)
  NMOTION_DATA_DIR    where jobs are stored (default: ./data_runtime)
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.compact import compute_magnitude_timeseries  # noqa: E402
from pipeline.features import extract_features_single  # noqa: E402
from pipeline.flow_extract import VIDEO_EXTENSIONS  # noqa: E402
from pipeline.predict import load_model, predict_video  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nmotion.web")

DEVICE = os.environ.get("NMOTION_DEVICE", "").strip() or None
MAX_FRAMES = int(os.environ.get("NMOTION_MAX_FRAMES", "240"))
MAX_WIDTH = int(os.environ.get("NMOTION_MAX_WIDTH", "640"))
DATA_DIR = Path(os.environ.get("NMOTION_DATA_DIR", ROOT / "webapp" / "data_runtime"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path(os.environ.get("NMOTION_MODEL_PATH", ROOT / "models" / "nmotion_model.joblib"))
MODEL_BUNDLE = load_model(MODEL_PATH)
if MODEL_BUNDLE:
    logger.info("Classifier loaded from %s (classes: %s)", MODEL_PATH, MODEL_BUNDLE["classes"])
else:
    logger.info("No classifier at %s — running in features-only mode.", MODEL_PATH)

app = FastAPI(title="Nmotion", version="1.0")


@dataclass
class Job:
    id: str
    status: str = "queued"
    message: str = ""
    stage: str = "queued"          # queued|prepare|flow|features|finalize|done|error
    stage_index: int = 0           # 0..4 for the step indicator
    percent: float = 0.0           # overall progress 0..100
    frame: int = 0                 # current flow frame
    total_frames: int = 0          # total flow frames
    started_at: float = field(default_factory=time.time)
    features: Dict[str, object] = field(default_factory=dict)
    prediction: Dict[str, object] = field(default_factory=dict)
    error: str = ""

    def set(self, *, stage=None, stage_index=None, percent=None,
            message=None, frame=None, total_frames=None):
        if stage is not None: self.stage = stage
        if stage_index is not None: self.stage_index = stage_index
        if percent is not None: self.percent = float(percent)
        if message is not None: self.message = message
        if frame is not None: self.frame = frame
        if total_frames is not None: self.total_frames = total_frames


JOBS: Dict[str, Job] = {}
_MODEL = None
_MODEL_LOCK = threading.Lock()


def _prep_video(src: Path, dst: Path) -> float:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f"Cannot open uploaded video: {src.name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 1.0
    if MAX_WIDTH and w > MAX_WIDTH:
        scale = MAX_WIDTH / float(w)
    out_w = int(round(w * scale)) if scale != 1.0 else w
    out_h = int(round(h * scale)) if scale != 1.0 else h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (out_w, out_h))
    written = 0
    cap_frames = MAX_FRAMES if MAX_FRAMES > 0 else 10**9
    while written < cap_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    if written < 2:
        raise ValueError("Video has fewer than 2 readable frames.")
    logger.info("Prepped %s: %d/%d frames, %dx%d @ %.1ffps",
                src.name, written, n_total, out_w, out_h, fps)
    return fps


def _get_model(device: str):
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            import torch
            from pipeline.flow_extract import _load_raft
            _MODEL = _load_raft(torch.device(device))
        return _MODEL


def _magnitude_plot(flow: np.ndarray, fps: float, out_png: Path) -> None:
    mag_ts = compute_magnitude_timeseries(flow)
    ts = mag_ts[:, 0]
    t = np.arange(len(ts)) / (fps or 30.0)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, ts, color="#2c7fb8", lw=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean flow magnitude")
    ax.set_title("Movement intensity over time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _process(job: Job, video_path: Path, job_dir: Path) -> None:
    try:
        import torch
        device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        job.status = "running"
        job.started_at = time.time()

        # Stage 1: prepare
        job.set(stage="prepare", stage_index=1, percent=2,
                message="Preparing video (trim + downscale)...")
        prepped = job_dir / "prepped.mp4"
        fps = _prep_video(video_path, prepped)
        job.set(percent=8)

        # Stage 2: optical flow (the slow part — report per-frame progress)
        job.set(stage="flow", stage_index=2, percent=10,
                message=f"Loading motion model on {device.upper()} "
                        f"(first run downloads weights)...")
        model = _get_model(device)
        job.set(message="Extracting dense optical flow (RAFT)...")
        flow_npy = job_dir / "flow.npy"
        from pipeline.flow_extract import extract_flow

        def _flow_progress(done: int, total: int) -> None:
            pct = 10 + 70 * (done / total if total else 0)
            job.set(percent=pct, frame=done, total_frames=total,
                    message=f"Analyzing motion — frame {done} of {total}")

        extract_flow(prepped, flow_npy, device=device, model=model,
                     progress_cb=_flow_progress)

        # Stage 3: features
        job.set(stage="features", stage_index=3, percent=83,
                message="Computing movement features...")
        flow = np.load(flow_npy, mmap_mode="r")
        feats = extract_features_single(
            np.asarray(flow), fps=fps, video_name=video_path.stem, group="uploaded")

        # Stage 4: finalize (plot + optional prediction)
        job.set(stage="finalize", stage_index=4, percent=90,
                message="Rendering movement plot...")
        _magnitude_plot(np.asarray(flow), fps, job_dir / "magnitude.png")

        if MODEL_BUNDLE is not None:
            job.set(percent=95, message="Running classifier...")
            try:
                job.prediction = predict_video(np.asarray(flow), fps, MODEL_BUNDLE)
            except Exception as pexc:
                logger.exception("Prediction failed")
                job.prediction = {"error": str(pexc)}
        clean = {}
        for k, v in feats.items():
            if isinstance(v, (int, float, str)):
                clean[k] = v
            else:
                try:
                    clean[k] = float(v)
                except Exception:
                    clean[k] = str(v)
        job.features = clean
        job.set(stage="done", stage_index=4, percent=100, message="Complete.")
        job.status = "done"
    except Exception as exc:
        logger.exception("Job %s failed", job.id)
        job.status = "error"
        job.stage = "error"
        job.error = f"{exc}\n\n{traceback.format_exc()}"
        job.message = "Failed."


app.mount("/runs", StaticFiles(directory=str(DATA_DIR)), name="runs")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400,
            detail=f"Unsupported type '{ext}'. Allowed: {sorted(VIDEO_EXTENSIONS)}")
    job = Job(id=uuid.uuid4().hex[:12])
    JOBS[job.id] = job
    job_dir = DATA_DIR / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    src = job_dir / f"input{ext}"
    with src.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    threading.Thread(target=_process, args=(job, src, job_dir), daemon=True).start()
    return JSONResponse({"job_id": job.id})


@app.get("/status/{job_id}")
def status(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    payload = {
        "status": job.status,
        "message": job.message,
        "error": job.error,
        "stage": job.stage,
        "stage_index": job.stage_index,
        "percent": round(job.percent, 1),
        "frame": job.frame,
        "total_frames": job.total_frames,
        "elapsed": round(time.time() - job.started_at, 1),
    }
    if job.status == "done":
        payload["features"] = job.features
        payload["prediction"] = job.prediction
        payload["plot_url"] = f"/runs/{job_id}/magnitude.png"
    return JSONResponse(payload)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "device_env": DEVICE, "max_frames": MAX_FRAMES,
            "model_loaded": MODEL_BUNDLE is not None,
            "classes": MODEL_BUNDLE["classes"] if MODEL_BUNDLE else []}


# ---------------------------------------------------------------------------
# PWA: make the site installable on an iPhone home screen ("Add to Home Screen")
# ---------------------------------------------------------------------------

_MANIFEST = json.dumps({
    "name": "Nmotion",
    "short_name": "Nmotion",
    "description": "Neonatal movement analysis",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#0f172a",
    "theme_color": "#0f172a",
    "icons": [
        {"src": "/icon-180.png", "sizes": "180x180", "type": "image/png"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png",
         "purpose": "any maskable"},
    ],
})

# Minimal service worker — required for install; network-first, no offline cache
# of uploads (video processing needs the server anyway).
_SW = """
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => self.clients.claim());
self.addEventListener('fetch', e => {});
"""

_ICON_CACHE: Dict[int, bytes] = {}


def _make_icon(size: int) -> bytes:
    if size in _ICON_CACHE:
        return _ICON_CACHE[size]
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (size, size), (15, 23, 42, 255))  # slate bg
    d = ImageDraw.Draw(img)
    # simple stylised motion glyph: three rising bars
    pad = size // 5
    bw = (size - 2 * pad) // 5
    heights = [0.35, 0.6, 0.9]
    base = size - pad
    for i, h in enumerate(heights):
        x0 = pad + i * bw * 2
        y0 = int(base - (size - 2 * pad) * h)
        d.rounded_rectangle([x0, y0, x0 + bw, base], radius=bw // 3,
                            fill=(56, 189, 248, 255))
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    _ICON_CACHE[size] = data
    return data


@app.get("/manifest.webmanifest")
def manifest() -> Response:
    return Response(content=_MANIFEST, media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker() -> Response:
    return Response(content=_SW, media_type="application/javascript")


@app.get("/icon-180.png")
def icon_180() -> Response:
    return Response(content=_make_icon(180), media_type="image/png")


@app.get("/icon-512.png")
def icon_512() -> Response:
    return Response(content=_make_icon(512), media_type="image/png")


INDEX_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Nmotion - neonatal movement analysis</title>
<meta name="theme-color" content="#0f172a">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Nmotion">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="apple-touch-icon" href="/icon-180.png">
<link rel="icon" href="/icon-180.png">
<style>
  :root{--bg:#0f172a;--card:#1e293b;--fg:#e2e8f0;--mut:#94a3b8;--acc:#38bdf8;}
  *{box-sizing:border-box} body{margin:0;font-family:system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg);}
  .wrap{max-width:920px;margin:0 auto;padding:32px 20px 80px;}
  h1{font-size:26px;margin:0 0 4px} .sub{color:var(--mut);margin:0 0 28px}
  .card{background:var(--card);border:1px solid #334155;border-radius:14px;padding:24px;margin-bottom:22px;}
  .drop{border:2px dashed #475569;border-radius:12px;padding:44px;text-align:center;cursor:pointer;transition:.15s;}
  .drop:hover,.drop.hot{border-color:var(--acc);background:#0b1220;}
  .btn{background:var(--acc);color:#04283a;border:0;border-radius:9px;padding:11px 20px;font-weight:600;cursor:pointer;font-size:15px;}
  .btn:disabled{opacity:.5;cursor:not-allowed}
  .btn2{background:#334155;color:var(--fg)}
  .pickrow{display:flex;gap:12px;margin-top:16px}
  .pickrow .btn{flex:1;padding:16px;font-size:16px}
  @media(max-width:640px){
    .wrap{padding:20px 14px 60px}
    h1{font-size:23px}
    .card{padding:18px}
    .drop{padding:30px 16px}
    .pickrow{flex-direction:column}
    .pickrow .btn{width:100%}
    #go{width:100%;padding:16px;font-size:16px}
  }
  .mut{color:var(--mut);font-size:13px}
  .status{font-size:14px;margin-top:14px;min-height:22px}
  @keyframes load{0%{margin-left:-35%}100%{margin-left:100%}}
  .prog{display:none;margin-top:20px}
  .wave{display:flex;gap:4px;justify-content:center;align-items:flex-end;height:28px;margin-bottom:16px}
  .wave span{width:5px;height:6px;background:var(--acc);border-radius:3px;animation:wv 1s ease-in-out infinite}
  @keyframes wv{0%,100%{height:6px;opacity:.45}50%{height:26px;opacity:1}}
  .steps{display:flex;margin:4px 0 20px}
  .step{flex:1;text-align:center;position:relative;font-size:11.5px}
  .step:before{content:"";position:absolute;top:13px;left:-50%;width:100%;height:2px;background:#334155;z-index:0}
  .step:first-child:before{display:none}
  .step .dot{width:28px;height:28px;border-radius:50%;background:#0b1220;border:2px solid #334155;display:flex;align-items:center;justify-content:center;margin:0 auto 7px;position:relative;z-index:1;font-weight:700;color:var(--mut);transition:.25s;font-size:13px}
  .step .lbl{color:var(--mut)}
  .step.active .dot{border-color:var(--acc);color:var(--acc);box-shadow:0 0 0 4px rgba(56,189,248,.16)}
  .step.done .dot{background:var(--acc);border-color:var(--acc);color:#04283a}
  .step.active:before,.step.done:before{background:var(--acc)}
  .step.active .lbl,.step.done .lbl{color:var(--fg)}
  .pct{font-size:36px;font-weight:800;letter-spacing:-1.5px;line-height:1}
  .pmsg{color:var(--mut);font-size:13px;margin:4px 0 14px;min-height:18px}
  .pbar{height:12px;background:#0b1220;border-radius:8px;overflow:hidden;border:1px solid #334155}
  .pbar>i{display:block;height:100%;width:0;background:linear-gradient(90deg,#38bdf8,#818cf8);border-radius:8px;transition:width .45s ease}
  .pbar.indet>i{width:35%!important;background:var(--acc);animation:load 1.1s infinite;transition:none}
  .meta{display:flex;justify-content:space-between;color:var(--mut);font-size:12px;margin-top:9px}
  img{max-width:100%;border-radius:10px;margin-top:10px;background:#fff}
  code{background:#0b1220;padding:2px 6px;border-radius:5px}
  .err{color:#fda4af;white-space:pre-wrap;font-size:12px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px 26px}
  @media(max-width:640px){.grid{grid-template-columns:1fr}}
  .pred-label{font-size:22px;font-weight:700;margin:2px 0 14px}
  .prow{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:13px}
  .prow .nm{width:120px;color:var(--mut)}
  .ptrack{flex:1;height:14px;background:#0b1220;border-radius:7px;overflow:hidden}
  .ptrack>i{display:block;height:100%;background:var(--acc)}
  .prow .pct{width:52px;text-align:right}
  .disc{margin-top:16px;padding:11px 13px;border-radius:9px;background:#3b1d1d;border:1px solid #7f1d1d;color:#fecaca;font-size:12px;line-height:1.5}
</style></head><body><div class="wrap">
  <h1>Nmotion</h1>
  <p class="sub">Upload a neonatal video to extract optical-flow movement features.</p>
  <div class="card">
    <div id="drop" class="drop">
      <div style="font-size:16px">Record a clip or choose a video</div>
      <div class="mut" style="margin-top:8px">.mp4 .avi .mov .mkv .wmv</div>
      <input id="file" type="file" accept="video/*" style="display:none">
      <input id="cam" type="file" accept="video/*" capture="environment" style="display:none">
    </div>
    <div class="pickrow">
      <button id="record" class="btn">Record video</button>
      <button id="choose" class="btn btn2">Choose video</button>
    </div>
    <div style="margin-top:14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap">
      <button id="go" class="btn" disabled>Process video</button>
      <span id="fname" class="mut"></span>
    </div>
    <div id="prog" class="prog">
      <div class="wave">
        <span style="animation-delay:0s"></span><span style="animation-delay:.09s"></span>
        <span style="animation-delay:.18s"></span><span style="animation-delay:.27s"></span>
        <span style="animation-delay:.36s"></span><span style="animation-delay:.27s"></span>
        <span style="animation-delay:.18s"></span><span style="animation-delay:.09s"></span>
      </div>
      <div class="steps">
        <div class="step" data-s="1"><div class="dot">1</div><div class="lbl">Prepare</div></div>
        <div class="step" data-s="2"><div class="dot">2</div><div class="lbl">Optical flow</div></div>
        <div class="step" data-s="3"><div class="dot">3</div><div class="lbl">Features</div></div>
        <div class="step" data-s="4"><div class="dot">4</div><div class="lbl">Finalize</div></div>
      </div>
      <div class="pct" id="pct">0%</div>
      <div class="pmsg" id="pmsg">Starting…</div>
      <div class="pbar" id="pbar"><i></i></div>
      <div class="meta"><span id="metaFrame"></span><span id="metaTime"></span></div>
    </div>
    <div id="status" class="status"></div>
  </div>
  <div id="predcard" class="card" style="display:none">
    <h2 style="margin-top:0;font-size:19px">Predicted class</h2>
    <div id="predlabel" class="pred-label"></div>
    <div id="predbars"></div>
    <div class="disc"><b>Research tool — not a medical diagnosis.</b> This is a
      statistical classification from a model trained on a limited dataset. It must
      not be used for clinical decisions. Always consult a qualified clinician.</div>
  </div>
  <div id="results" class="card" style="display:none">
    <h2 style="margin-top:0;font-size:19px">Movement analysis</h2>
    <img id="plot" alt="movement plot">
    <h3 style="font-size:15px;margin:22px 0 8px">Features</h3>
    <div id="feats" class="grid"></div>
  </div>
<script>
const drop=document.getElementById('drop'),fileEl=document.getElementById('file'),
  go=document.getElementById('go'),fname=document.getElementById('fname'),
  statusEl=document.getElementById('status'),
  prog=document.getElementById('prog'),pct=document.getElementById('pct'),
  pmsg=document.getElementById('pmsg'),pbar=document.getElementById('pbar'),
  metaFrame=document.getElementById('metaFrame'),metaTime=document.getElementById('metaTime'),
  steps=[...document.querySelectorAll('.step')],
  results=document.getElementById('results'),plot=document.getElementById('plot'),
  feats=document.getElementById('feats'),
  predcard=document.getElementById('predcard'),predlabel=document.getElementById('predlabel'),
  predbars=document.getElementById('predbars');
let chosen=null;
const cam=document.getElementById('cam'),recordBtn=document.getElementById('record'),
  chooseBtn=document.getElementById('choose');
drop.onclick=()=>fileEl.click();
recordBtn.onclick=()=>cam.click();
chooseBtn.onclick=()=>fileEl.click();
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hot')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('hot')}));
drop.addEventListener('drop',ev=>{if(ev.dataTransfer.files.length){pick(ev.dataTransfer.files[0])}});
fileEl.onchange=()=>{if(fileEl.files.length)pick(fileEl.files[0])};
cam.onchange=()=>{if(cam.files.length)pick(cam.files[0])};
function pick(f){chosen=f;fname.textContent=f.name;go.disabled=false;}
if('serviceWorker' in navigator){navigator.serviceWorker.register('/sw.js').catch(()=>{});}
go.onclick=async()=>{
  if(!chosen)return;
  go.disabled=true;results.style.display='none';predcard.style.display='none';
  prog.style.display='block';statusEl.textContent='';
  pbar.classList.add('indet');
  setProgress({stage:'prepare',stage_index:0,percent:0,message:'Uploading video…',total_frames:0,elapsed:0});
  const fd=new FormData();fd.append('file',chosen);
  let r=await fetch('/upload',{method:'POST',body:fd});
  if(!r.ok){const e=await r.json();statusEl.innerHTML='<span class="err">'+(e.detail||'Upload failed')+'</span>';prog.style.display='none';go.disabled=false;return;}
  const j=await r.json();poll(j.job_id);
};
function fmtTime(s){s=Math.round(s||0);const m=Math.floor(s/60),ss=s%60;return m+':'+String(ss).padStart(2,'0');}
function setProgress(j){
  const si=j.stage_index||0, pc=Math.round(j.percent||0), done=(j.status==='done');
  steps.forEach((el,i)=>{const n=i+1;el.classList.toggle('done',n<si||done);el.classList.toggle('active',n===si&&!done);});
  pct.textContent=pc+'%';
  pmsg.textContent=j.message||'';
  const indet=(j.stage==='flow'&&!j.total_frames)&&!done;
  pbar.classList.toggle('indet',!!indet);
  if(!indet)pbar.querySelector('i').style.width=pc+'%';
  metaFrame.textContent=j.total_frames?('Frame '+j.frame+' / '+j.total_frames):'';
  metaTime.textContent='Elapsed '+fmtTime(j.elapsed);
}
function renderPrediction(pred){
  predcard.style.display='none';
  if(!pred||!pred.label){return;}
  predlabel.textContent=pred.label.toUpperCase();
  const probs=pred.probabilities||{};
  const entries=Object.entries(probs).sort((a,b)=>b[1]-a[1]);
  predbars.innerHTML='';
  for(const [name,p] of entries){
    const pc=(p*100).toFixed(1)+'%';
    const row=document.createElement('div');row.className='prow';
    row.innerHTML='<span class="nm">'+name+'</span><span class="ptrack"><i style="width:'+(p*100)+'%"></i></span><span class="pct">'+pc+'</span>';
    predbars.appendChild(row);
  }
  const note=document.createElement('div');note.className='mut';note.style.marginTop='10px';
  note.textContent='Averaged over '+(pred.n_clips||0)+' clip(s).';
  predbars.appendChild(note);
  predcard.style.display='block';
}
async function poll(id){
  let j;
  try{const r=await fetch('/status/'+id);j=await r.json();}
  catch(e){setTimeout(()=>poll(id),1200);return;}
  if(j.status==='error'){
    prog.style.display='none';go.disabled=false;
    statusEl.innerHTML='<span class="err">'+j.error+'</span>';return;
  }
  setProgress(j);
  if(j.status==='done'){
    go.disabled=false;
    setTimeout(()=>{prog.style.display='none';},700);
    plot.src=j.plot_url+'?t='+Date.now();
    renderPrediction(j.prediction);
    feats.innerHTML='';
    const skip=new Set(['video','group']);
    for(const [k,v] of Object.entries(j.features)){
      if(skip.has(k))continue;
      let val=v;
      if(typeof v==='number'){val=(Math.abs(v)>=1000||(Math.abs(v)<0.001&&v!==0))?v.toExponential(3):(+v).toFixed(4);}
      const row=document.createElement('div');
      row.innerHTML='<code>'+k+'</code> <span class="mut" style="float:right">'+val+'</span>';
      feats.appendChild(row);
    }
    results.style.display='block';return;
  }
  setTimeout(()=>poll(id),1000);
}
</script>
</div></body></html>
"""
