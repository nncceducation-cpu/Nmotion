# Nmotion iOS — Architecture & Technical Spec

On-device neonatal movement analysis for iPhone. Record or import a clip, and
the phone extracts optical-flow movement features and (optionally) predicts a
movement class — **without the video ever leaving the device**.

Decided scope: **on-device processing**, **private clinical/research
distribution** (TestFlight / Apple Business Manager), output includes a **class
prediction**.

Companion files in this folder (all prefixed `ios_`):
- `ios_convert_classifier_to_coreml.py` — XGBoost model → Core ML
- `ios_convert_flow_to_coreml.py` — RAFT flow → Core ML (Option A)
- `ios_export_feature_reference.py` — dump desktop features for parity tests
- `NMOTION-iOS-APP-SCAFFOLD.md` — the SwiftUI app source, ready to paste into Xcode

---

## 1. Why on-device (and what it costs)

Neonatal video is sensitive patient data. On-device processing means the clip is
never uploaded, which removes the biggest privacy/consent problem and is the
right default here. The cost is engineering: the desktop app's Python/PyTorch
stack (RAFT, XGBoost, OpenCV, SciPy) cannot run on iOS, so each stage is
reimplemented with Apple-native frameworks or converted to Core ML.

| Desktop stage (Python)                          | iOS equivalent                                     |
|-------------------------------------------------|----------------------------------------------------|
| Read + downscale frames (OpenCV)                | `AVAssetReader` + `vImage`/Core Image              |
| Dense optical flow (RAFT, PyTorch)              | **RAFT → Core ML**, or Apple `Vision` optical flow |
| Compact + feature battery (NumPy/SciPy/antropy) | Swift port using `Accelerate`/`vDSP`               |
| Classifier (XGBoost)                            | **XGBoost → Core ML** tree ensemble                |
| Plot + table (matplotlib)                       | Swift Charts + SwiftUI                             |

---

## 2. The critical decision: which optical flow?

The classifier's features come from optical flow, so **training and inference
must use the same flow method** or the numbers won't line up.

### Option A — RAFT → Core ML (fidelity)
Convert the exact RAFT-Large model so on-device features match the desktop
features the model was trained on.
- Pros: a desktop-trained model transfers directly; one source of truth.
- Cons: RAFT's iterative refinement + correlation-volume ops make conversion via
  `coremltools` fiddly and heavier at runtime.
- Mitigation: fewer refinement iterations (8→4), run at 384–512 px, and verify
  feature parity vs desktop.

### Option B — Apple Vision optical flow (simplicity)
`VNGenerateOpticalFlowRequest` computes dense flow on-device, natively, no model
to convert.
- Pros: far simpler, fast, Metal-accelerated, always available.
- Cons: it is **not** RAFT — scale differs — so a model trained on RAFT features
  will misbehave. You must **train the classifier on Vision-flow features**.

### Recommendation
There is **no validated classifier yet**, so do the science once, on the method
you will actually ship. For a clean iOS-native app, **Option B (Vision)** is the
pragmatic choice: build a small macOS command-line tool that reuses the Swift
`FeatureExtractor` to compute training features from your labeled clips, train
the classifier on those, then ship the same code on the phone. Use **Option A**
only if you must reuse an existing RAFT-trained model.

The scaffold ships with **Option B (Vision)** wired up because it runs today;
`ios_convert_flow_to_coreml.py` covers Option A when needed.

---

## 3. On-device pipeline (per clip)

```
AVURLAsset (recorded or imported)
   │  AVAssetReader → CVPixelBuffer frames
   ▼
Preprocess:  trim to N frames (default 240), downscale to width 640, grayscale
   ▼
Optical flow per adjacent frame pair (Vision or Core ML RAFT) → [N-1] flow fields
   ▼
Compact:  per-frame magnitude time series [N-1, 6] + spatial summary [N-1, 12]
   ▼
Feature battery:  ~90 scalars (entropy, spectral, DFA, symmetry, KE, MSE, …)
   ├── Results: movement-intensity plot + feature table
   └── Classifier (Core ML) → probabilities → predicted label + disclaimer
```

Runs off the main thread (`Task`), publishing progress (stage + percent + frame
counter) to the UI — mirroring the web app's progress screen.

### Feature-battery port (the real work)
Reimplement `pipeline/feature_battery.py` + `pipeline/compact.py` in Swift, mostly
`vDSP`/`Accelerate`. The tricky ones: sample entropy / multiscale entropy (MSE),
DFA (detrended fluctuation analysis), spectral entropy / peak frequency (Welch
PSD via FFT), and left/right symmetry.

**Validate every feature numerically** against the Python output on the same clip
(tolerance ~1e-3). `ios_export_feature_reference.py` dumps the desktop feature
vector for a set of clips so the Swift port can be unit-tested. **Do not ship the
classifier until parity passes** — a mismatch silently corrupts predictions.

### Classifier
`ios_convert_classifier_to_coreml.py` converts the trained
`models/nmotion_model.joblib` (XGBoost + label list + feature-column order) into
`NmotionClassifier.mlmodel`. The Swift classifier assembles the feature vector in
the **exact stored column order**, runs the Core ML model, and returns per-class
probabilities. If no model is bundled, the app runs in features-only mode — same
as the web app.

---

## 4. App structure (SwiftUI)

Screens:
1. **Home** — "Record" and "Import" buttons; list of past on-device results.
2. **Record** — camera preview, tap to record, stop → hand the URL to the pipeline.
3. **Analyzing** — 4-step tracker (Prepare → Optical flow → Features → Finalize)
   with live percentage + frame counter.
4. **Results** — movement-intensity chart (Swift Charts), feature table, and a
   prediction card with per-class probabilities + the mandatory disclaimer.

Info.plist permissions: `NSCameraUsageDescription`,
`NSPhotoLibraryUsageDescription` (and `NSMicrophoneUsageDescription` only if you
record audio). Minimum iOS 16 (Swift Charts, Vision flow, PhotosUI).

Full source is in `NMOTION-iOS-APP-SCAFFOLD.md`.

---

## 5. Distribution (private clinical/research)

- **TestFlight** for a known tester group, or **Apple Business/School Manager**
  custom-app distribution to managed devices. Both avoid public App-Store review
  friction while staying inside Apple's rules.
- Keep it **research-labeled**: the prediction card must always show the
  "research tool — not a medical diagnosis" disclaimer, and onboarding + the app
  description must say the same. Ethical requirement, and it reduces review risk.
- If you later go public or it is used to inform clinical decisions, it likely
  becomes a regulated **medical device (SaMD)** — plan that pathway (FDA / Health
  Canada) before wide release. This is a flag, not legal/regulatory advice;
  confirm with your institution's regulatory/QI office.

---

## 6. What must be true before it "analyzes" for real

1. A **trained, validated classifier** exists (built + evaluated on the desktop
   pipeline with grouped cross-validation on labeled clips). Until then: features
   only.
2. It was trained on features from the **same optical-flow method** the app uses.
3. **Feature parity** between the Swift port and the Python battery is verified.
4. Clinical validation appropriate to the intended use, disclaimer always shown.

---

## 7. Rough effort estimate (developer familiar with iOS/Core ML)

| Piece                                       | Estimate |
|---------------------------------------------|----------|
| App shell, capture, import, results UI      | 3–5 days |
| Vision optical-flow + frame pipeline        | 2–3 days |
| Feature-battery Swift port + parity tests   | 5–10 days (the bulk) |
| Classifier conversion + wiring              | 1–2 days |
| RAFT→Core ML (only if Option A)             | +3–7 days, risk-dependent |
| TestFlight setup, polish, disclaimer        | 2–3 days |

The feature-battery port is the long pole and the part that most needs the parity
harness. Everything else is standard iOS work.
