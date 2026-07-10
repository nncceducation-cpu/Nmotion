# Nmotion iOS — SwiftUI app scaffold

A starting scaffold for the on-device app (Option B: Apple Vision optical flow).
It compiles conceptually and gives you the full skeleton — Home → Record/Import →
Analyzing (progress) → Results — with the on-device pipeline wired to Vision
optical flow. **The one big TODO is porting the feature battery** (marked in
`FeatureExtractor.swift`); until that matches the Python output (verify with
`ios_export_feature_reference.py`), keep the classifier off (features-only).

## How to use
1. Xcode → New Project → iOS App → SwiftUI → name it `Nmotion`. Min iOS 16.
2. Create files below with these exact names; paste each block in.
3. Info.plist: add `NSCameraUsageDescription` and `NSPhotoLibraryUsageDescription`.
4. (Later) run `ios_convert_classifier_to_coreml.py`, drag `NmotionClassifier.mlmodel`
   into the project, and paste the feature-column order into `NmotionClassifier.swift`.
5. Build to a real device (camera + Vision flow don't work in the Simulator).

---

### NmotionApp.swift
```swift
import SwiftUI

@main
struct NmotionApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}
```

### Models.swift
```swift
import Foundation

struct Prediction: Identifiable {
    let id = UUID()
    let label: String
    let probabilities: [(String, Double)]   // sorted desc
    let nClips: Int
}

struct AnalysisResult: Identifiable {
    let id = UUID()
    let clipName: String
    let magnitudeSeries: [Double]           // for the plot
    let fps: Double
    let features: [(String, Double)]        // ~90 name/value pairs
    let prediction: Prediction?             // nil in features-only mode
    let date = Date()
}

enum Stage: Int, CaseIterable {
    case prepare = 1, flow = 2, features = 3, finalize = 4
    var title: String {
        switch self {
        case .prepare: return "Prepare"
        case .flow: return "Optical flow"
        case .features: return "Features"
        case .finalize: return "Finalize"
        }
    }
}

@MainActor
final class ProgressModel: ObservableObject {
    @Published var running = false
    @Published var stage: Stage = .prepare
    @Published var percent: Double = 0
    @Published var frame = 0
    @Published var totalFrames = 0
    @Published var message = ""
    @Published var startedAt = Date()

    func reset() {
        running = true; stage = .prepare; percent = 0
        frame = 0; totalFrames = 0; message = "Preparing…"; startedAt = Date()
    }
    var elapsed: String {
        let s = Int(Date().timeIntervalSince(startedAt))
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}
```

### ContentView.swift
```swift
import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var progress = ProgressModel()
    @State private var result: AnalysisResult?
    @State private var showRecorder = false
    @State private var pickerItem: PhotosPickerItem?
    private let analyzer = Analyzer()

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                Text("Nmotion").font(.largeTitle).bold()
                Text("Record or import a neonatal clip to analyze movement on-device.")
                    .foregroundStyle(.secondary).multilineTextAlignment(.center)

                if progress.running {
                    AnalyzingView(progress: progress)
                } else {
                    HStack(spacing: 16) {
                        Button { showRecorder = true } label: {
                            Label("Record", systemImage: "camera.fill")
                                .frame(maxWidth: .infinity)
                        }.buttonStyle(.borderedProminent)

                        PhotosPicker(selection: $pickerItem, matching: .videos) {
                            Label("Import", systemImage: "square.and.arrow.up")
                                .frame(maxWidth: .infinity)
                        }.buttonStyle(.bordered)
                    }.padding(.horizontal)
                }
                Spacer()
            }
            .padding(.top, 40)
            .navigationDestination(item: $result) { ResultsView(result: $0) }
            .sheet(isPresented: $showRecorder) {
                CameraRecorderView { url in showRecorder = false; run(url) }
            }
            .onChange(of: pickerItem) { _, item in
                guard let item else { return }
                Task {
                    if let url = try? await item.loadTransferable(type: VideoFile.self)?.url {
                        run(url)
                    }
                }
            }
        }
    }

    private func run(_ url: URL) {
        progress.reset()
        Task {
            do {
                let r = try await analyzer.analyze(url: url, progress: progress)
                progress.running = false
                result = r
            } catch {
                progress.running = false
                progress.message = "Failed: \(error.localizedDescription)"
            }
        }
    }
}

// Lets PhotosPicker hand us a file URL.
struct VideoFile: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { file in
            SentTransferredFile(file.url)
        } importing: { received in
            let copy = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + ".mov")
            try? FileManager.default.copyItem(at: received.file, to: copy)
            return Self(url: copy)
        }
    }
}
```

### AnalyzingView.swift
```swift
import SwiftUI

struct AnalyzingView: View {
    @ObservedObject var progress: ProgressModel

    var body: some View {
        VStack(spacing: 16) {
            HStack(spacing: 24) {
                ForEach(Stage.allCases, id: \.rawValue) { s in
                    VStack {
                        ZStack {
                            Circle().fill(color(for: s)).frame(width: 30, height: 30)
                            Text("\(s.rawValue)").foregroundStyle(.white).bold()
                        }
                        Text(s.title).font(.caption2)
                            .foregroundStyle(s.rawValue <= progress.stage.rawValue ? .primary : .secondary)
                    }
                }
            }
            Text("\(Int(progress.percent))%").font(.system(size: 34, weight: .heavy))
            Text(progress.message).font(.caption).foregroundStyle(.secondary)
            ProgressView(value: progress.percent, total: 100).tint(.blue)
            HStack {
                Text(progress.totalFrames > 0 ? "Frame \(progress.frame)/\(progress.totalFrames)" : "")
                Spacer()
                Text("Elapsed \(progress.elapsed)")
            }.font(.caption2).foregroundStyle(.secondary)
        }.padding()
    }

    private func color(for s: Stage) -> Color {
        s.rawValue < progress.stage.rawValue ? .blue
            : (s.rawValue == progress.stage.rawValue ? .blue : .gray.opacity(0.4))
    }
}
```

### Analyzer.swift  (pipeline orchestration)
```swift
import Foundation
import AVFoundation

struct Analyzer {
    let maxFrames = 240
    let maxWidth = 640

    func analyze(url: URL, progress: ProgressModel) async throws -> AnalysisResult {
        // Stage 1: prepare (read + downscale + grayscale frames)
        await update(progress, .prepare, 2, "Preparing video…")
        let frames = try await VideoFrames.load(url: url, maxFrames: maxFrames, maxWidth: maxWidth)
        let fps = frames.fps
        await update(progress, .prepare, 8, "Prepared \(frames.images.count) frames")

        // Stage 2: optical flow (Vision), per adjacent pair, with progress
        await update(progress, .flow, 10, "Analyzing motion…")
        let flow = OpticalFlow()
        var magTS: [[Double]] = []      // per-frame magnitude summary [6]
        var spatial: [[Double]] = []    // per-frame spatial summary [12]
        let n = frames.images.count - 1
        await MainActor.run { progress.totalFrames = n }
        for i in 0..<n {
            let f = try flow.compute(frames.images[i], frames.images[i + 1])
            magTS.append(Compact.magnitudeSummary(f))
            spatial.append(Compact.spatialSummary(f))
            let pct = 10 + 70 * Double(i + 1) / Double(n)
            await update(progress, .flow, pct, "Analyzing motion — frame \(i + 1) of \(n)", frame: i + 1)
        }

        // Stage 3: features
        await update(progress, .features, 83, "Computing movement features…")
        let features = FeatureExtractor.compute(magTS: magTS, spatial: spatial, fps: fps)

        // Stage 4: finalize (+ optional classify)
        await update(progress, .finalize, 92, "Finalizing…")
        let series = magTS.map { $0.first ?? 0 }      // mean-magnitude time series
        var prediction: Prediction? = nil
        if let clf = NmotionClassifier.shared {
            await update(progress, .finalize, 96, "Classifying…")
            prediction = clf.predict(features: features)
        }
        await update(progress, .finalize, 100, "Complete")

        return AnalysisResult(clipName: url.lastPathComponent,
                              magnitudeSeries: series, fps: fps,
                              features: features, prediction: prediction)
    }

    private func update(_ p: ProgressModel, _ stage: Stage, _ pct: Double,
                        _ msg: String, frame: Int? = nil) async {
        await MainActor.run {
            p.stage = stage; p.percent = pct; p.message = msg
            if let frame { p.frame = frame }
        }
    }
}
```

### VideoFrames.swift
```swift
import AVFoundation
import CoreImage
import UIKit

struct VideoFrames {
    let images: [CIImage]   // grayscale, downscaled
    let fps: Double

    static func load(url: URL, maxFrames: Int, maxWidth: Int) async throws -> VideoFrames {
        let asset = AVURLAsset(url: url)
        let track = try await asset.loadTracks(withMediaType: .video).first
        guard let track else { throw NSError(domain: "Nmotion", code: 1) }
        let nominalFPS = try await track.load(.nominalFrameRate)
        let fps = nominalFPS > 0 ? Double(nominalFPS) : 30.0

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        reader.add(output); reader.startReading()

        var out: [CIImage] = []
        let ctx = CIContext()
        while out.count < maxFrames, let sb = output.copyNextSampleBuffer(),
              let px = CMSampleBufferGetImageBuffer(sb) {
            var img = CIImage(cvPixelBuffer: px)
            let w = img.extent.width
            if w > CGFloat(maxWidth) {
                let s = CGFloat(maxWidth) / w
                img = img.transformed(by: CGAffineTransform(scaleX: s, y: s))
            }
            // grayscale
            img = img.applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0])
            out.append(img.clampedToExtent())
        }
        reader.cancelReading()
        _ = ctx
        guard out.count >= 2 else { throw NSError(domain: "Nmotion", code: 2,
            userInfo: [NSLocalizedDescriptionKey: "Video has fewer than 2 frames."]) }
        return VideoFrames(images: out, fps: fps)
    }
}
```

### OpticalFlow.swift  (Apple Vision — Option B)
```swift
import Vision
import CoreImage

struct OpticalFlow {
    // Returns dense flow as a CVPixelBuffer of 2-channel float (dx, dy).
    func compute(_ a: CIImage, _ b: CIImage) throws -> CVPixelBuffer {
        let request = VNGenerateOpticalFlowRequest(targetedCIImage: b, options: [:])
        // Lower computationAccuracy = faster; .high matches research fidelity best.
        request.computationAccuracy = .high
        request.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float
        let handler = VNImageRequestHandler(ciImage: a, options: [:])
        try handler.perform([request])
        guard let obs = request.results?.first as? VNPixelBufferObservation else {
            throw NSError(domain: "Nmotion", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Optical flow failed."])
        }
        return obs.pixelBuffer   // [H,W,2] float32 dx,dy
    }
}
```

### Compact.swift  (per-frame summaries from one flow field)
```swift
import CoreVideo
import Accelerate

enum Compact {
    /// [mean_mag, std_mag, max_mag, p90_mag, active_fraction, mean_dir_coherence]
    static func magnitudeSummary(_ flow: CVPixelBuffer) -> [Double] {
        let (dx, dy, count) = channels(flow)
        var mags = [Float](repeating: 0, count: count)
        for i in 0..<count { mags[i] = (dx[i]*dx[i] + dy[i]*dy[i]).squareRoot() }
        let mean = vDSP.mean(mags)
        let std  = stddev(mags, mean: mean)
        let mx   = vDSP.maximum(mags)
        let p90  = percentile(mags, 0.90)
        let active = Float(mags.filter { $0 > 0.5 }.count) / Float(count)
        // TODO: directional coherence — placeholder 0 for now
        return [Double(mean), Double(std), Double(mx), Double(p90), Double(active), 0]
    }

    /// 12 spatial summaries (e.g., per-quadrant mean magnitude, L/R, top/bottom).
    /// TODO: implement to match pipeline/compact.compute_spatial_summary.
    static func spatialSummary(_ flow: CVPixelBuffer) -> [Double] {
        return Array(repeating: 0, count: 12)
    }

    private static func channels(_ buf: CVPixelBuffer) -> ([Float], [Float], Int) {
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
        let w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf)
        let count = w * h
        var dx = [Float](repeating: 0, count: count)
        var dy = [Float](repeating: 0, count: count)
        if let base = CVPixelBufferGetBaseAddress(buf) {
            let p = base.assumingMemoryBound(to: Float.self)  // interleaved dx,dy
            for i in 0..<count { dx[i] = p[2*i]; dy[i] = p[2*i+1] }
        }
        return (dx, dy, count)
    }
    private static func stddev(_ v: [Float], mean: Float) -> Float {
        var d = v.map { $0 - mean }; return (vDSP.sumOfSquares(d) / Float(v.count)).squareRoot()
    }
    private static func percentile(_ v: [Float], _ q: Float) -> Float {
        let s = v.sorted(); return s[min(s.count - 1, Int(q * Float(s.count)))]
    }
}
```

### FeatureExtractor.swift  ← THE PORT (main TODO)
```swift
import Foundation
import Accelerate

/// Reimplements pipeline/feature_battery.compute_all_features in Swift.
/// Order MUST match the classifier's feature_cols. Validate every value against
/// ios_export_feature_reference.json (tolerance ~1e-3) before enabling the model.
enum FeatureExtractor {
    static func compute(magTS: [[Double]], spatial: [[Double]], fps: Double) -> [(String, Double)] {
        let ts = magTS.map { $0.first ?? 0 }            // mean-magnitude time series
        var f: [(String, Double)] = []

        // --- Implemented examples (match the desktop battery) ---
        let mean = ts.reduce(0, +) / Double(max(ts.count, 1))
        let variance = ts.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(max(ts.count, 1))
        let std = variance.squareRoot()
        f.append(("flow_mean", mean))
        f.append(("flow_std", std))
        f.append(("flow_skew", skew(ts, mean: mean, std: std)))
        f.append(("flow_kurtosis", kurtosis(ts, mean: mean, std: std)))
        // kinetic-energy proxy from magnitude^2
        let ke = ts.map { $0 * $0 }
        f.append(("ke_mean", ke.reduce(0, +) / Double(max(ke.count, 1))))
        f.append(("ke_std", stddev(ke)))

        // --- TODO: port the rest to match Python exactly ---
        // sample_entropy       (feature_battery.sample_entropy)
        // spectral_entropy     (Welch PSD via vDSP FFT, then entropy of PSD)
        // peak_frequency       (argmax of PSD * fps scaling)
        // dfa_alpha            (detrended fluctuation analysis slope)
        // symmetry_mean        (left/right flow comparison from `spatial`)
        // mse_scale_1..N       (multiscale sample entropy)
        // ... plus any remaining columns in feature_cols
        // Until these are implemented + parity-checked, NmotionClassifier stays nil.

        return f
    }

    private static func skew(_ v: [Double], mean: Double, std: Double) -> Double {
        guard std > 0 else { return 0 }
        let m3 = v.map { pow(($0 - mean) / std, 3) }.reduce(0, +) / Double(v.count)
        return m3
    }
    private static func kurtosis(_ v: [Double], mean: Double, std: Double) -> Double {
        guard std > 0 else { return 0 }
        let m4 = v.map { pow(($0 - mean) / std, 4) }.reduce(0, +) / Double(v.count)
        return m4 - 3.0
    }
    private static func stddev(_ v: [Double]) -> Double {
        let m = v.reduce(0, +) / Double(max(v.count, 1))
        return (v.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(max(v.count, 1))).squareRoot()
    }
}
```

### NmotionClassifier.swift  (Core ML — enable after conversion + parity)
```swift
import CoreML

final class NmotionClassifier {
    // Set to nil to run features-only. After you add NmotionClassifier.mlmodel
    // and confirm feature parity, load it here and paste featureCols from
    // nmotion_feature_cols.json.
    static let shared: NmotionClassifier? = nil   // = try? NmotionClassifier()

    // Paste from build/nmotion_feature_cols.json (EXACT order):
    let featureCols: [String] = [ /* "sample_entropy", "spectral_entropy", ... */ ]
    private let model: MLModel

    init() throws {
        let url = Bundle.main.url(forResource: "NmotionClassifier", withExtension: "mlmodelc")!
        model = try MLModel(contentsOf: url)
    }

    func predict(features: [(String, Double)]) -> Prediction? {
        let lookup = Dictionary(uniqueKeysWithValues: features)
        var input = [String: Double]()
        for name in featureCols { input[name] = lookup[name] ?? 0 }
        guard let provider = try? MLDictionaryFeatureProvider(dictionary: input),
              let out = try? model.prediction(from: provider) else { return nil }

        // Core ML classifier exposes the top label + a probability dictionary.
        let label = out.featureValue(for: "classLabel")?.stringValue ?? "?"
        var probs: [(String, Double)] = []
        if let dict = out.featureValue(for: "classProbability")?.dictionaryValue as? [String: Double] {
            probs = dict.sorted { $0.value > $1.value }.map { ($0.key, $0.value) }
        }
        return Prediction(label: label, probabilities: probs, nClips: 1)
    }
}
```

### ResultsView.swift
```swift
import SwiftUI
import Charts

struct ResultsView: View {
    let result: AnalysisResult

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                if let p = result.prediction {
                    predictionCard(p)
                }
                Text("Movement intensity over time").font(.headline)
                Chart(Array(result.magnitudeSeries.enumerated()), id: \.offset) { i, v in
                    LineMark(x: .value("Frame", i), y: .value("Magnitude", v))
                }.frame(height: 180)

                Text("Features").font(.headline)
                ForEach(result.features, id: \.0) { name, value in
                    HStack {
                        Text(name).font(.system(.footnote, design: .monospaced))
                        Spacer()
                        Text(String(format: "%.4f", value)).foregroundStyle(.secondary)
                    }
                }
            }.padding()
        }
        .navigationTitle(result.clipName)
    }

    @ViewBuilder private func predictionCard(_ p: Prediction) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Predicted class").font(.headline)
            Text(p.label.uppercased()).font(.title2).bold()
            ForEach(p.probabilities, id: \.0) { name, prob in
                HStack {
                    Text(name).frame(width: 120, alignment: .leading).foregroundStyle(.secondary)
                    ProgressView(value: prob).tint(.blue)
                    Text(String(format: "%.0f%%", prob * 100)).frame(width: 44)
                }.font(.caption)
            }
            Text("Research tool — not a medical diagnosis. This is a statistical "
               + "classification and must not be used for clinical decisions.")
                .font(.caption2).foregroundStyle(.red).padding(.top, 4)
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(.secondarySystemBackground)))
    }
}
```

### CameraRecorderView.swift  (minimal recorder — stub to complete)
```swift
import SwiftUI
import AVFoundation

/// Minimal camera recorder. This is a stub: wire an AVCaptureSession with
/// AVCaptureMovieFileOutput, show the preview, and call `onFinish(url)` with the
/// recorded file. Kept short here so the rest of the app compiles; complete this
/// with a standard AVFoundation recording setup (many samples exist).
struct CameraRecorderView: View {
    var onFinish: (URL) -> Void
    var body: some View {
        VStack(spacing: 16) {
            Text("Camera recorder").font(.headline)
            Text("TODO: implement AVCaptureSession + movie output, then call onFinish(url).")
                .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }.padding()
    }
}
```

---

## Wiring the classifier (later)
1. Train + validate on desktop (`train.py`) so `models/nmotion_model.joblib` exists.
2. `python ios_convert_classifier_to_coreml.py` → `build/NmotionClassifier.mlmodel` + `nmotion_feature_cols.json`.
3. Drag the `.mlmodel` into Xcode; paste `feature_cols` into `NmotionClassifier.swift`;
   flip `shared` to `try? NmotionClassifier()`.
4. Only after `FeatureExtractor` passes parity vs `feature_reference.json`.

## Reality check
- Camera + Vision optical flow require a **real device** (not Simulator).
- The scaffold runs the pipeline end-to-end but returns a **partial feature set**
  until the battery is fully ported — so predictions stay off by design.
- Keep the disclaimer visible at all times; this is a research tool.
