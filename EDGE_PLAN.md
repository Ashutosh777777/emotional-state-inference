# Edge / Offline Deployment Plan — ArvyaX

> How the ArvyaX emotional intelligence system runs on mobile, on-device, with acceptable latency and model size.

---

## Overview

The system must work:
- **Offline** (no network required after install)
- **On low-end Android/iOS devices**
- **With <200ms latency** for real-time responsiveness
- **With <50MB total model footprint**

---

## Current Pipeline Components

| Component | Size (approx) | Latency |
|---|---|---|
| TF-IDF vectorizer (200 features) | ~50KB | <1ms |
| Random Forest (300 trees) | ~8MB | ~15ms |
| Gradient Boosting (intensity) | ~3MB | ~5ms |
| Label encoders + metadata | <1MB | <1ms |
| **Total** | **~12MB** | **~22ms** |

✅ Already fits within mobile constraints without any optimization.

---

## Deployment Architecture

```
Mobile App
│
├── On-Device Inference Engine
│   ├── TF-IDF vectorizer (serialized, ~50KB)
│   ├── RF Classifier → emotional state (ONNX, ~6MB)
│   ├── GB Classifier → intensity (ONNX, ~2MB)
│   └── Decision Rules (pure Python / JSON config)
│
├── Input Layer
│   ├── Text input (journal)
│   └── Contextual sliders (sleep, stress, energy)
│
└── Output Layer
    ├── Predicted state + intensity
    ├── What to do + when
    ├── Confidence badge (🟢 / 🟡 / 🔴)
    └── Supportive message
```

---

## Step-by-Step Deployment

### Step 1 — Export to ONNX

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Export state classifier
initial_type = [("input", FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(rf_base_model, initial_types=initial_type)
with open("state_clf.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

### Step 2 — Mobile Runtime

**Android:** Use ONNX Runtime for Android (`onnxruntime-android`, ~3MB AAR)  
**iOS:** Use ONNX Runtime for iOS or CoreML (convert via `onnxmltools`)  
**React Native / Flutter:** Both support ONNX Runtime via bindings

### Step 3 — Quantization (Optional, if size is tight)

```python
# INT8 quantization reduces model size ~4×
from onnxruntime.quantization import quantize_dynamic
quantize_dynamic("state_clf.onnx", "state_clf_int8.onnx")
# Result: ~2MB instead of 8MB
```

### Step 4 — TF-IDF on Device

The TF-IDF vocabulary (200 words + bigrams) is serialized to a JSON file:
```json
{
  "vocabulary": {"calm": 0, "anxious": 1, ...},
  "idf_weights": [2.1, 3.4, ...]
}
```
This runs natively in any language (Swift/Kotlin/Dart) with ~5 lines of code.

### Step 5 — Decision Engine

Decision rules are a JSON config file (`rules.json`), evaluated client-side:
```json
{
  "anxious": { "what": "box_breathing", "urgent_when": "now" },
  "overwhelmed": { "what": "grounding", "when_intensity_lt_4": "pause" }
}
```
No ML required — pure logic, zero latency.

---

## Model Size Optimization

| Strategy | Before | After | Tradeoff |
|---|---|---|---|
| ONNX export | ~8MB pkl | ~6MB | None |
| INT8 quantization | ~6MB | ~2MB | Tiny accuracy drop (<1%) |
| Reduce trees (200→100) | ~8MB | ~4MB | ~2% F1 drop |
| Reduce TF-IDF (200→100 features) | ~50KB | ~25KB | ~3% F1 drop |
| **Recommended config** | | **~4MB total** | Acceptable |

---

## Latency Budget

| Step | Latency | Notes |
|---|---|---|
| Text preprocessing (TF-IDF) | <1ms | Pure arithmetic |
| ONNX inference (RF) | ~8ms | On mid-range phone |
| ONNX inference (GB intensity) | ~3ms | Faster than RF |
| Decision engine (rules) | <0.1ms | JSON lookup |
| Message generation | <0.1ms | Template fill |
| **Total** | **~12ms** | Well within 200ms budget |

---

## Offline-First Strategy

1. **All models bundled** in the app package — no download needed
2. **No API calls** — inference is fully local
3. **Periodic model updates** via silent background sync (delta updates only)
4. **Graceful degradation**: if the session completes offline, predictions are queued locally and synced later

---

## Privacy Considerations

- Journal text **never leaves the device** (processed locally)
- Predictions stored in encrypted local DB (SQLite + SQLCipher)
- Opt-in telemetry: user can allow anonymized prediction logs for model improvement
- Face emotion hints processed via on-device vision (Apple Face ID API / ML Kit)

---

## Alternative: SLM (Small Language Model) Approach

For the **supportive message generation**, a small local LLM can replace templates:

| Model | Size | Speed (CPU) | Notes |
|---|---|---|---|
| Phi-2 (Microsoft) | 2.7B / ~5GB | ~2s/token | Too large for real-time |
| TinyLlama | 1.1B / ~700MB | ~0.5s/token | Acceptable for async |
| **MiniLM (sentence transformer)** | **80MB** | **<10ms** | **✅ Recommended for semantic matching** |
| distilbert-base | 268MB | ~30ms | Good for classification |

**Recommendation:** Use MiniLM for sentence embeddings (replacing TF-IDF) + keep template-based message generation. This gives:
- Better emotional understanding (semantic similarity)
- Fast, deterministic supportive messages
- Total model footprint: ~80MB + 4MB = ~84MB

---

## Tradeoffs Summary

| Dimension | Current (sklearn pkl) | Optimized Edge (ONNX INT8) | SLM Upgrade |
|---|---|---|---|
| Size | 12MB | 4MB | ~84MB |
| Latency | 22ms | 12ms | 50-200ms |
| Accuracy | Baseline | -1% | +5-10% |
| Offline | ✅ | ✅ | ✅ |
| Privacy | ✅ | ✅ | ✅ |
| Complexity | Low | Medium | High |

---

## Recommended Path

1. **Phase 1 (Now):** Deploy sklearn → ONNX pipeline as described. 4MB, <15ms, works offline.
2. **Phase 2 (Month 2):** Replace TF-IDF with MiniLM embeddings. Better understanding, ~84MB.
3. **Phase 3 (Month 4):** Fine-tune TinyLlama on domain-specific supportive messages for richer guidance.
