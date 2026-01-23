# FastVLM Android Adaptation Analysis

This document explores the possibility of adapting the FastVLM iOS app to Android using frameworks such as QNN (Qualcomm Neural Network SDK).

## Current iOS Architecture Summary

The FastVLM app is built with:

| Component | Framework | Purpose |
|-----------|-----------|---------|
| ML Inference | MLX (Apple's ML framework) | Language model inference |
| Vision Encoder | CoreML (.mlpackage) | Image feature extraction |
| UI | SwiftUI | User interface |
| Camera | AVFoundation | Camera capture and video handling |
| Image Processing | CoreImage + Accelerate/vImage | Preprocessing pipeline |

### Project Structure

```
app/
├── FastVLM/                     # Core ML framework
│   ├── FastVLM.swift           # Main model implementation
│   ├── MediaProcessingExtensions.swift  # Image processing
│   └── model/fastvithd.mlpackage        # CoreML vision encoder
├── FastVLM App/                # Main app target
│   ├── ContentView.swift       # Main UI
│   ├── FastVLMModel.swift      # Model orchestration
│   └── ...
├── Video/                      # Camera framework
│   ├── CameraController.swift  # AVCapture integration
│   └── ...
└── FastVLM.xcodeproj           # Xcode project
```

### Model Inference Flow

```
Input Image (CVImageBuffer)
    ↓
Image Preprocessing (resize, crop, normalize)
    ↓
CoreML Vision Model (fastvithd) → [1, 256, 3072] features
    ↓
Multi-Modal Projector (MLX)
    ↓
Language Model (Qwen2-based, MLX)
    ↓
Token Generation (autoregressive)
    ↓
Output: Text response
```

---

## Android Adaptation Using QNN

**QNN (Qualcomm Neural Network SDK)** is a strong candidate for Android since it provides:
- Hardware acceleration on Snapdragon NPU/GPU/DSP
- Optimized inference for vision and language models
- Support for quantized models (INT8, INT4) which this app already uses

### Component Migration Strategy

| Component | iOS (Current) | Android (Proposed) |
|-----------|--------------|-------------------|
| Vision Encoder | CoreML (.mlpackage) | QNN (.so/.bin) or ONNX Runtime |
| Language Model | MLX | QNN / ONNX Runtime / llama.cpp |
| Image Processing | CoreImage + Accelerate | OpenCV or Android RenderScript |
| Camera | AVFoundation | Camera2 API |
| UI | SwiftUI | Jetpack Compose |
| Build System | Xcode | Gradle + CMake |

---

## Portability Assessment

### Highly Portable Components

1. **Language model architecture** - Qwen2-style transformer is standard
2. **Tokenizer logic** - JSON config files are platform-agnostic
3. **Text processing and prompt generation**
4. **Generation loop and sampling strategies**
5. **Model weights** - safetensors format is cross-platform

### Moderate Portability (Need Alternatives)

1. **Vision encoder** - CoreML only, needs conversion to ONNX/QNN
2. **Image processing** - CoreImage + Accelerate needs OpenCV replacement
3. **Camera capture** - AVFoundation → Android Camera2 API

### Not Portable (Requires Rewrite)

1. **SwiftUI UI framework** - Complete Android UI rewrite needed
2. **Xcode build system** - Needs Gradle/CMake
3. **CoreML model format** - Needs conversion to ONNX/QNN

---

## Model Conversion Steps

### 1. Vision Encoder (FastViT-HD)

```bash
# Step 1: Export CoreML to ONNX
python -m coremltools.converters.onnx fastvithd.mlpackage -o fastvithd.onnx

# Step 2: Convert ONNX to QNN
qnn-onnx-converter \
    --input_network fastvithd.onnx \
    --output_path fastvithd_qnn \
    --input_list input_list.txt
```

### 2. Language Model (Qwen2-based)

Options:
- **QNN Direct**: Convert safetensors → ONNX → QNN format
- **llama.cpp**: Use GGUF format with QNN backend (experimental)
- **ONNX Runtime**: Use with QNN Execution Provider

### 3. Model Quantization

Current iOS models and QNN equivalents:

| Model | iOS Format | QNN Target |
|-------|-----------|------------|
| 0.5B | FP16 | FP16 or INT8 |
| 1.5B | INT8 | INT8 |
| 7B | INT4 | INT4 (W4A16) |

---

## Alternative Frameworks Comparison

| Framework | Pros | Cons | Best For |
|-----------|------|------|----------|
| **QNN** | Best Snapdragon performance, NPU access | Qualcomm-only | Flagship Android devices |
| **ONNX Runtime** | Cross-platform, QNN backend available | Less optimized than native QNN | Wide device compatibility |
| **TensorFlow Lite** | Wide device support, GPU delegate | May be slower than QNN | Budget devices |
| **MNN (Alibaba)** | Good mobile optimization | Less community support | Chinese market |
| **NCNN (Tencent)** | Lightweight, Vulkan support | Manual model conversion | Resource-constrained devices |
| **llama.cpp** | Active community, many backends | Primarily for LLMs | Quick prototyping |

---

## Recommended Architecture for Android

```
┌─────────────────────────────────────────────────────────┐
│                    Jetpack Compose UI                    │
├─────────────────────────────────────────────────────────┤
│                     Camera2 API                          │
├─────────────────────────────────────────────────────────┤
│              OpenCV Image Preprocessing                  │
├──────────────────────┬──────────────────────────────────┤
│   Vision Encoder     │      Language Model               │
│   (QNN / ONNX)       │      (QNN / llama.cpp)           │
├──────────────────────┴──────────────────────────────────┤
│           Qualcomm NPU / GPU / CPU                       │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

### Phase 1: Model Conversion
- [ ] Export vision encoder to ONNX format
- [ ] Convert ONNX to QNN format
- [ ] Convert language model to ONNX/GGUF
- [ ] Validate model outputs match iOS version

### Phase 2: Core Inference Layer
- [ ] Set up Android NDK project with QNN SDK
- [ ] Implement vision encoder inference
- [ ] Implement language model inference
- [ ] Implement KV cache management
- [ ] Benchmark performance on target devices

### Phase 3: Image Processing
- [ ] Port preprocessing pipeline to OpenCV
- [ ] Implement resize, crop, normalize operations
- [ ] Match iOS preprocessing exactly (critical for accuracy)

### Phase 4: Camera Integration
- [ ] Implement Camera2 API capture
- [ ] Handle frame buffer conversion
- [ ] Implement camera controls (front/back, flash)

### Phase 5: UI Development
- [ ] Build Jetpack Compose UI matching iOS design
- [ ] Implement streaming text output
- [ ] Add model selection and settings
- [ ] Performance metrics display (TTFT, tokens/sec)

### Phase 6: Optimization
- [ ] Profile and optimize inference pipeline
- [ ] Implement batch processing if beneficial
- [ ] Memory optimization for large models
- [ ] Power consumption optimization

---

## Feasibility Summary

**Overall Assessment: FEASIBLE with significant effort**

| Aspect | Difficulty | Notes |
|--------|------------|-------|
| Model Conversion | Medium | Standard ONNX pipeline available |
| Vision Encoder Port | Medium | Well-defined input/output |
| LLM Port | Medium-High | KV cache and attention need careful porting |
| Image Preprocessing | Low | OpenCV has equivalent functions |
| Camera Integration | Medium | Camera2 API is well-documented |
| UI Rewrite | High | Complete rewrite in Jetpack Compose |
| Performance Parity | Medium-High | Depends on device and optimization |

**Key Success Factors:**
1. Access to Qualcomm AI SDK and documentation
2. Target device selection (Snapdragon 8 Gen 2+ recommended)
3. Careful validation of model output accuracy after conversion
4. Performance testing across device tiers

---

## References

- [Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
- [ONNX Runtime QNN Execution Provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [Android Camera2 API](https://developer.android.com/reference/android/hardware/camera2/package-summary)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [OpenCV Android](https://opencv.org/android/)
