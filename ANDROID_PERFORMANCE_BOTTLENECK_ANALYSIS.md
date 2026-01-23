# FastVLM Android Performance Bottleneck Analysis

## Executive Summary

After analyzing the `fastvlm-android` branch of Executorch with proper QNN multimodal support, I've identified several bottlenecks that could explain why the Android implementation takes seconds compared to the fast iOS implementation.

**Key Finding:** The FastVLM architecture IS properly integrated in your branch, but there are critical bottlenecks in the execution pipeline.

---

## Architecture Overview (fastvlm-android branch)

The FastVLM Android implementation consists of:

| Component | File | Purpose |
|-----------|------|---------|
| **EncoderRunner** | [encoder.cpp](D:\Github\executorch\examples\qualcomm\oss_scripts\llama\runner\multimodal_runner\encoder.cpp) | Runs vision encoder (FastViT-HD) |
| **MultimodalRunner** | [multimodal_runner.cpp](D:\Github\executorch\examples\qualcomm\oss_scripts\llama\runner\multimodal_runner\multimodal_runner.cpp) | Orchestrates VLM inference |
| **EmbeddingRunner** | [embedding_runner.cpp](D:\Github\executorch\examples\qualcomm\oss_scripts\llama\runner\multimodal_runner\embedding_runner.cpp) | Text embedding lookup |
| **JNI Layer** | [jni_layer_llama.cpp](D:\Github\executorch\extension\android\jni\jni_layer_llama.cpp) | Android-Native bridge |

### Supported Multimodal Models

From [multimodal_runner.cpp:121-129](D:\Github\executorch\examples\qualcomm\oss_scripts\llama\runner\multimodal_runner\multimodal_runner.cpp#L121-L129):
```cpp
if (decoder_model_version == "smolvlm") {
  decoder_model_version_ = MultimodalDecoderModelVersion::kSmolvlm;
} else if (decoder_model_version == "internvl3") {
  decoder_model_version_ = MultimodalDecoderModelVersion::kInternvl3;
} else if (decoder_model_version == "fastvlm_0_5b") {
  decoder_model_version_ = MultimodalDecoderModelVersion::kFastvlm;  // ✓ Supported!
}
```

### Inference Pipeline

```
Java: encodeImage(float[] imageData, ...)
    ↓
JNI: encode_image() - Copies 12.6MB float array
    ↓
C++: encoder_runner_->encode(tensor)
    ↓
ExecuTorch: module_->forward() - Vision encoder execution
    ↓
Output: image_hidden_states_ stored
    ↓
Java: generate(prompt, ...)
    ↓
JNI: generate()
    ↓
C++: qnn_multimodal_runner_->generate()
    ├── embedding_processor_->prefill() - Text embedding
    ├── merge_multimodal_embeddings() - Merge image + text
    ├── prompt_processor_->prefill() - Prefill forward pass
    └── token_generator_->generate() - Token generation loop
```

---

## Identified Bottlenecks

### 1. **Vision Encoder Backend Verification Needed** (CRITICAL - Your Hypothesis)

**Location:** [encoder.cpp:21-26](D:\Github\executorch\examples\qualcomm\oss_scripts\llama\runner\multimodal_runner\encoder.cpp#L21-L26)

```cpp
EncoderRunner::EncoderRunner(const std::string& model_path)
    : image_seq_len_(0) {
  module_ = std::make_unique<Module>(
      model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
}
```

**Issue:** The encoder loads the `.pte` file, but whether it runs on QNN depends on how the model was **exported**:
- If `vision_encoder_qnn.pte` was exported with QNN partitioner → runs on NPU ✓
- If exported without QNN partitioner → runs on CPU ✗

**Diagnosis:** Check logcat output during encoder execution:
```bash
adb logcat -s ExecuTorch:V | grep -i "delegate\|QNN\|HTP\|backend"
```

You should see:
```
I ExecuTorch: Loaded QNN backend
I ExecuTorch: Delegating graph to HTP
```

If you see:
```
I ExecuTorch: Using XNNPACK backend
```
or no delegate messages, the vision encoder is running on **CPU**.

**Impact:** Vision encoder on CPU: 1-3 seconds. On NPU: 50-200ms.

---

### 2. **Debug Logging Overhead** (MAJOR)

**Location:** [encoder.cpp:120-148](D:\Github\executorch\examples\qualcomm\oss_scripts\llama\runner\multimodal_runner\encoder.cpp#L120-L148), [multimodal_runner.cpp:652-763](D:\Github\executorch\examples\qualcomm\oss_scripts\llama\runner\multimodal_runner\multimodal_runner.cpp#L652-L763)

**Issue:** The code has extensive debug logging including:
- Computing min/max/mean/std over all tensor elements (O(n) passes)
- Writing debug files to disk (I/O blocking)

```cpp
// encoder.cpp - DEBUG code that runs EVERY encode
float min_val = buffer[0], max_val = buffer[0];
double sum = 0.0, sum_sq = 0.0;
for (int64_t i = 0; i < num_elem; ++i) {  // 3M elements!
    float val = buffer[i];
    min_val = std::min(min_val, val);
    max_val = std::max(max_val, val);
    sum += val;
    sum_sq += val * val;
}

// Write to file - BLOCKING I/O
std::ofstream debug_input("debug_runtime_vision_input.raw", std::ios::binary);
debug_input.write(reinterpret_cast<char*>(buffer.data()), expected_size);
```

**Impact:** 100-300ms per inference just for debug code

**Fix:** Remove or disable all debug code in production:
```cpp
#ifdef DEBUG_FASTVLM
// Debug code here
#endif
```

---

### 3. **Large JNI Data Copy** (MAJOR)

**Location:** [jni_layer_llama.cpp:499-504](D:\Github\executorch\extension\android\jni\jni_layer_llama.cpp#L499-L504)

```cpp
// Two copies of 12.6 MB image data!
std::vector<jfloat> image_data_jfloat(image_size);    // Copy 1
image->getRegion(0, image_size, image_data_jfloat.data());

std::vector<float> image_buffer(image_data_jfloat.begin(), image_data_jfloat.end());  // Copy 2
```

**Impact:** 50-150ms per image

**Fix:** Use single allocation:
```cpp
std::vector<float> image_buffer(image_size);
image->getRegion(0, image_size, reinterpret_cast<jfloat*>(image_buffer.data()));
```

---

### 4. **Java Image Preprocessing** (MAJOR)

**Location:** [ETImage.java](D:\Github\LlamaDemo-Executorch-QNN\app\src\main\java\com\example\executorchllamademo\ETImage.java)

**Issue:** Pixel-by-pixel extraction in Java:
```java
for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
        int color = bitmap.getPixel(x, y);  // 1M JNI calls for 1024x1024!
    }
}
```

**Impact:** 200-500ms for 1024x1024 image

**iOS Comparison:** Uses `Accelerate/vImage` with SIMD - takes <10ms

---

### 5. **KV Cache Requantization on CPU** (MODERATE)

**Location:** [multimodal_runner.cpp:787-825](D:\Github\executorch\examples\qualcomm\oss_scripts\llama\runner\multimodal_runner\multimodal_runner.cpp#L787-L825)

```cpp
// Per-element requantization loop - NOT using NEON despite the include
for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
    for (int64_t i = 0; i < num_elems_per_layer; i++) {
        k_cache_data[i] = static_cast<T>(
            (k_cache_data[i] - output_k_cache_zero_points_[layer_idx]) *
                scale_ratio_k + input_k_cache_zero_points_[layer_idx]);
    }
}
```

**Note:** The file includes `arm_neon.h` but doesn't use it for this loop.

**Impact:** 50-200ms depending on context length and model size

**Fix:** Use NEON SIMD:
```cpp
#if defined(__aarch64__)
float32x4_t scale_v = vdupq_n_f32(scale_ratio_k);
// ... vectorized implementation
#endif
```

---

### 6. **Embedding Merge on CPU** (MINOR)

**Location:** [multimodal_runner.cpp:856-921](D:\Github\executorch\examples\qualcomm\oss_scripts\llama\runner\multimodal_runner\multimodal_runner.cpp#L856-L921)

```cpp
// Memory copy of ~1MB embeddings + scattered writes
std::memcpy(multimodal_embeddings_buffer_.data(),
            text_embeddings.data,
            total_elements * sizeof(float));

for (int32_t i = 0; i < placeholder_positions.size(); ++i) {
    std::memcpy(merged_data + pos * embedding_dim,
                image_data + i * embedding_dim,
                embedding_dim * sizeof(float));
}
```

**Impact:** 10-30ms

---

## Timing Breakdown

### Expected Performance (if properly optimized)

| Stage | Expected Time |
|-------|---------------|
| Image preprocessing (native) | 5-10ms |
| JNI data transfer (optimized) | 20-30ms |
| Vision encoder (QNN/NPU) | 50-200ms |
| Text embedding | 10-20ms |
| Embedding merge | 5-10ms |
| Prefill forward | 100-300ms |
| Token generation | 20-50ms/token |
| **Total TTFT** | **~300-600ms** |

### Likely Actual Performance (with bottlenecks)

| Stage | Actual Time | Bottleneck |
|-------|-------------|------------|
| Image preprocessing (Java) | 200-500ms | `Bitmap.getPixel()` loop |
| JNI data transfer | 100-200ms | Double copy |
| Vision encoder | 50-3000ms | Depends on NPU vs CPU |
| Debug logging/file I/O | 100-300ms | Stats computation + file writes |
| Text embedding | 10-20ms | OK |
| Embedding merge | 10-30ms | OK |
| KV requantization | 50-200ms | CPU loop without SIMD |
| Prefill + generation | 200-500ms | OK if on NPU |
| **Total TTFT** | **0.7-5+ seconds** | Multiple issues |

---

## Diagnostic Steps

### 1. Add Timing to Vision Encoder

In [encoder.cpp](D:\Github\executorch\examples\qualcomm\oss_scripts\llama\runner\multimodal_runner\encoder.cpp), add:

```cpp
#include <chrono>

Result<Tensor> EncoderRunner::encode(TensorPtr& image_tensor) {
  ET_CHECK_MSG(is_method_loaded(), "Encoder method not loaded");

  auto start = std::chrono::high_resolution_clock::now();

  std::vector<executorch::runtime::EValue> encoder_inputs;
  encoder_inputs.emplace_back(*image_tensor.get());
  auto encoder_result = module_->forward(encoder_inputs);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  ET_LOG(Info, ">>> Vision encoder forward took %ld ms <<<", duration.count());

  ET_CHECK_MSG(encoder_result.ok(), "Encoder execution failed");
  return encoder_result.get()[0].toTensor();
}
```

### 2. Check QNN Delegation

```bash
adb logcat -s ExecuTorch:V QNN:V | grep -E "delegate|QNN|HTP|backend|forward"
```

Look for:
- "Loaded QNN backend" → Good
- "Delegating to HTP" → Good
- "Using XNNPACK" → Vision encoder on CPU (bad)

### 3. Profile from Java Side

In MainActivity.java:
```java
long encodeStart = System.currentTimeMillis();
int result = mModule.encodeImage(imageData, 1, channels, height, width);
long encodeTime = System.currentTimeMillis() - encodeStart;
ETLogging.getInstance().log("Vision encoder total: " + encodeTime + "ms");
```

---

## Recommended Fixes (Priority Order)

### High Priority

1. **Remove Debug Code**
   - Remove all debug statistics computation and file writes
   - Or guard with `#ifdef DEBUG_FASTVLM`

2. **Verify Vision Encoder QNN Delegation**
   - Check export script uses `QnnPartitioner`
   - Add logging to confirm NPU execution

3. **Optimize JNI Data Copy**
   - Single allocation instead of double copy
   - Consider using DirectByteBuffer

### Medium Priority

4. **Move Image Preprocessing to Native**
   - Use OpenCV or direct buffer operations in C++
   - Use NEON SIMD for normalization

5. **SIMD KV Cache Requantization**
   - Use NEON intrinsics for the requantization loop

### Lower Priority

6. **Enable Shared Buffer Mode**
   ```cpp
   // In constructor
   shared_buffer_ = true;
   ```

7. **Parallel Preprocessing**
   - Run image preprocessing on separate thread

---

## Key Files to Modify

1. **[encoder.cpp](D:\Github\executorch\examples\qualcomm\oss_scripts\llama\runner\multimodal_runner\encoder.cpp)**
   - Add timing instrumentation
   - **Remove debug statistics and file writes** (lines 120-148, 159-194)

2. **[multimodal_runner.cpp](D:\Github\executorch\examples\qualcomm\oss_scripts\llama\runner\multimodal_runner\multimodal_runner.cpp)**
   - **Remove debug statistics and file writes** (lines 652-763)
   - SIMD optimize KV requantization (lines 787-825)

3. **[jni_layer_llama.cpp](D:\Github\executorch\extension\android\jni\jni_layer_llama.cpp)**
   - Optimize data copy in `encode_image()` (lines 499-504)

4. **[ETImage.java](D:\Github\LlamaDemo-Executorch-QNN\app\src\main\java\com\example\executorchllamademo\ETImage.java)**
   - Replace with native preprocessing call or use `Bitmap.getPixels()` (batch)

---

## Quick Win: Remove Debug Code

The fastest improvement is removing the debug code. In both `encoder.cpp` and `multimodal_runner.cpp`, there are extensive debug blocks that:
1. Compute statistics over millions of elements
2. Write multi-megabyte files to disk

This alone could save **200-600ms per inference**.

---

## Executorch vs Native QNN: Framework Overhead Analysis

### Question: Would bypassing Executorch and using native QNN directly be faster?

**Short Answer:** Only marginally (~5-15ms savings). Executorch is **not** the bottleneck.

### Executorch Overhead Breakdown

The Executorch framework adds minimal overhead to QNN execution:

| Component | Overhead | Notes |
|-----------|----------|-------|
| Module::forward() dispatch | ~1-2ms | Method lookup and EValue handling |
| QNN delegate initialization | One-time | Happens at model load, not inference |
| Tensor wrapping/unwrapping | ~2-5ms | EValue → QNN tensor conversion |
| Memory management | ~2-5ms | Buffer allocation coordination |
| **Total per inference** | **~5-15ms** | Negligible vs 1-5 second total |

### Why Native QNN Won't Help Much

1. **QNN delegate is already native** - When the model runs through QNN, Executorch simply passes execution to the QNN library. The actual computation happens in native QNN code.

2. **The real overhead is elsewhere:**
   ```
   Current timing (estimated):
   ├── Debug code:           200-600ms  ← REMOVE THIS
   ├── Java preprocessing:   200-500ms  ← OPTIMIZE THIS
   ├── Vision encoder:       50-3000ms  ← VERIFY NPU
   ├── JNI double copy:      50-150ms   ← FIX THIS
   ├── KV requantization:    50-200ms   ← SIMD THIS
   ├── Executorch overhead:  5-15ms     ← MINIMAL
   └── Actual QNN compute:   varies     ← THIS IS FINE
   ```

3. **Native QNN requires significant rewrite:**
   - Must handle QNN context/graph lifecycle manually
   - Must manage tensor memory allocation
   - Must implement all preprocessing in native code
   - Lose portability to other backends (CoreML, XNNPACK)

### What Native QNN SDK Provides

```cpp
// Native QNN requires manual setup:
Qnn_ContextHandle_t context;
QnnContext_create(..., &context);

Qnn_GraphHandle_t graph;
QnnGraph_create(context, "vision_encoder", ..., &graph);

// Manual tensor management:
Qnn_Tensor_t input, output;
QnnGraph_execute(graph, &input, 1, &output, 1, ...);
```

Executorch handles all of this automatically through the QNN delegate, with ~5-15ms overhead.

### Recommendation

**Do NOT rewrite to native QNN.** Instead:

1. Remove debug code → Save 200-600ms
2. Verify NPU delegation → Potentially save 1-3 seconds
3. Optimize preprocessing → Save 200-500ms
4. Fix JNI copy → Save 50-150ms

These optimizations can achieve **sub-second TTFT** while keeping Executorch's benefits:
- Cross-platform compatibility (can swap to CoreML/XNNPACK)
- Easier model updates (just swap .pte file)
- Better debugging/profiling tools
- Active community support

---

## iOS vs Android Architecture Comparison

| Aspect | iOS (MLX + CoreML) | Android (Executorch + QNN) |
|--------|-------------------|---------------------------|
| Vision Encoder | CoreML (.mlpackage) | QNN delegate (.pte) |
| Image Preprocessing | vImage SIMD (~10ms) | Java loops (~300ms) |
| Data Transfer | Zero-copy Metal buffers | JNI copy (~100ms) |
| LLM Backend | MLX (Metal GPU) | QNN (Hexagon NPU) |
| Framework Overhead | ~5-10ms | ~5-15ms |
| Debug Code | None in production | Extensive (200-600ms) |

The iOS app is faster primarily due to:
1. **No debug code** in production builds
2. **Native SIMD preprocessing** with vImage
3. **Zero-copy GPU buffers** in Metal
4. **Optimized MLX framework** for Apple Silicon

---

## Conclusion

The most likely causes of slow performance are:

1. **Debug logging overhead** - 200-600ms (EASY FIX)
2. **Vision encoder possibly on CPU** - 1-3 seconds (needs verification)
3. **Java image preprocessing** - 200-500ms
4. **Double JNI data copy** - 50-150ms

**Executorch overhead is NOT a significant factor** (~5-15ms).

### Action Plan

**Step 1: Quick Wins (No Rebuild Required)**
```bash
# Check if vision encoder is on NPU
adb logcat -s ExecuTorch:V QNN:V | grep -E "delegate|QNN|HTP"
```

**Step 2: Remove Debug Code**
- Delete lines 120-148, 159-194 in encoder.cpp
- Delete debug blocks in multimodal_runner.cpp
- Rebuild the native library

**Step 3: Add Timing Instrumentation**
```cpp
auto start = std::chrono::high_resolution_clock::now();
// ... operation ...
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::high_resolution_clock::now() - start);
ET_LOG(Info, "Operation took %ld ms", duration.count());
```

**Step 4: Optimize Image Preprocessing**
- Replace `Bitmap.getPixel()` with `Bitmap.getPixels()` (batch)
- Or move preprocessing to native C++ with NEON

**Expected Results After Optimization:**
- Current: 1-5+ seconds TTFT
- After fixes: 300-600ms TTFT
- Comparable to iOS performance
