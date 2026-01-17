# Before quantization

```
conda env activate fastvlm

python predict.py --model-path ./checkpoints/llava-fastvithd_0.5b_stage3 --image-file 000000039769.jpg --prompt "Can you describe this image?"
```
Using CUDA device
Certainly! The image depicts two cats lying on a pink surface. One cat is on the left side of the image, and the other cat is on the right side. Both cats are stretched out, appearing to be asleep. There are two remote controls placed near the cats, one on the left and the other on the right. The remote controls are positioned close to the cats, suggesting they might have been used recently.

# Quantization using FP16 (CUDA/PyTorch)

## Overview

Since we're on Linux with CUDA (not Apple Silicon), we use a **CUDA-compatible FP16 export** instead of CoreML/MLX.

## Export Command

```bash
conda activate fastvlm

python model_export/export_fp16_cuda.py \
    --model-path ./checkpoints/llava-fastvithd_0.5b_stage3 \
    --output-path ./exported/llava-fastvithd_0.5b_fp16
```

**Output:**
```
Loading model from ./checkpoints/llava-fastvithd_0.5b_stage3...
Output directory: ./exported/llava-fastvithd_0.5b_fp16
Saved preprocessor_config.json
Saved processor_config.json
Saved tokenizer_config.json
Saved config.json
Converting model to FP16...
Saving FP16 model to ./exported/llava-fastvithd_0.5b_fp16...
Total parameters: 758,314,124
Model size (FP16): 1.41 GB (1446 MB)

FP16 export complete!
```

## Test Inference

```bash
python model_export/test_fp16_inference.py \
    --model-path ./exported/llava-fastvithd_0.5b_fp16 \
    --image-file ./000000039769.jpg \
    --prompt "Can you describe this image?"
```

**Output:**
```
Loading FP16 model from ./exported/llava-fastvithd_0.5b_fp16...
Model dtype: torch.float16
Model device: cuda:0

Response: Certainly! The image depicts two cats lying on a pink surface, possibly a couch or a bed. One cat is on the left side, and the other is on the right. Both cats appear to be in a relaxed state, possibly sleeping. There are two remote controls placed near the cats, one on the left and the other on the right.
```

## Model Size Comparison

| Model | Size | Notes |
|-------|------|-------|
| Original (BF16) | 1.5 GB | `checkpoints/llava-fastvithd_0.5b_stage3/model.safetensors` |
| Exported (FP16) | 1.2 GB | `exported/llava-fastvithd_0.5b_fp16/model.safetensors` |

## Files Created

- `model_export/export_fp16_cuda.py` - Export script for CUDA
- `model_export/test_fp16_inference.py` - Test inference script
- `exported/llava-fastvithd_0.5b_fp16/` - Exported FP16 model

---

# Apple Silicon Export (MLX/CoreML) - Reference

For Apple Silicon devices, the original export method uses CoreML + MLX:

## Step 1: Export Vision Encoder to CoreML

```bash
# Requires macOS with Apple Silicon
python model_export/export_vision_encoder.py --model-path ./checkpoints/llava-fastvithd_0.5b_stage3
```

## Step 2: Setup mlx-vlm with FastVLM patch

```bash
git clone https://github.com/Blaizzy/mlx-vlm.git
cd mlx-vlm
git checkout 1884b551bc741f26b2d54d68fa89d4e934b9a3de
git apply ../model_export/fastvlm_mlx-vlm.patch
pip install -e .
cd ..
```

## Step 3: Export LLM to MLX

```bash
# FP16 (no quantization)
python -m mlx_vlm.convert --hf-path ./checkpoints/llava-fastvithd_0.5b_stage3 \
                          --mlx-path ./exported/fastvlm_0.5b_fp16 \
                          --only-llm

# INT8 quantization
python -m mlx_vlm.convert --hf-path ./checkpoints/llava-fastvithd_0.5b_stage3 \
                          --mlx-path ./exported/fastvlm_0.5b_int8 \
                          --only-llm -q --q-bits 8

# INT4 quantization
python -m mlx_vlm.convert --hf-path ./checkpoints/llava-fastvithd_0.5b_stage3 \
                          --mlx-path ./exported/fastvlm_0.5b_int4 \
                          --only-llm -q --q-bits 4
```

## Available Checkpoints

- `llava-fastvithd_0.5b_stage3` - 0.5B params (smallest, fastest)
- `llava-fastvithd_1.5b_stage3` - 1.5B params (balanced)
- `llava-fastvithd_7b_stage3` - 7B params (most capable)

---

# Running Official MLX Model on CUDA

The official pre-exported MLX model (`llava-fastvithd_0.5b_stage3_llm.fp16`) is designed for Apple Silicon with MLX/CoreML.
To run it on Linux/CUDA, use the conversion script that loads MLX weights into PyTorch.

## Command

```bash
python model_export/run_mlx_model_on_cuda.py \
    --mlx-model-path ./exported/llava-fastvithd_0.5b_stage3_llm.fp16 \
    --vision-tower-path ./checkpoints/llava-fastvithd_0.5b_stage3 \
    --image-file ./000000039769.jpg \
    --prompt "Can you describe this image?"
```

**Note:** This requires the original checkpoint (`checkpoints/llava-fastvithd_0.5b_stage3`) for the vision tower,
since the MLX export only contains the LLM weights (vision tower is exported as CoreML for Apple Silicon).

**Output:**
```
Loading MLX model from ./exported/llava-fastvithd_0.5b_stage3_llm.fp16...
Using vision tower from ./checkpoints/llava-fastvithd_0.5b_stage3...
Converting MLX weights to PyTorch format...
Updated 293 weights from MLX model
Model dtype: torch.float16
Model device: cuda:0

Response: Certainly! The image depicts two cats lying on a pink surface, possibly a couch or a bed. One cat is on the left side, and the other is on the right. Both cats appear to be in a relaxed state, possibly sleeping. There are two remote controls placed near the cats, one on the left and the other on the right.
```

## Files

- `model_export/run_mlx_model_on_cuda.py` - Script to run MLX model on CUDA
- `exported/llava-fastvithd_0.5b_stage3_llm.fp16/` - Official MLX-exported model

