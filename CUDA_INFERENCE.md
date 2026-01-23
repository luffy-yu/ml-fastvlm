# FastVLM CUDA Inference with MLX-VLM

This guide explains how to use the modified MLX-VLM library to run FastVLM inference on Ubuntu/Linux with NVIDIA GPUs.

## Setup

### 1. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install safetensors transformers huggingface_hub
```

### 2. Clone MLX-VLM

```bash
git clone https://github.com/anthropics/mlx-vlm.git
cd mlx-vlm
```

The CUDA backend is located in `mlx_vlm/cuda/`.

## Usage

### Basic Inference

```python
import sys
sys.path.insert(0, '/path/to/mlx-vlm')

from mlx_vlm import load, generate

# Load model from checkpoint
model, processor = load('/path/to/ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3')

# Run inference with an image
response = generate(
    model,
    processor,
    prompt="Describe this image in detail",
    image="/path/to/ml-fastvlm/000000039769.jpg",
    max_tokens=200,
    temp=0.0,
    verbose=True
)

print(response)
```

### Example Output

```
==========
Image: /path/to/000000039769.jpg
Prompt: Describe this image in detail

Generated: This image depicts two cats lying on a pink couch. The cat on the left
is a tabby with a striped pattern, and the cat on the right is a tabby with a more
mottled or striped pattern. Both cats are sleeping or resting peacefully...
==========
Prompt: 5 tokens, 14.9 tokens/sec
Generation: 100 tokens, 139.8 tokens/sec
```

## Model Quantization

### Export to 8-bit Quantized Format

```bash
cd /path/to/mlx-vlm

python -m mlx_vlm.convert \
    --hf-path /path/to/ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3 \
    --mlx-path /path/to/output/fastvlm_0.5b_8bit \
    -q \
    --q-bits 8
```

### Export to 4-bit Quantized Format

```bash
python -m mlx_vlm.convert \
    --hf-path /path/to/ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3 \
    --mlx-path /path/to/output/fastvlm_0.5b_4bit \
    -q \
    --q-bits 4
```

### Export LLM Only (Smaller File Size)

This exports only the language model and projector, without the vision tower weights:

```bash
python -m mlx_vlm.convert \
    --hf-path /path/to/ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3 \
    --mlx-path /path/to/output/fastvlm_0.5b_llm_8bit \
    --only-llm \
    -q \
    --q-bits 8
```

## Model Size Comparison

| Model | Original | 8-bit | 4-bit |
|-------|----------|-------|-------|
| FastVLM 0.5B | ~1.5 GB | ~627 MB | ~454 MB |

Note: Vision tower (~200 MB) is always kept in full precision.

## Available Checkpoints

```
ml-fastvlm/checkpoints/
├── llava-fastvithd_0.5b_stage3/   # 0.5B parameter model
├── llava-fastvithd_1.5b_stage3/   # 1.5B parameter model
└── llava-fastvithd_7b_stage3/     # 7B parameter model
```

## Configuration

### Vision Tower Path

The CUDA backend loads the MobileCLIP vision tower from this repository. If you move the repository, update the path in:

```
mlx-vlm/mlx_vlm/cuda/models/fastvlm/vision.py
```

Line ~390:
```python
fastvlm_path = "/path/to/ml-fastvlm"
```

### Image Size

FastVLM expects 1024x1024 images. The `generate()` function automatically resizes images.

## Troubleshooting

### Import Errors

If you see MLX-related import errors, ensure PyTorch is installed and CUDA is available:

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Memory Errors

For large models (7B), try:
1. Use 4-bit quantization
2. Reduce `max_tokens`
3. Use a GPU with more VRAM

### Vision Tower Not Loading

Ensure the ml-fastvlm repository path is correct and contains the `llava/` module:

```
ml-fastvlm/
├── llava/
│   └── model/
│       └── multimodal_encoder/
│           └── mobileclip_encoder.py
```

## Performance Tips

1. **Use greedy decoding** (`temp=0.0`) for faster, deterministic outputs
2. **Batch similar prompts** when processing multiple images
3. **Use 8-bit quantization** for a good balance of speed and quality
4. **Pre-load the model** once and reuse for multiple inferences

## API Quick Reference

```python
# Load model
model, processor = load(checkpoint_path)

# Generate with image
text = generate(model, processor, prompt, image=image_path, max_tokens=100)

# Generate text-only
text = generate(model, processor, prompt, max_tokens=100)
```
