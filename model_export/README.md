# Model Export for inference on Apple Silicon
Disclaimer: this is not an official recommendation, just research and exploration. 

## Export Vision Encoder
We found that LLaVA trainer does not save all the states needed for auto inference, 
predominantly used in third party libraries like `mlx-vlm`. We save additional metadata
to model checkpoint directory and export the vision model using coremltools. 
Export vision encoder and patch the checkpoint using the instruction below. 
```bash
python export_vision_encoder.py --model-path /path/to/fastvlm-checkpoint
```

## Export VLM 

### Install mlx-vlm
We provide a patch to `mlx-vlm` to support inference of FastVLM.
```bash
git clone https://github.com/Blaizzy/mlx-vlm.git
cd mlx-vlm 
git checkout 1884b551bc741f26b2d54d68fa89d4e934b9a3de
git apply ../fastvlm_mlx-vlm.patch
pip install -e .
```

Export model using the following instruction.
```bash
python -m mlx_vlm.convert --hf-path  /path/to/fastvlm-checkpoint \
                          --mlx-path /path/to/exported-fastvlm \
                          --only-llm
```
To quantize the LLM, additional options can be provided as shown below.
`--q-bits` specifies bits per weight, the command below exports the LLM with 8-bit quantization. 
```bash
python -m mlx_vlm.convert --hf-path  /path/to/fastvlm-checkpoint \
                          --mlx-path /path/to/exported-fastvlm \
                          --only-llm \
                          -q \
                          --q-bits 8       # For 4-bit quantization, specify 4
```

### Generate
The exported model can be used for inference in a python environment following the instruction below.
```bash
python -m mlx_vlm.generate --model /path/to/exported-fastvlm \
                           --image /path/to/image.png \
                           --prompt "Describe the image." \ 
                           --max-tokens 256 \
                           --temp 0.0
```

## Troubleshooting
We noticed that sometimes `config.json` for the LLaVA model incorrectly sets the value for `tie_word_embeddings`.
This causes the following error during conversion, `ValueError: Received parameters not in model: language_model.lm_head.weight.`
If you encounter this error, set the value of `tie_word_embeddings` accordingly.

## Export fp16 using CUDA

- Export

```
python model_export/export_fp16_cuda.py \
    --model-path ./checkpoints/llava-fastvithd_0.5b_stage3 \
    --output-path ./exported/llava-fastvithd_0.5b_fp16
```

- Test

```
python model_export/test_fp16_inference.py \
    --model-path ./exported/llava-fastvithd_0.5b_fp16 \
    --image-file ./000000039769.jpg \
    --prompt "Can you describe this image?"
```

## Export int8 using CUDA

- Export

```
python model_export/export_int8_cuda.py \
    --model-path ./checkpoints/llava-fastvithd_0.5b_stage3 \
    --output-path ./exported/llava-fastvithd_0.5b_int8
```

- Test

```
python model_export/test_int8_inference.py \
    --model-path ./exported/llava-fastvithd_0.5b_int8 \
    --image-file ./000000039769.jpg \
    --prompt "Can you describe this image?"
```

## Export using adapted MLX on Linux

- Export to MLX

```
python -m mlx_vlm.convert \
	--hf-path ./checkpoints/llava-fastvithd_0.5b_stage3 \
	--mlx-path ./exported/fastvlm_0.5b_8bit \
	-q \
	--q-bits 8
```

- Test MLX Output

```
python predict_mlxvlm.py \
    --model-path ./exported/fastvlm_0.5b_8bit \
    --image-file ./000000039769.jpg \
    --prompt "Describe this image in detail" \
    --temperature 0.0 \
    --max-new-tokens 256 \
    --verbose
```

```
Loading model from ./exported/fastvlm_0.5b_8bit...
[INFO] Found 170 quantized layers to dequantize
[INFO] Loading vision tower with 653 weights
[INFO] Found 653 vision tower weights with prefix 'vision_tower.model.vision_tower.model.'
[INFO] Found 12 quantized vision tower layers, dequantizing...
[INFO] Dequantized 12 vision tower layers
[INFO] Adding 'model.' prefix to vision weights
[INFO] Successfully loaded vision tower using original LLaVA implementation
Model loaded in 2.46s

Image: ./000000039769.jpg
Prompt: Describe this image in detail
Temperature: 0.0
Max tokens: 256

Generating response...
==========
Image: ./000000039769.jpg
Prompt: Describe this image in detail

Generated: :
In this image, two cats are sleeping on a pink couch. The cat on the left is a tabby with a striped pattern, and the cat on the right is a tabby with a more mottled, striped pattern. Both cats have their eyes closed and are in a relaxed, sleeping position. The tabby on the left has a green collar, while the tabby on the right has a brown collar. There are two remote controls on the couch, one in front of each cat. The remote controls are white with multiple buttons. The couch is a bright pink color, and the cats' fur is a mix of dark and light stripes. The image captures a peaceful moment of the cats resting on the couch.
==========
Prompt: 7 tokens, 40.8 tokens/sec
Generation: 148 tokens, 143.5 tokens/sec
```

- Test Original

```
python predict_mlxvlm.py \
    --model-path ./checkpoints/llava-fastvithd_0.5b_stage3 \
    --image-file ./000000039769.jpg \
    --prompt "Describe this image in detail" \
    --temperature 0.0 \
    --max-new-tokens 256 \
    --verbose
```

- Export to PyTorch

```
python -m mlx_vlm.convert \
    --hf-path ./checkpoints/llava-fastvithd_0.5b_stage3 \
    --pytorch-path ./exported/fastvlm_0.5b_8bit_torch \
    -q --q-bits 8 \
    --skip-vision  # Recommended for FastVLM
```

- Test PyTorch Output

```
python predict_mlxvlm.py \
    --model-path ./exported/fastvlm_0.5b_8bit_torch \
    --image-file ./000000039769.jpg \
    --prompt "Describe this image in detail" \
    --temperature 0.0 \
    --max-new-tokens 256 \
    --verbose
```