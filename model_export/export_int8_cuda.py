#
# Export FastVLM model to INT8 quantized format for CUDA inference
# Uses bitsandbytes for GPU-accelerated INT8 quantization
#
import os
import json
import copy
import argparse

import torch

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path

try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    print("Warning: bitsandbytes not installed. Install with: pip install bitsandbytes")


def export(args):
    if not HAS_BITSANDBYTES:
        raise RuntimeError("bitsandbytes is required for GPU INT8 quantization. Install with: pip install bitsandbytes")

    # Load model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    print(f"Loading model from {model_path}...")

    # Configure INT8 quantization using bitsandbytes
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,  # Threshold for outlier detection
        llm_int8_has_fp16_weight=False,
    )

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device="cuda",
        quantization_config=quantization_config,
    )

    print(f"Model loaded with INT8 quantization on CUDA")

    # Create output directory
    output_path = args.output_path or os.path.join(os.path.dirname(model_path), f"{model_name}_int8")
    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Save extra metadata that is not saved during LLaVA training
    # required by HF for auto-loading model and for mlx-vlm preprocessing

    # Save image processing config
    setattr(image_processor, "processor_class", "LlavaProcessor")
    preprocessor_path = os.path.join(output_path, "preprocessor_config.json")
    image_processor.to_json_file(preprocessor_path)
    print(f"Saved preprocessor_config.json")

    # Create processor config
    processor_config = dict()
    processor_config["image_token"] = "<image>"
    processor_config["num_additional_image_tokens"] = 0
    processor_config["processor_class"] = "LlavaProcessor"
    processor_config["patch_size"] = 64
    processor_config_path = os.path.join(output_path, "processor_config.json")
    json.dump(processor_config, open(processor_config_path, "w"), indent=2)
    print(f"Saved processor_config.json")

    # Copy and modify tokenizer config
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
    tokenizer_config = json.load(open(tokenizer_config_path, 'r'))
    token_ids = list()
    image_token_is_present = False
    for k, v in tokenizer_config['added_tokens_decoder'].items():
        token_ids.append(int(k))
        if v["content"] == "<image>":
            image_token_is_present = True
            token_ids.pop()

    # Append only if <image> token is not present
    if not image_token_is_present:
        tokenizer_config['added_tokens_decoder'][f'{max(token_ids) + 1}'] = copy.deepcopy(
            tokenizer_config['added_tokens_decoder'][f'{token_ids[0]}'])
        tokenizer_config['added_tokens_decoder'][f'{max(token_ids) + 1}']["content"] = "<image>"

    output_tokenizer_path = os.path.join(output_path, "tokenizer_config.json")
    json.dump(tokenizer_config, open(output_tokenizer_path, 'w'), indent=2)
    print(f"Saved tokenizer_config.json")

    # Copy model config and add image_token_index and quantization info
    config_path = os.path.join(model_path, "config.json")
    model_config = json.load(open(config_path, 'r'))
    model_config["image_token_index"] = max(token_ids) + 1
    model_config["quantization_config"] = {
        "load_in_8bit": True,
        "llm_int8_threshold": 6.0,
        "quant_method": "bitsandbytes"
    }
    output_config_path = os.path.join(output_path, "config.json")
    json.dump(model_config, open(output_config_path, 'w'), indent=2)
    print(f"Saved config.json")

    # Save the quantized model
    print(f"\nSaving INT8 quantized model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    # Calculate model size from saved files
    total_size = 0
    for filename in os.listdir(output_path):
        filepath = os.path.join(output_path, filename)
        if os.path.isfile(filepath) and (filename.endswith('.safetensors') or filename.endswith('.bin')):
            total_size += os.path.getsize(filepath)

    # Get original model size for comparison
    original_size = 0
    for filename in os.listdir(model_path):
        filepath = os.path.join(model_path, filename)
        if os.path.isfile(filepath) and (filename.endswith('.safetensors') or filename.endswith('.bin')):
            original_size += os.path.getsize(filepath)

    if original_size > 0:
        compression_ratio = original_size / total_size if total_size > 0 else 0
        print(f"\n{'='*50}")
        print(f"Quantization Summary:")
        print(f"{'='*50}")
        print(f"Original size:   {original_size / (1024**3):.2f} GB")
        print(f"Quantized size:  {total_size / (1024**3):.2f} GB ({total_size / (1024**2):.0f} MB)")
        print(f"Compression:     {compression_ratio:.2f}x")
        print(f"Size reduction:  {(1 - total_size/original_size) * 100:.1f}%")
        print(f"{'='*50}")

    print(f"\nINT8 export complete!")
    print(f"Output saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FastVLM model to INT8 quantized format (GPU)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to input checkpoint")
    parser.add_argument("--output-path", type=str, default=None, help="Path for output (default: <model-path>_int8)")
    parser.add_argument("--model-base", type=str, default=None)

    args = parser.parse_args()
    export(args)
