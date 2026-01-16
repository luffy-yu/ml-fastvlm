#
# Export FastVLM model to FP16 format for CUDA inference
#
import os
import json
import copy
import argparse

import torch
from safetensors.torch import save_file

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path


def export(args):
    # Load model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    print(f"Loading model from {model_path}...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device="cuda"
    )

    # Create output directory
    output_path = args.output_path or os.path.join(os.path.dirname(model_path), f"{model_name}_fp16")
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

    # Copy model config and add image_token_index
    config_path = os.path.join(model_path, "config.json")
    model_config = json.load(open(config_path, 'r'))
    model_config["image_token_index"] = max(token_ids) + 1
    model_config["torch_dtype"] = "float16"
    output_config_path = os.path.join(output_path, "config.json")
    json.dump(model_config, open(output_config_path, 'w'), indent=2)
    print(f"Saved config.json")

    # Convert model to FP16
    print("Converting model to FP16...")
    model = model.half()  # Convert to FP16

    # Save using model.save_pretrained which handles LlavaQwen2 architecture properly
    print(f"Saving FP16 model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    # Calculate and print model size
    state_dict = model.state_dict()
    total_params = sum(p.numel() for p in state_dict.values())
    model_size_bytes = sum(p.numel() * p.element_size() for p in state_dict.values())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size (FP16): {model_size_bytes / (1024**3):.2f} GB ({model_size_bytes / (1024**2):.0f} MB)")

    print(f"\nFP16 export complete!")
    print(f"Output saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to input checkpoint")
    parser.add_argument("--output-path", type=str, default=None, help="Path for output (default: <model-path>_fp16)")
    parser.add_argument("--model-base", type=str, default=None)

    args = parser.parse_args()
    export(args)
