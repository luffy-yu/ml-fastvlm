#
# Run official MLX-exported FastVLM model on CUDA
# This script converts MLX weights to PyTorch format and uses the vision tower from original checkpoint
#
import os
import argparse

import torch
from safetensors import safe_open

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from PIL import Image


def convert_mlx_weights_to_pytorch(mlx_weights_path):
    """Convert MLX weight names to PyTorch format."""
    weights = {}
    with safe_open(mlx_weights_path, framework='pt') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)

            # Convert MLX naming to PyTorch naming
            new_key = key

            # language_model.model.* -> model.*
            if key.startswith('language_model.model.'):
                new_key = key.replace('language_model.model.', 'model.')
            # language_model.lm_head.* -> lm_head.*
            elif key.startswith('language_model.lm_head.'):
                new_key = key.replace('language_model.lm_head.', 'lm_head.')
            # multi_modal_projector.linear_0.* -> model.mm_projector.0.*
            elif key.startswith('multi_modal_projector.linear_0.'):
                new_key = key.replace('multi_modal_projector.linear_0.', 'model.mm_projector.0.')
            # multi_modal_projector.linear_2.* -> model.mm_projector.2.*
            elif key.startswith('multi_modal_projector.linear_2.'):
                new_key = key.replace('multi_modal_projector.linear_2.', 'model.mm_projector.2.')

            weights[new_key] = tensor

    return weights


def run_inference(args):
    disable_torch_init()

    mlx_model_path = os.path.expanduser(args.mlx_model_path)
    vision_tower_path = os.path.expanduser(args.vision_tower_path)

    print(f"Loading MLX model from {mlx_model_path}...")
    print(f"Using vision tower from {vision_tower_path}...")

    # First, load the original model to get the architecture and vision tower
    model_name = get_model_name_from_path(vision_tower_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        vision_tower_path,
        None,
        model_name,
        device="cuda",
        torch_dtype=torch.float16
    )

    # Convert and load MLX weights (LLM + projector only)
    print("Converting MLX weights to PyTorch format...")
    mlx_weights_file = os.path.join(mlx_model_path, "model.safetensors")
    mlx_weights = convert_mlx_weights_to_pytorch(mlx_weights_file)

    # Get current model state dict
    current_state = model.state_dict()

    # Update with MLX weights (preserving vision tower weights)
    updated_count = 0
    for key, value in mlx_weights.items():
        if key in current_state:
            if current_state[key].shape == value.shape:
                current_state[key] = value.to(current_state[key].dtype)
                updated_count += 1
            else:
                print(f"Shape mismatch for {key}: model={current_state[key].shape}, mlx={value.shape}")
        else:
            print(f"Key not found in model: {key}")

    print(f"Updated {updated_count} weights from MLX model")

    # Load updated weights
    model.load_state_dict(current_state)
    model = model.half().cuda()

    print(f"Model dtype: {model.dtype}")
    print(f"Model device: {model.device}")

    # Load image
    image_path = os.path.expanduser(args.image_file)
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert('RGB')

    # Process image
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # Prepare conversation
    conv = conv_templates[args.conv_mode].copy()
    prompt = DEFAULT_IMAGE_TOKEN + '\n' + args.prompt
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    # Tokenize
    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(model.device)

    print(f"\nPrompt: {args.prompt}")
    print("Generating response...")

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            max_new_tokens=args.max_new_tokens,
            use_cache=True
        )

    # Decode
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(f"\nResponse: {output}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run official MLX FastVLM model on CUDA")
    parser.add_argument("--mlx-model-path", type=str, required=True,
                        help="Path to MLX-exported model (e.g., exported/llava-fastvithd_0.5b_stage3_llm.fp16)")
    parser.add_argument("--vision-tower-path", type=str, required=True,
                        help="Path to original checkpoint with vision tower (e.g., checkpoints/llava-fastvithd_0.5b_stage3)")
    parser.add_argument("--image-file", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--prompt", type=str, default="Describe this image.",
                        help="Text prompt")
    parser.add_argument("--conv-mode", type=str, default="qwen_2",
                        help="Conversation mode")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for sampling (0 = greedy)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum new tokens to generate")

    args = parser.parse_args()
    run_inference(args)
