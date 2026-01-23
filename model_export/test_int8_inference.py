#
# Test inference with INT8 quantized model on GPU
# Uses bitsandbytes for GPU-accelerated INT8 inference
#
import os
import argparse

import torch

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from PIL import Image

try:
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    print("Warning: bitsandbytes not installed. Install with: pip install bitsandbytes")


def test_inference(args):
    if not HAS_BITSANDBYTES:
        raise RuntimeError("bitsandbytes is required for GPU INT8 inference. Install with: pip install bitsandbytes")

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    print(f"Loading INT8 quantized model from {model_path}...")

    # Configure INT8 quantization using bitsandbytes
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    # Load model with INT8 quantization on GPU
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device="cuda",
        quantization_config=quantization_config,
    )

    print(f"Model loaded with INT8 quantization on CUDA")

    # Load image
    image_path = os.path.expanduser(args.image_file)
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert('RGB')

    # Process image - use float16 for image tensors on GPU
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to("cuda", dtype=torch.float16)

    # Prepare conversation
    conv = conv_templates[args.conv_mode].copy()
    prompt = DEFAULT_IMAGE_TOKEN + '\n' + args.prompt
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    # Tokenize
    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to("cuda")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)

    args = parser.parse_args()
    test_inference(args)
