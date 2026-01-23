#
# Inference script using mlx-vlm CUDA backend
# Uses the same parameters as model_export/test_fp16_inference.py
#
import os
import sys
import argparse
import time

# Add mlx-vlm to path
sys.path.insert(0, '/home/n10288/Documents/Code/mlx-vlm')

from mlx_vlm import load, generate


def predict(args):
    model_path = os.path.expanduser(args.model_path)
    image_path = os.path.expanduser(args.image_file)

    print(f"Loading model from {model_path}...")
    start_time = time.perf_counter()
    model, processor = load(model_path)
    load_time = time.perf_counter() - start_time
    print(f"Model loaded in {load_time:.2f}s")

    print(f"\nImage: {image_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_new_tokens}")
    print("\nGenerating response...")

    response = generate(
        model,
        processor,
        prompt=args.prompt,
        image=image_path,
        max_tokens=args.max_new_tokens,
        temp=args.temperature,
        verbose=args.verbose,
    )

    if not args.verbose:
        print(f"\nResponse: {response}")

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FastVLM inference using mlx-vlm CUDA backend"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the model checkpoint (original or quantized)"
    )
    parser.add_argument(
        "--image-file", type=str, required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--prompt", type=str, default="Describe this image.",
        help="Prompt for the model"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0.0 = greedy)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed generation statistics"
    )

    args = parser.parse_args()
    predict(args)
