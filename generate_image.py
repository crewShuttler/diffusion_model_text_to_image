#!/usr/bin/env python3
"""Generate an image from text with a diffusion model."""

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt used to generate the image.",
    )
    parser.add_argument(
        "--model-id",
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model id for the diffusion model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated.png"),
        help="Path to save the generated image.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    image = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved generated image to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
