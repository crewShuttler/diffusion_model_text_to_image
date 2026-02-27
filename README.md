# diffusion_model_text_to_image

Generate images from text prompts with a Stable Diffusion model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python generate_image.py "a cinematic photo of a futuristic city at sunset" \
  --output outputs/city.png \
  --steps 35 \
  --guidance-scale 8.0 \
  --seed 123
```

### Optional arguments

- `--model-id`: Hugging Face model ID (default: `runwayml/stable-diffusion-v1-5`)
- `--output`: Output image path (default: `generated.png`)
- `--steps`: Number of denoising steps (default: `30`)
- `--guidance-scale`: Guidance scale (default: `7.5`)
- `--seed`: Random seed for reproducibility (default: `42`)

> Note: The first run downloads model weights from Hugging Face.
