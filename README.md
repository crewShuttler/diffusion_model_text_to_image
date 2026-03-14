# diffusion_model_text_to_image

Generate images from text prompts with a Stable Diffusion model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Create a continuous sequence from a 3D point cloud

Use `point_cloud_sequence.py` to transform an unordered point cloud into an ordered, continuous trajectory.

```bash
python point_cloud_sequence.py data/object_points.xyz \
  --output outputs/object_path.xyz \
  --start-index 0 \
  --spacing 0.01
```

### Point-cloud input format

- One 3D point per line
- Either space-separated (`x y z`) or comma-separated (`x,y,z`)
- Empty lines and lines starting with `#` are ignored

Example:

```text
# x y z
0.0 0.0 0.0
0.5,0.0,0.1
1.0 0.1 0.2
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
In this repository diffusion model architectures are used to generated images data. The diffusion model takes as input the text data and processes text input embedding to generated images or video data.

Diffusion models network architectures used- in progress

Datasets used explanation:

Text Data

Images output data

Training methodology of diffusion model:

Validation methodology of diffusion model:

Average training / inference time per image:

Failure scenarios:

Error functions:

Metrics for validation:

Precision/recall

TP, FP, FN, TN in validation data

Confusion matrix:

Model weights:
