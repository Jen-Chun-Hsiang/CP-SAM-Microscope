# Cellpose Image Enhancement

This project enhances microscopy images (including immunochemistry and fluorescent microscope images) using open-source tools. It specifically uses the open-source `cellpose-sam` package to improve segmentation and downstream image processing in workflows derived from immunochemistry and fluorescence microscopy.

## What this repo contains

- `ImageProcessing.py` — main script for image processing (segmentation, post-processing, export).

## Key point

This project uses the open-source `cellpose-sam` package to enhance the imaging process for immunochemistry and fluorescent microscopy data. `cellpose-sam` combines generalist segmentation (Cellpose) with the Segment Anything Model (SAM) to deliver faster and more accurate object delineation on diverse fluorescent and immunostained samples.

## Requirements

- Python 3.8+
- GPU recommended for large datasets (optional)

Assumed Python packages (install below). If the exact package name differs in your environment, adapt as needed.

## Installation

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies. This repository expects the open-source `cellpose-sam` package; install it along with common imaging libraries:

```bash
pip install cellpose-sam numpy scipy scikit-image opencv-python matplotlib
```

If `cellpose-sam` is not available under that exact name in your environment, please refer to the package's upstream repository or replace it with the appropriate Cellpose + SAM integration package you use.

## Usage

A simple way to run the processing script:

```bash
python ImageProcessing.py \
	--input-image-folder /path/to/images \
	--out-images /path/to/processed_images \
	--out-results /path/to/results \
	--type png \
	--contains None
```

Flags:
- `--input-image-folder` (alias: `--input`, `-i`): Input image folder (default: `input_images`)
- `--out-images, -oi`: Folder to save visualization and mask images (default: `output_images`)
- `--out-results, -or`: Folder to save NPZ/JSON results (default: `output_results`)
- `--type, -t`: Image file extension to process (default: `png`; examples: `tif`, `jpg`)
- `--contains, -c`: Optional filename substring filter (case-insensitive). Use `None` or omit to process all images of the selected type.

Notes:
- For best results on fluorescent or immunochemistry images, ensure image bit depth and channel assignments are correct before processing.

## Input / Output

- Input: microscopy image files (TIFF, PNG, or other formats supported by your script).
- Output: segmentation masks, labeled overlays, quantitative measurements (as implemented in `ImageProcessing.py`).

## Notes and assumptions

- This README assumes `ImageProcessing.py` integrates with `cellpose-sam`. If the script requires different import names or API calls, update the installation instructions accordingly.
- If you plan to process large datasets or want GPU acceleration, configure appropriate CUDA/cuDNN drivers and install GPU-enabled versions of dependencies.

## License & attribution

This project uses the open-source `cellpose-sam` package. Please consult the upstream `cellpose` and `SAM` repositories for license information and appropriate citations when publishing results.

## Contact / Next steps

- Check `ImageProcessing.py` for command-line options and any additional dependency requirements.
- Optionally add a `requirements.txt` or `environment.yml` to pin exact dependency versions.


