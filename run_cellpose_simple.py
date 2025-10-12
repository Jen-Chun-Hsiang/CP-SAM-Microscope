#!/usr/bin/env python3
"""
Simple CellPose runner

Loads PNG or TIFF images from a raw image folder and saves processed
visualizations and NPZ/JSON results into a destination folder.

Usage example:
  python run_cellpose_simple.py --raw ./raw_images --out ./cell_pose_save_folder

This script wraps the existing CellPoseProcessor defined in
`ImageProcessing.py` in this repository.
"""
import argparse
from pathlib import Path
import sys
import glob

# Make sure the script directory is on sys.path so local imports work
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

try:
    from ImageProcessing import CellPoseProcessor
except Exception as e:
    print(f"Error importing CellPoseProcessor from ImageProcessing.py: {e}")
    sys.exit(1)


def detect_image_type(raw_folder: Path) -> str:
    """Choose 'tif' if any .tif/.tiff files exist, otherwise default to 'png'."""
    raw = Path(raw_folder)
    if not raw.exists() or not raw.is_dir():
        return 'png'
    tif_files = list(raw.glob('*.tif')) + list(raw.glob('*.tiff'))
    if len(tif_files) > 0:
        return 'tif'
    # fallback to png
    return 'png'


def main():
    parser = argparse.ArgumentParser(description="Simple CellPose runner: process a folder of images")
    parser.add_argument('--raw', '--raw_image_folder', dest='raw', required=True,
                        help='Folder containing raw images (png or tiff)')
    parser.add_argument('--out', '--cell_pose_save_folder', dest='out', required=True,
                        help='Destination folder where processed images and results will be saved')
    parser.add_argument('--type', '-t', dest='itype', default=None,
                        help="Image type/extension to process (png or tif). If omitted the script will auto-detect")
    parser.add_argument('--contains', '-c', default=None, help='Optional substring to filter filenames')
    parser.add_argument('--gpu', action='store_true', help='Force GPU (if available)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--three-d', dest='do_3D', action='store_true', help='Enable 3D processing for TIFF stacks')

    args = parser.parse_args()

    raw_folder = Path(args.raw).expanduser().resolve()
    out_folder = Path(args.out).expanduser().resolve()

    if not raw_folder.exists():
        print(f"Raw folder does not exist: {raw_folder}")
        sys.exit(1)

    # Create destination subfolders
    images_out = out_folder / 'images'
    results_out = out_folder / 'results'
    images_out.mkdir(parents=True, exist_ok=True)
    results_out.mkdir(parents=True, exist_ok=True)

    # Determine image type if not provided
    if args.itype:
        image_type = args.itype.lstrip('.')
    else:
        image_type = detect_image_type(raw_folder)

    # Resolve GPU preference
    gpu_pref = None
    if args.gpu:
        gpu_pref = True
    if args.cpu:
        gpu_pref = False

    print("Simple CellPose runner")
    print(f"  Raw folder: {raw_folder}")
    print(f"  Output images: {images_out}")
    print(f"  Output results: {results_out}")
    print(f"  Image type: .{image_type}")
    print(f"  Filename contains filter: {args.contains}")
    print(f"  3D processing: {'ON' if args.do_3D else 'OFF'}")
    if gpu_pref is True:
        print("  Device preference: GPU (forced)")
    elif gpu_pref is False:
        print("  Device preference: CPU (forced)")
    else:
        print("  Device preference: auto-detect")

    # Minimal, sensible defaults
    processor = CellPoseProcessor(
        input_folder=str(raw_folder),
        image_save_folder=str(images_out),
        result_save_folder=str(results_out),
        model_type='cyto2',
        channels=[0, 0],
        diameter=30,
        flow_threshold=0.4,
        cellprob_threshold=-2.0,
        min_size=15,
        normalize=True,
        filename_contains=args.contains,
        image_type=image_type,
        do_3D=args.do_3D,
        gpu=gpu_pref
    )

    # Run
    processor.process_all_images()


if __name__ == '__main__':
    main()
