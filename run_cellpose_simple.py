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

# Default folders (change these to your preferred defaults)
DEFAULT_RAW_FOLDER = '/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Immunochemistry/CellPose/test_input_images'
DEFAULT_CELL_POSE_SAVE_FOLDER = '/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Immunochemistry/CellPose/test_output_images'


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
    parser.add_argument('--raw', '--raw_image_folder', dest='raw', default=DEFAULT_RAW_FOLDER,
                        help=f'Folder containing raw images (png or tiff). Default: {DEFAULT_RAW_FOLDER}')
    parser.add_argument('--out', '--cell_pose_save_folder', dest='out', default=DEFAULT_CELL_POSE_SAVE_FOLDER,
                        help=f'Destination folder where processed images and results will be saved. Default: {DEFAULT_CELL_POSE_SAVE_FOLDER}')
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
        # Create the folder so user can drop images there; don't error out
        try:
            raw_folder.mkdir(parents=True, exist_ok=True)
            print(f"Raw folder did not exist, created: {raw_folder}\nPlease add images to this folder and re-run the script.")
        except Exception as e:
            print(f"Could not create raw folder '{raw_folder}': {e}")
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

    # Resolve GPU preference - Default to auto-detect (will use GPU if available)
    gpu_pref = None
    if args.gpu:
        gpu_pref = True
    elif args.cpu:
        gpu_pref = False

    print("Simple CellPose runner")
    print(f"  Raw folder: {raw_folder}")
    print(f"  Output: {out_folder}")
    print(f"  Image type: .{image_type}")
    if args.contains:
        print(f"  Filter: contains '{args.contains}'")
    print(f"  3D: {'ON' if args.do_3D else 'OFF'}")
    print(f"  GPU: {'Force ON' if gpu_pref is True else 'Force OFF' if gpu_pref is False else 'Auto'}")
    print()

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
