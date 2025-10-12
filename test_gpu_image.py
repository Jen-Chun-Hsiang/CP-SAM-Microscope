#!/usr/bin/env python3
"""
Minimal Cellpose GPU test with actual image processing
Usage: python test_gpu_image.py [path_to_image.png]
"""
import sys
import numpy as np
from cellpose import models, io

# Get image path or create test image
if len(sys.argv) > 1:
    img = io.imread(sys.argv[1])
    print(f"Loaded: {sys.argv[1]}")
else:
    img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    print("Using random test image (512x512)")

# Initialize with GPU
gpu = models.use_gpu()
print(f"Using: {'GPU' if gpu else 'CPU'}")

model = models.Cellpose(model_type='cyto2', gpu=gpu)

# Run segmentation
import time
start = time.time()
masks, flows, styles, diams = model.eval(img, diameter=30, channels=[0,0])
elapsed = time.time() - start

# Results
n_cells = len(np.unique(masks)) - 1
print(f"Found {n_cells} cells in {elapsed:.2f}s")
print(f"✅ {'GPU' if gpu else 'CPU'} test complete!")
