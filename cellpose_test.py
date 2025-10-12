#!/usr/bin/env python3
"""Minimal GPU test for Cellpose"""
import torch
import numpy as np
from cellpose import models

# Check GPU
cuda = torch.cuda.is_available()
gpu_ok = models.use_gpu()

print(f"CUDA available: {cuda}")
if cuda:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Cellpose GPU: {gpu_ok}")

# Test inference
print("\nTesting inference...")
model = models.Cellpose(model_type='cyto2', gpu=gpu_ok)
img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
masks, _, _, _ = model.eval(img, diameter=30, channels=[0,0])
print(f"✅ Success! Found {len(np.unique(masks))-1} objects")
print(f"Device: {'GPU' if gpu_ok else 'CPU'}")
