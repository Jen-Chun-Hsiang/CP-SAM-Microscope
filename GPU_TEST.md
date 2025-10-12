## GPU Testing - Quick Guide

### Files Overview
1. **cellpose_test.py** - Quick GPU verification (20 lines)
2. **test_gpu_image.py** - Process test image with GPU (30 lines)  
3. **run_cellpose_simple.py** - Main processing script (auto-detects GPU)
4. **ImageProcessing.py** - Core processor (auto-detects GPU)

### Quick Start on Server

**1. Test GPU availability:**
```bash
python cellpose_test.py
```

**2. Test with actual image:**
```bash
python test_gpu_image.py your_image.png
```

**3. Process your images:**
```bash
python run_cellpose_simple.py --raw /path/to/images --out /path/to/output
```

### Expected Output

If GPU is working, you'll see:
```
CUDA available: True
GPU: NVIDIA A100-SXM4-40GB
Cellpose GPU: True
Device: GPU
```

If using CPU (slower):
```
CUDA available: False
Device: CPU
```

### Flags
- `--gpu` - Force GPU (fail if unavailable)
- `--cpu` - Force CPU only
- Default: Auto-detect (uses GPU if available)

### That's it!
All scripts automatically use GPU when available. No configuration needed.
