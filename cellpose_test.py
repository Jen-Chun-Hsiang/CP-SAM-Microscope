# cellpose_test.py
import torch
import cellpose
from cellpose import models

print("✅ Successfully imported PyTorch and Cellpose")

# GPU status
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Device name:", torch.cuda.get_device_name(0))

# Cellpose version
print("Cellpose version:", getattr(cellpose, "__version__", "unknown"))

# Cellpose GPU diagnostics
print("\nCellpose GPU diagnostics:")
cellpose_gpu = None
try:
    cellpose_gpu = bool(models.use_gpu())
    print("models.use_gpu():", cellpose_gpu)
except Exception as exc:
    print("models.use_gpu() check failed:", exc)

if cellpose_gpu:
    try:
        test_model = models.Cellpose(model_type="cyto", gpu=True)
        device = getattr(getattr(test_model, "net", None), "device", None)
        print("Cellpose model device:", device or "unknown")
    except Exception as exc:
        print("Cellpose GPU model init failed:", exc)
    else:
        del test_model
else:
    print("Cellpose reports GPU unavailable; using CPU fallback.")

# Simple sanity run: show help for CLI
print("\nRunning Cellpose help command...")
import subprocess
subprocess.run(["python3", "-m", "cellpose", "--help"])

print("\n✅ Test finished successfully.")
