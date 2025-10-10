# cellpose_test.py
import torch
import cellpose

print("✅ Successfully imported PyTorch and Cellpose")

# GPU status
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Device name:", torch.cuda.get_device_name(0))

# Cellpose version
print("Cellpose version:", getattr(cellpose, "__version__", "unknown"))

# Simple sanity run: show help for CLI
print("\nRunning Cellpose help command...")
import subprocess
subprocess.run(["python3", "-m", "cellpose", "--help"])

print("\n✅ Test finished successfully.")
