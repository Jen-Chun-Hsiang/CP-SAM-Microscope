#!/bin/bash

# Upgrade pip first
python3 -m pip install --upgrade pip

# Install numpy and scipy with version constraints that are compatible with cellpose dependencies
# numpy>=2.0 breaks matplotlib, pandas, and contourpy - use numpy<2.0 instead
python3 -m pip install --upgrade "numpy>=1.22.4,<2.0" "scipy>=1.9.0,<1.15.0"

# Install all project dependencies from requirements.txt
python3 -m pip install -r "$HOME/CellPose/CP-SAM-Microscope/requirements.txt"

# Run the cellpose processor
#python3 $HOME/CellPose/CP-SAM-Microscope/cellpose_test.py
python3 $HOME/CellPose/CP-SAM-Microscope/run_cellpose_simple.py