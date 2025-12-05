"""
ScaleDetection package

This package contains modules and scripts for detecting scale bars in images,
running OCR on scale text, and measuring the bar length.

The `src` folder contains the following modules:
- `scaledetection.py`: main pipeline for detection + OCR matching
- `postprocess_scalebar.py`: precise endpoint localization and refinement
- `atypical_scalebars.py`: handlers for graduated/ruler-like scale bars
- `ocr.py`: OCR and text parsing utilities
- `classifier.py`: template-based scale bar classifier using ORB
- `precompute_ORB_descriptors.py`: utility to build ORB template descriptors
- `utils.py`: helper functions (e.g. model download)
"""
