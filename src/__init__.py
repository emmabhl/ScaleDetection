"""
ScaleDetection package

This package contains modules and scripts for detecting scale bars in images,
running OCR on scale text, converting annotations to YOLO format, training
and running YOLO models, and various helper utilities.

The `src` folder contains the following modules:
- `scaledetection.py`: main pipeline for detection + OCR matching
- `postprocess_scalebar.py`: precise endpoint localization and refinement
- `atypical_scalebars.py`: handlers for graduated/ruler-like scale bars
- `ocr.py`: OCR and text parsing utilities
- `classifier.py`: template-based scale bar classifier using ORB
- `precompute_ORB_descriptors.py`: utility to build ORB template descriptors
- `convert_jsons_to_yolo.py`: convert JSON annotations to YOLO format
- `get_data.py`: download and prepare the dataset
- `train_yolo.py`: train YOLOv8 models
- `utils.py`: helper functions (e.g. model download)
"""
