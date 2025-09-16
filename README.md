# Scale Detection Pipeline

A comprehensive pipeline for detecting and analyzing scale bars in microscopy images, following the Uni-AIMS paper architecture. This implementation provides joint YOLOv8 detection of scale bars and text regions, fine-grained endpoint localization, OCR processing, and pixel-to-physical unit conversion.

## Features

- **Joint Detection**: YOLOv8m model for detecting both scale bars and text regions
- **Endpoint Localization**: Fine-grained refinement of scale bar endpoints for accurate pixel length measurement
- **OCR Processing**: PaddleOCR integration with scientific symbol support and unit normalization
- **Scale Matching**: Spatial matching between detected text and scale bars
- **Unit Conversion**: Convert pixel measurements to physical units (mm, μm, nm)
- **Comprehensive Evaluation**: Detection accuracy, OCR performance, and scale conversion metrics

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ScaleDetection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For GPU acceleration (optional):
```bash
pip install paddlepaddle-gpu
```

## Quick Start

### 1. Dataset Preparation

Place your images in `data/figures/` and JSON annotations in `data/jsons/`. The JSON format should include:

```json
{
  "width": 671,
  "height": 476,
  "bars": [
    {
      "id": 0,
      "points": [[387.49, 462.90], [651.19, 469.88]]
    }
  ],
  "labels": [
    {
      "id": 1,
      "points": [[69.95, 212.17], [93.86, 221.53]],
      "text": "1um"
    }
  ]
}
```

### 2. Convert Dataset to YOLO Format

```bash
python src/convert_jsons_to_yolo.py --json_dir data/jsons --output_dir outputs/yolo_dataset
```

### 3. Train YOLOv8 Model

```bash
python src/train_yolov8.py --data_yaml outputs/yolo_dataset/data.yaml --epochs 100 --batch 8
```

### 4. Test Endpoint Localization

```bash
python src/postprocess_scalebar.py --image data/figures/1.jpg --bbox "100,200,150,50" --visualize
```

### 5. Run OCR and Matching

```bash
python src/ocr_and_match.py --image data/figures/1.jpg --bars_json outputs/bars.json --output outputs/results.json
```

### 6. Convert Pixels to Physical Units

```bash
python src/pixels_to_mm.py --um_per_pixel 0.1 --pixel_length 100 --physical_length 10 --unit um
```

### 7. Evaluate Pipeline

```bash
python src/evaluate_pipeline.py --results outputs/results.json --ground_truth data/ground_truth.json
```

## Jupyter Notebook

For interactive usage, run the complete pipeline notebook:

```bash
jupyter notebook pipeline.ipynb
```

## Architecture

The pipeline consists of several key components:

### 1. Dataset Conversion (`src/convert_jsons_to_yolo.py`)
- Converts JSON annotations to YOLO format
- Handles polygon-to-bounding-box conversion
- Creates train/validation splits
- Validates conversion accuracy

### 2. Model Training (`src/train_yolov8.py`)
- YOLOv8m pretrained model
- Custom hyperparameters for microscopy images
- Data augmentation strategies
- Model export to ONNX format

### 3. Endpoint Localization (`src/postprocess_scalebar.py`)
- Channel selection for strongest edges
- Adaptive thresholding (Sauvola/Otsu)
- Morphological operations
- Peak detection and subpixel refinement

### 4. OCR and Matching (`src/ocr_and_match.py`)
- PaddleOCR integration
- Text parsing with regex patterns
- Unit normalization (μm/um/µm → um)
- Spatial matching between text and bars

### 5. Pixel Conversion (`src/pixels_to_mm.py`)
- Convert coordinates, distances, and areas
- Support for mm, μm, and nm units
- Batch processing capabilities
- Validation and error handling

### 6. Evaluation (`src/evaluate_pipeline.py`)
- Detection mAP calculation
- OCR accuracy metrics
- Scale conversion accuracy
- Comprehensive reporting

## Configuration

### Training Parameters
- **Model**: YOLOv8m (medium)
- **Input Size**: 1280x1280 pixels
- **Batch Size**: 8 (adjust based on GPU memory)
- **Epochs**: 50-100
- **Learning Rate**: 0.01 with cosine scheduler
- **Weight Decay**: 0.0005

### OCR Settings
- **Backend**: PaddleOCR (default), EasyOCR, Tesseract
- **Confidence Threshold**: 0.15
- **Language**: English with scientific symbols
- **Unit Normalization**: Automatic conversion to standard units

### Endpoint Localization
- **Thresholding**: Sauvola (default) or Otsu
- **Window Size**: 15 pixels
- **Peak Prominence**: 0.1
- **Subpixel Refinement**: Enabled

## Output Files

The pipeline generates several output files:

- `outputs/yolo_dataset/`: YOLO format dataset
- `outputs/training/`: Trained model checkpoints
- `models/scale_detection_model.onnx`: Exported ONNX model
- `outputs/ocr_results_*.json`: OCR and matching results
- `outputs/evaluation_report.txt`: Comprehensive evaluation report
- `outputs/*.png`: Visualization plots

## Performance

Expected performance metrics (on typical microscopy datasets):

- **Detection mAP@0.5**: >0.8
- **OCR Character Accuracy**: >0.9
- **Scale Conversion Success Rate**: >0.9
- **Endpoint Localization MAE**: <2 pixels

## Troubleshooting

### Common Issues

1. **PaddleOCR Installation**:
   ```bash
   pip install paddlepaddle paddleocr
   ```

2. **CUDA/GPU Issues**:
   - Check CUDA installation: `nvidia-smi`
   - Use CPU if GPU not available: set `device='cpu'`

3. **Memory Issues**:
   - Reduce batch size: `--batch 4`
   - Reduce image size: `--imgsz 640`

4. **Poor Detection Results**:
   - Increase training epochs
   - Adjust confidence thresholds
   - Check data quality and annotations

### Performance Tips

1. Use GPU acceleration when available
2. Batch process multiple images
3. Use appropriate image sizes (1280 for high-res, 640 for speed)
4. Cache OCR models to avoid reloading
5. Use ONNX models for faster inference

## Citation

If you use this pipeline in your research, please cite the Uni-AIMS paper:

```bibtex
@article{uni-aims-2023,
  title={Uni-AIMS: Unified AI for Microscopy Scale Detection},
  author={[Authors]},
  journal={[Journal]},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and support, please open an issue on GitHub.