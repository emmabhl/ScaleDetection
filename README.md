# **ScaleDetection**

ScaleDetection is a complete computer-vision pipeline for **automatically detecting scale bars and reading their associated text labels** (e.g., “100 μm”, “1 cm”) in microscopy, biology, and satellite images.
It integrates **YOLO-based detection**, **OCR**, **post-processing**, and **atypical scale-bar classification** to return a standardized measurement output.

The pipeline supports:

* Processing **single images** or **full folders**.
* Detection of **normal scale bars** and **atypical scale bars** (graduated bars, rulers, etc.)
* OCR-based reading of physical units (µm, mm, cm…)
* Pixel-to-metric conversion.
* Optional visualization/debugging output.

---

## **✨ Features**

### **1. Scale Bar Detection (YOLO)**

* Trained YOLO models detect:

  * Class 0 → Scale bar
  * Class 1 → Text label
* Handles multiple detections and performs a matching procedure between bar and text.

### **2. Post-Processing**

Core logic implemented in `postprocess_scalebar.py`:

* Localizes scale bar endpoints.
* Estimates pixel length.
* Infers orientation and confidence.

### **3. OCR Label Recognition**

Implemented in `ocr.py`:

* Extracts and parses text labels.
* Converts recognized tokens into numerical scale values and units.

### **4. Atypical Scale Bar Handling**

Implemented in:

* `classifier.py` or `clip_classifier.py` (visual classifier for scale-bar type)
* `atypical_scalebars.py`
  Supports:
* Graduated bars with central unit
* Ruler-like photo scale bars
* Other non-standard bar shapes

### **5. Full End-to-End Pipeline**

Main class: `ScaleDetectionPipeline` (in `scaledetection.py`)

Combines YOLO, OCR, and post-processing to output a unified `Scale` dataclass:

```python
Scale(
    scale_bar_found=True/False,
    measured_scale_length=...,
    declared_scale_length=...,
    units=...,
    pixel_to_mm_ratio=...,
    orientation=...,
    type_=...,
)
```

### **6. Single-Image or Folder Processing**

The command-line interface detects whether the input path is:

* a file → process once
* a folder → processes each image inside recursively

---

## **📁 Repository Structure**

```
ScaleDetection/
│
├── README.md
├── VLM.ipynb                   # Vision-language experimentation
├── draft.ipynb                 # Misc. dev notes / experiments
├── VLM_detection.sh            # Script for running VLM detection
├── train_yolo.sh               # SLURM script for YOLO training
├── technical_note.md           # Technical documentation (internal)
│
├── src/
│   ├── scaledetection.py       # Main pipeline: ScaleDetectionPipeline + Scale dataclass
│   ├── postprocess_scalebar.py # ScalebarProcessor + ScalebarDetection
│   ├── ocr.py                  # OCRProcessor + LabelDetection
│   ├── classifier.py           # ORB-based scale bar type classifier
│   ├── clip_classifier.py      # CLIP-based scale bar classifier
│   ├── atypical_scalebars.py   # Extraction for atypical scale bars
│   ├── convert_jsons_to_yolo.py# Convert annotated datasets to YOLO format
│   ├── train_yolo.py           # YOLO training script
│   └── get_data.py             # Data import utilities
│
└── models/                     # (optional) place YOLO + CLIP/ORB templates
```

---

## **🚀 Installation**

### **1. Clone the repository**

```bash
git clone https://github.com/.../ScaleDetection.git
cd ScaleDetection
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. (Optional) download trained YOLO weight files**

Place them in:

```
models/yolo/
```

---

## **🔧 Usage**

### **Run the pipeline on a single image**

```bash
python src/scaledetection.py --image path/to/image.jpg --output_dir results_folder/
```

### **Run the pipeline on a folder**

```bash
python src/scaledetection.py --image_dir path/to/folder/ --output_dir results_folder/
```

The pipeline automatically:

* iterates through the folder,
* processes all images (`.jpg`, `.png`, `.tif`, `.jpeg`),
* saves results for each file.

---

## **🧠 YOLO Dataset Conversion**

Convert JSON annotations to YOLO format:

```bash
python src/convert_jsons_to_yolo.py \
    --input annotations/ \
    --output yolo_dataset/
```

---

## **🎯 YOLO Training**

Train a YOLO model (Ultralytics):

```bash
python src/train_yolo.py --data config.yaml --epochs 200
```

Or submit the SLURM script:

```bash
sbatch train_yolo.sh
```

---

## **🖼️ Vision-Language Experiments**

In the folder `VLM/` you can find code for experimenting with Qwen3 vision-language models 
for scale-bar detection and classification.

Script interface:

* `VLM/VLM.py`

Used for:

* Prompting vision-language models
* Scale-bar reasoning when YOLO fails
* Experimentation with multimodal embeddings

---

## **📈 Outputs**

For each image, the pipeline generates a JSON containing:

```json
{
  "scale_bar_found": true,
  "type": "normal",
  "measured_scale_length": 235.3,
  "declared_scale_length": 100,
  "units": "µm",
  "pixel_to_mm_ratio": 0.42,
  "orientation": "horizontal",
  "scale_bar_confidence": 0.92,
  "text_label_confidence": 0.87
}
```

If debug mode is enabled, annotated images and intermediate plots are saved automatically.

---

These contain architecture thoughts, TODOs, and experimental ideas.

---

## **📬 Contact**

Created and maintained by **emmabhl**.
If you use or extend this tool, feel free to open issues or PRs.