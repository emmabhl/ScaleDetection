#!/bin/bash
#SBATCH --job-name=train_yolo
#SBATCH --output=train_yolo_%j.log
#SBATCH --error=train_yolo_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Load necessary modules
module load gcc python

# Activate virtual environment if needed
source ~/.bashrc
source activate scale

# Navigate to the project directory
cd src

# Step 1: Get the data
python get_data.py --path original1/scalebar-dataset --data_dir data

# Step 2: Convert JSONs to YOLO format
python convert_jsons_to_yolo.py --data_dir data --validate

# Step 3: Train the YOLO model
python train_yolo.py --data_yaml data/data.yaml --model_name yolov8m.pt --epochs 100 --batch_size 16