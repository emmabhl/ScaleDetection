#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=1:00:00
#SBATCH --job-name=train_yolo
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
#SBATCH --qos=normal
#SBATCH --open-mode=append

# Load necessary modules
module load gcc python

# Activate virtual environment if needed
source ~/.bashrc
source activate scale

clearml-init
api {
    # Emma Boehly's workspace
    web_server:https://app.clear.ml/
    api_server:https://api.clear.ml
    files_server:https://files.clear.ml
    credentials {
    "access_key"="A75PPTZQV7QNMJKKUKJM2X17XM8L4M"
    "secret_key"="shECi5qzWFIxqr7aVkqI5imYcNWwdro0bdtaNTQxYdhle9s3J6pqUzYPi0Np4iJ3Xhk"
}

# Navigate to the project directory
cd src

# Step 1: Get the data
python get_data.py --path original1/scalebar-dataset --data_dir data

# Step 2: Convert JSONs to YOLO format
python convert_jsons_to_yolo.py --data_dir data --validate

# Step 3: Train the YOLO model
python train_yolo.py --data_yaml data/data.yaml --model_name yolov8m.pt --epochs 100 --batch 16