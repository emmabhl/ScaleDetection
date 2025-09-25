#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0:10:00
#SBATCH --job-name=train_yolo
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
#SBATCH --qos=normal
#SBATCH --open-mode=append

# Load necessary modules
module load python/3.12 cuda/12.6 arrow/21.0.0 opencv/4.12.0 

# Activate virtual environment
source ~/.bashrc
source ~/scale/bin/activate

# Step 1: Get the data
python src/get_data.py --path original1/scalebar-dataset --data_dir data

# Step 2: Convert JSONs to YOLO format
python src/convert_jsons_to_yolo.py --data_dir data --validate

# Step 3: Train the YOLO model
python src/train_yolo.py --data_yaml data/data.yaml --model_name yolov8m.pt --epochs 500 --batch 16 --device 0