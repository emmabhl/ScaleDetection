#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:01:00
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

# Step 1: Get scale bar length in pixels
python src/postprocess_scalebar.py --image data/images/val/9.jpg --model models/train/weights/best.pt --output_dir outputs --visualize

# Step 2: Scale text recognition from label bounding box
python src/ocr_and_match.py