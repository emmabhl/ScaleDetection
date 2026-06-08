#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=scale_detection_errimgs
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
#SBATCH --qos=normal
#SBATCH --open-mode=append

# Load necessary modules
module load python/3.12 cuda/12.6 arrow/21.0.0 opencv/4.12.0

# Activate virtual environment
source ~/.bashrc
source ~/scale/bin/activate

# Add paddle's lib dir so stub MKL dispatch libs are found at dlopen time
export LD_LIBRARY_PATH=/home/eboehly/scale/lib/python3.12/site-packages/paddle/libs:${LD_LIBRARY_PATH}

# Re-run the scale detection pipeline on the flagged error images, WITH visualizations
# (--plot saves per-image debug images alongside the JSON results in --output_dir).
python ../src/scaledetection.py \
    --model models/yolov8m_train/weights/best.pt \
    --image_dir error_analysis/error_images/ \
    --output_dir error_analysis/results/ \
    --plot
