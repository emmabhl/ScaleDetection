#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=0:30:00
#SBATCH --job-name=VLM_scale_detection
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
#SBATCH --qos=normal
#SBATCH --open-mode=append

# Load necessary modules
module load python/3.12 cuda/12.6 arrow/21.0.0 opencv/4.12.0 

# Activate virtual environment
source ~/.bashrc
source ~/scale/bin/activate

# Run the VLM scale detection script
python src/VLM.py --filepath data/annot --output_folder outputs_vlm --model_id Qwen/Qwen3-VL-8B-Instruct --max_side 2048
