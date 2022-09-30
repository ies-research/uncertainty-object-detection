#!/usr/bin/zsh
#SBATCH --job-name=create_datasets
#SBATCH --mem=128gb
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/UOD/%x_%A.log
#SBATCH --exclude=vana
date;hostname;pwd
source /mnt/home/dhuseljic/.zshrc
conda activate uod
echo "Using GPUs ${CUDA_VISIBLE_DEVICES}."

DIRECTORY=$(pwd)
cd $DIRECTORY

export PYTHONPATH=$PYTHONPATH:$DIRECTORY
export DETECTRON2_DATASETS="/mnt/datasets/"

OUTPUT_DIR=/mnt/work/dhuseljic/datasets/open-images/

echo '=== Creating Animals Shifted ==='
srun python ./create_datasets.py \
    --ds_path /mnt/datasets/open-images/train \
    --ann_path /mnt/datasets/open-images/annotations \
    --output_dir $OUTPUT_DIR \
    --subset animals

echo '=== Creating Traffic Shifted ==='
srun python ./create_datasets.py \
    --ds_path /mnt/datasets/open-images/train \
    --ann_path /mnt/datasets/open-images/annotations \
    --output_dir $OUTPUT_DIR \
    --subset traffic

echo '=== Creating OOD ==='
srun python ./create_datasets.py \
    --ds_path /mnt/datasets/open-images/train \
    --ann_path /mnt/datasets/open-images/annotations \
    --output_dir $OUTPUT_DIR \
    --subset ood
