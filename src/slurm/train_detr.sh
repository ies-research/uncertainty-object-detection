#!/usr/bin/zsh
#SBATCH --job-name=train_detr
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --output=/mnt/work/dhuseljic/logs/UOD/%x_%A.log
#SBATCH --exclude=vana
date;hostname;pwd
source /mnt/home/dhuseljic/.zshrc
conda activate uod
echo "Using GPUs ${CUDA_VISIBLE_DEVICES}."

DIRECTORY=$(pwd)
cd $DIRECTORY

OUTPUT_DIR="/mnt/work/dhuseljic/detectron2/outputs/${SLURM_JOB_ID}/"
echo "Saving outputs to ${OUTPUT_DIR}"

export PYTHONPATH=$PYTHONPATH:$DIRECTORY
# For detectron2 to find coco
export DETECTRON2_DATASETS="/mnt/datasets/"
ln -s /mnt/datasets/COCO data/coco
# For detectron2 to find coco

# DETR
MODEL_CONFIG=/mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/configs/detr_vanilla.yaml
MODEL_WEIGHTS=/mnt/work/dhuseljic/detectron2/models/detr_coco.pth

# animals=53_060 for 10 epochs
echo '=== Train DETR on animals==='
OUTPUT_DIR="/mnt/work/dhuseljic/detectron2/outputs/${SLURM_JOB_ID}/animals"
python /mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/train.py \
    --coco_path /mnt/datasets/COCO/ \
    --ann_path /mnt/datasets/COCO/annotations/ \
    --subset animals \
    --config-file $MODEL_CONFIG \
    SOLVER.MAX_ITER 53_060  \
    MODEL.WEIGHTS $MODEL_WEIGHTS \
    OUTPUT_DIR $OUTPUT_DIR \
    SEED 42

# traffic=370_000 for 10 epochs
echo '=== Train DETR on traffic ==='
OUTPUT_DIR="/mnt/work/dhuseljic/detectron2/outputs/${SLURM_JOB_ID}/traffic"
python /mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/train.py \
    --coco_path /mnt/datasets/COCO/ \
    --ann_path /mnt/datasets/COCO/annotations/ \
    --subset traffic \
    --config-file $MODEL_CONFIG \
    SOLVER.MAX_ITER 370_000 \
    MODEL.WEIGHTS $MODEL_WEIGHTS \
    OUTPUT_DIR $OUTPUT_DIR \
    SEED 42 
