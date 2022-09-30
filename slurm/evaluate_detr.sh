#!/usr/bin/zsh
#SBATCH --job-name=evaluate_detr
#SBATCH --mem=128gb
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

export PYTHONPATH=$PYTHONPATH:$DIRECTORY
export DETECTRON2_DATASETS="/mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/data"
ln -s /mnt/datasets/COCO $DETECTRON2_DATASETS/coco

# Experiment paths
DETR_OUTPUT_ANIMALS="/mnt/work/dhuseljic/detectron2/outputs/8207/animals"
DETR_OUTPUT_TRAFFIC="/mnt/work/dhuseljic/detectron2/outputs/8207/traffic"

# Dataset paths
COCO_PATH="/mnt/datasets/COCO/"
COCO_ANN_PATH="/mnt/datasets/COCO/annotations/"
echo '=== Starting script: ANIMALS (COCO) ==='
srun python /mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/evaluate.py \
    --dataset coco \
    --ds_path $COCO_PATH \
    --ann_path $COCO_ANN_PATH \
    --experiment_path $DETR_OUTPUT_ANIMALS \
    --result_path "/mnt/work/dhuseljic/uod_results/detr_animals" \
    --perspective modeling \
    SEED 42 
echo '=== Starting script: TRAFFIC (COCO) ==='
srun python /mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/evaluate.py \
    --dataset coco \
    --ds_path $COCO_PATH \
    --ann_path $COCO_ANN_PATH \
    --experiment_path $DETR_OUTPUT_TRAFFIC \
    --result_path "/mnt/work/dhuseljic/uod_results/detr_traffic" \
    --perspective modeling \
    SEED 42 
echo '=== Starting script: ALL (COCO) ==='
srun python /mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/evaluate.py \
    --dataset coco \
    --ds_path $COCO_PATH \
    --ann_path $COCO_ANN_PATH \
    --eval_from_config "/mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/configs/detr_vanilla.yaml" \
    --result_path "/mnt/work/dhuseljic/uod_results/detr_all" \
    --perspective modeling \
    MODEL.WEIGHTS "/mnt/work/dhuseljic/detectron2/models/detr_coco.pth" \
    SEED 42 

# Dataset paths
OI_PATH=/mnt/datasets/open-images/train
OI_ANN_PATH_ANIMALS=/mnt/work/dhuseljic/datasets/open-images/open-images_animals-subset_coco-format.json
OI_ANN_PATH_TRAFFIC=/mnt/work/dhuseljic/datasets/open-images/open-images_traffic-subset_coco-format.json
OI_ANN_PATH_OOD=/mnt/work/dhuseljic/datasets/open-images/open-images_ood-subset_coco-format.json
echo '=== Starting script: ANIMALS (Open-Images) ==='
srun python /mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/evaluate.py \
    --dataset open-images \
    --ds_path $OI_PATH \
    --ann_path $OI_ANN_PATH_ANIMALS \
    --experiment_path $DETR_OUTPUT_ANIMALS \
    --result_path "/mnt/work/dhuseljic/uod_results/detr_animals_shifted" \
    --perspective modeling \
    SEED 42 
echo '=== Starting script: TRAFFIC (Open-Images) ==='
srun python /mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/evaluate.py \
    --dataset open-images \
    --ds_path $OI_PATH \
    --ann_path $OI_ANN_PATH_TRAFFIC \
    --experiment_path $DETR_OUTPUT_TRAFFIC \
    --result_path "/mnt/work/dhuseljic/uod_results/detr_traffic_shifted" \
    --perspective modeling \
    SEED 42 
echo '=== Starting script: OOD (Open-Images) ==='
srun python /mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/evaluate.py \
    --dataset open-images \
    --ds_path $OI_PATH \
    --ann_path $OI_ANN_PATH_OOD \
    --experiment_path $DETR_OUTPUT_TRAFFIC \
    --result_path "/mnt/work/dhuseljic/uod_results/detr_ood" \
    --perspective modeling \
    SEED 42 

