#!/usr/bin/zsh
#SBATCH --job-name=evaluate_frcnn_modeling
#SBATCH --mem=512gb
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
FRCNN_OUTPUT_ANIMALS=/mnt/work/dhuseljic/detectron2/outputs/8204/animals
FRCNN_OUTPUT_TRAFFIC=/mnt/work/dhuseljic/detectron2/outputs/8204/traffic
FRCNN_OUTPUT_ALL=/mnt/work/dhuseljic/detectron2/outputs/8215/all
PERSPECTIVE=modeling

### COCO DATASET ###
# Dataset paths
COCO_PATH=/mnt/datasets/COCO/
COCO_ANN_PATH=/mnt/datasets/COCO/annotations/
echo '=== Starting script: ANIMALS (COCO) ==='
# srun python /mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/evaluate.py \
#     --dataset coco \
#     --ds_path $COCO_PATH \
#     --ann_path $COCO_ANN_PATH \
#     --experiment_path $FRCNN_OUTPUT_ANIMALS \
#     --result_path "/mnt/work/dhuseljic/uod_results/frcnn_animals_${PERSPECTIVE}" \
#     --perspective $PERSPECTIVE \
#     SEED 42 
echo '=== Starting script: TRAFFIC (COCO) ==='
# srun python /mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/evaluate.py \
#     --dataset coco \
#     --ds_path $COCO_PATH \
#     --ann_path $COCO_ANN_PATH \
#     --experiment_path $FRCNN_OUTPUT_TRAFFIC \
#     --result_path "/mnt/work/dhuseljic/uod_results/frcnn_traffic_${PERSPECTIVE}" \
#     --perspective $PERSPECTIVE \
#     SEED 42 
echo '=== Starting script: ALL (COCO) ==='
srun python /mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/evaluate.py \
    --dataset coco \
    --ds_path $COCO_PATH \
    --ann_path $COCO_ANN_PATH \
    --experiment_path $FRCNN_OUTPUT_ALL \
    --result_path "/mnt/work/dhuseljic/uod_results/frcnn_all_${PERSPECTIVE}" \
    --perspective $PERSPECTIVE \
    SEED 42 

### OPEN-IMAGES ###
OI_PATH=/mnt/datasets/open-images/train
OI_ANN_PATH_ANIMALS=/mnt/work/dhuseljic/datasets/open-images/open-images_animals-subset_coco-format.json
OI_ANN_PATH_TRAFFIC=/mnt/work/dhuseljic/datasets/open-images/open-images_traffic-subset_coco-format.json
OI_ANN_PATH_OOD=/mnt/work/dhuseljic/datasets/open-images/open-images_ood-subset_coco-format.json
echo '=== Starting script: ANIMALS (Open-Images) ==='
# srun python /mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/evaluate.py \
#     --dataset open-images \
#     --ds_path $OI_PATH \
#     --ann_path $OI_ANN_PATH_ANIMALS \
#     --experiment_path $FRCNN_OUTPUT_ANIMALS \
#     --result_path "/mnt/work/dhuseljic/uod_results/frcnn_animals_shifted_${PERSPECTIVE}" \
#     --perspective $PERSPECTIVE \
#     SEED 42 
echo '=== Starting script: TRAFFIC (Open-Images) ==='
# srun python /mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/evaluate.py \
#     --dataset open-images \
#     --ds_path $OI_PATH \
#     --ann_path $OI_ANN_PATH_TRAFFIC \
#     --experiment_path $FRCNN_OUTPUT_TRAFFIC \
#     --result_path "/mnt/work/dhuseljic/uod_results/frcnn_traffic_shifted_${PERSPECTIVE}" \
#     --perspective $PERSPECTIVE \
#     SEED 42 
echo '=== Starting script: OOD (Open-Images) ==='
# srun python /mnt/home/dhuseljic/projects/uncertainty-in-object-detection/src/evaluate.py \
#     --dataset open-images \
#     --ds_path $OI_PATH \
#     --ann_path $OI_ANN_PATH_OOD \
#     --experiment_path $FRCNN_OUTPUT_TRAFFIC \
#     --result_path "/mnt/work/dhuseljic/uod_results/frcnn_ood_${PERSPECTIVE}" \
#     --perspective $PERSPECTIVE \
#     SEED 42 
