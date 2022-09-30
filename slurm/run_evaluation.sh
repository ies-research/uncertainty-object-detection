#!/bin/bash
sbatch ./slurm/evaluate_detr.sh
sbatch ./slurm/evaluate_faster-rcnn_application.sh
sbatch ./slurm/evaluate_faster-rcnn_modeling.sh
