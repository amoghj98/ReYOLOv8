#!/bin/bash

module load conda
module load cuda
conda activate reyolov8

export DATASET="./vtei_gen1"
export WEIGHTS="./weights/reyolov8s_gen1_rps"
export SPLIT="test"
export SEQ=3
export HYP="/home/joshi157/ReYOLOv8/default_gen1"
export WANDB_RUN_NAME="toffe_reyolov8_exp"
export WANDB_PROJECT_NAME="toffe"
export MODEL_NAME="/home/joshi157/ReYOLOv8/ultralytics/models/v8/Recurrent/ReYOLOV8s"
export BATCH=48

# setting hyperparameters in the yaml itself, not here!

python train.py --batch ${BATCH} --nbs ${BATCH//2} --channels 5 --hyp ${HYP}.yaml --data ${DATASET}.yaml --model ${MODEL_NAME}.yaml --name ${WANDB_RUN_NAME} --project ${WANDB_PROJECT_NAME}

# python train.py --batch ${BATCH} --nbs ${BATCH//2} --epochs ${NUM_EPOCH} --data ${DATASET}.yaml  --model ${MODEL_NAME}.yaml --channels 5 --name ${WANDB_RUN_NAME} --project ${WANDB_PROJECT_NAME}  --hyp ${HYP}.yaml --suppress ${S} --positive ${P} --zoom_out ${Z} --flip ${F} --val_epoch ${VAL_EPOCH} --clip_length ${CLIP_LENGTH} --clip_stride ${CLIP_STRIDE}
