#!/bin/bash

module load conda
module load cuda
conda activate reyolov8

export DATASET="./vtei_gen1"
export WEIGHTS="./weights/reyolov8s_gen1_rps"
export SPLIT="test"
export SEQ=3

python val.py --data ${DATASET}.yaml --model ${WEIGHTS}.pt --channels 5 --split ${SPLIT} --show_sequences ${SEQ}
