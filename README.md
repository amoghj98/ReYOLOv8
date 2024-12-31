# ReYOLOv8

This repository contains the source code for "A Recurrent YOLOv8-Based Framework for Event-Based Object Detection" <br/>

You can read the full paper on: <br/>
[https://arxiv.org/pdf/2408.05321] <br/>

To check-out the original YOLOv8 repo, from Utralytics, you can check: <br/>
[(https://github.com/ultralytics/ultralytics)] <br/>

# Setting up the environment 

conda create -n reyolov8 python==3.9 <br/>
conda activate reyolov8 <br/>
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge <br/>
python -m pip install opencv-python tqdm==4.66.2 pandas==2.2.2 numpy==1.22.4 psutil==5.9.8 pyyaml==6.0.1 matplotlib==3.8.4 thop wandb h5py==3.11.0 hdf5plugin==4.4.0 tensorboard==2.16.2 <br/>


# Evaluate 

python val.py --data ${DATASET}.yaml --model ${WEIGHTS}.pt  --channels 5  <br/>


# Training 

# Single-gpu

 python train.py --batch 12 --nbs 6 --epochs 100 --data ${DATASET}.yaml  --model ${MODEL_NAME}.yaml --channels 5 --name ${WANDB_RUN_NAME} --project ${WANDB_PROJECT_NAME} <br/>



# TODO MULTI-GPU


# Datasets 

The raw datasets used in this work can be found on the following links:

GEN1 - https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/
PEDRo - https://github.com/SSIGPRO/PEDRo-Event-Based-Dataset
