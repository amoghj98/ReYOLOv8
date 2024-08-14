import ast
import contextlib
import json
import platform
import zipfile
from collections import OrderedDict, namedtuple
from pathlib import Path
from urllib.parse import urlparse
import sys
sys.path.append("/ibex/user/silvada/detectionTools/")
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ultralytics.yolo.utils import LOGGER, ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_requirements, check_suffix, check_version, check_yaml

from ultralytics.yolo.utils.ops import xywh2xyxy
from ultralytics.nn.tasks import DetectionModel2

from thop import profile

fuse = True
device = "cuda:0"

weights = "sample_ryolov8n_gen1.pt"

from ultralytics.nn.tasks import attempt_load_weights

model = torch.load(weights)

print(model)

def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it
    catches the error, logs a warning message, and attempts to install the missing module via the check_requirements()
    function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.

    Returns:
        The loaded PyTorch model.
    """
    from ultralytics.yolo.utils.downloads import attempt_download_asset

    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        return torch.load(file, map_location='cpu'), file  # load
    except ModuleNotFoundError as e:
        if e.name == 'omegaconf':  # e.name is missing module name
            LOGGER.warning(f'WARNING ⚠️ {weight} requires {e.name}, which is not in ultralytics requirements.'
                           f'\nAutoInstall will run now for {e.name} but this feature will be removed in the future.'
                           f'\nRecommend fixes are to train a new model using updated ultralytics package or to '
                           f'download updated models from https://github.com/ultralytics/assets/releases/tag/v0.0.0')
        if e.name != 'models':
            check_requirements(e.name)  # install missing module
        return torch.load(file, map_location='cpu'), file  # load
        

     

def get_model(cfg=None, weights=None, verbose=True):
        model = DetectionModel2(cfg, imgsz = 320, ch= 5, nc=2, verbose=True)
        if weights:
            model.load(weights)

        return model

#model = get_model(cfg = "/home/silvada/Desktop/Projects2024/detectionTools/ultralytics/models/v8/yolov9.yaml")

#torch.save(model, "model.pt")      

model = get_model(cfg = "/ibex/user/silvada/detectionTools/ultralytics/models/v8/ConvLSTM/ConvLSTM2_minus_yolov8s.yaml")

m = model.to("cuda:0")
x = torch.rand(1,5,320,256).to("cuda:0")


y = m(x, profile = True)

#flops, params = profile(m, inputs=(x, ))



