# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Common modules
"""
import sys
sys.path.insert(1,"/ibex/user/silvada/detectionTools/")


import math

import torch
import torch.nn as nn

import numpy as np

from ultralytics.yolo.utils.tal import dist2bbox, make_anchors
from ultralytics.yolo.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights,
                                                intersect_dicts, make_divisible, model_info, scale_img, time_sync)

import torch.nn.functional as F
from thop import profile

class Conv_LSTM(nn.Module):
### Inspired from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py#
      def __init__(self, in_channels, out_channels): 
       super().__init__()
       
       self.default_act = nn.Tanh()
       self.default_act2 = nn.Sigmoid()
       self.in_channels = in_channels
       self.out_channels = out_channels
      
       # input gate 
       self.Gates = nn.Conv2d(self.in_channels + self.out_channels , 4 * self.out_channels, 1)
       
  
      def forward(self,x, hidden_states):
        n, c, h, w = x.shape
        if not hidden_states:
            hidden_states = self.init_hidden(n, h, w)

        prev_h = hidden_states[0]
        prev_c = hidden_states[1]

        stack_inputs = torch.cat((x, prev_h), 1)
        gates = self.Gates(stack_inputs)
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate = self.default_act2(in_gate)
        remember_gate =self.default_act2(remember_gate)
        out_gate = self.default_act2(out_gate)

        # apply tanh non linearity
        cell_gate = self.default_act(cell_gate)

        # compute current cell and hidden state
        c_t = (remember_gate * prev_c) + (in_gate * cell_gate)
        h_t= out_gate * self.default_act(c_t)
        print(c_t.shape, h_t.shape)
        return [h_t, c_t]

      def init_hidden(self,n, h, w):
          return [torch.zeros((n,self.out_channels,h,w), requires_grad = False).to(self.Gates.weight),torch.zeros((n,self.out_channels,h,w),requires_grad = False).to(self.Gates.weight)]


m = Conv_LSTM(48,48)

x = torch.rand(1,48,80,80)

o = profile(m,inputs = (x,None))
print(o)
