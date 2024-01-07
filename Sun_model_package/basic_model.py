# 底层库
import numpy as np
from PIL import Image
import collections
import hashlib
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import importlib

# 检查对应的包是否安装
def is_module_available(package_name):
    loader = importlib.find_loader(package_name)
    return loader is not None

def is_dlmodel_available():
    if is_module_available("torch"):
        return "torch"
    elif is_module_available("paddle"):
        return "paddle"
    else:
        print("Please install pytorch or paddle first.")

def choose_dlmodel(package_name="torch"):
    if package_name == "torch":
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torchvision import transforms
        return torch, nn, F, transforms, 'torch'
    elif package_name == "paddle":
        import paddle
        from paddle import nn
        from paddle.nn import functional as F
        from paddle.vision import transforms
        return paddle, nn, F, transforms, "paddle"

dlmodel, nn, F, transforms, model_name= choose_dlmodel(is_dlmodel_available())
# dlmodel, nn, F, transforms, model_name= choose_dlmodel("paddle")
# 将模型转移到GPU中
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if model_name == "torch":
        if dlmodel.cuda.device_count() >= i + 1:
            return dlmodel.device(f'cuda:{i}')
        return dlmodel.device('cpu')
    elif model_name == "paddle":
        if dlmodel.device.cuda.device_count() >= i + 1:
            return dlmodel.CUDAPlace(i)
        return dlmodel.CPUPlace()
    else:
        print("Please install a valid DL model.")

def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    if model_name == "torch":
        devices = [dlmodel.device(f'cuda:{i}')
                for i in range(dlmodel.cuda.device_count())]
        return devices if devices else [dlmodel.device('cpu')]
    elif model_name == "paddle":
        devices = [dlmodel.CUDAPlace(i)
           for i in range(dlmodel.device.cuda.device_count())]
        return devices if devices else dlmodel.CPUPlace()
    else:
        print("Please install a valid DL model.")
# 基础模型库

## 线性层
class Linear_layer(nn.Module if model_name=='torch' else nn.Layer):
    """线性层。"""
    # 参数初始化
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=try_gpu(), dtype=None):
        super(Linear_layer, self).__init__()
        if model_name == "torch":
            self.fc = nn.Linear(in_features, out_features, bias=bias,
                                device=device, dtype=dtype)
        else:
            # paddle参数初始化设置
            self.fc = nn.Linear(in_features, out_features)
    # 前向传播介绍
    def forward(self, x):
        return self.fc(x)
    # 用指定参数初始化线性层
    def init_weights(self, initial_std=0.01, C=0):
        if model_name == "torch":
            self.fc.weight = nn.Parameter(
                self.fc.weight.data.normal_(0, initial_std))
            if self.fc.bias is not None:
                self.fc.bias = nn.Parameter(self.fc.bias.data.zero_() + C)
        else:
            self.fc.weight = None
    # 重设参数
    def reset_parameters(self):
        if model_name == "torch":
            self.fc.reset_parameters()
        else:
            self.fc.weight.set_value(dlmodel.zeros(shape=(self.fc.weight.shape),
                                           dtype=self.fc.weight.dtype) + 0.5)
            self.fc.bias.set_value(dlmodel.zeros(shape=(self.fc.bias.shape),
                                         dtype=self.fc.bias.dtype) + 1.0)
    
    def set_parameters(self, weight, bias=None):
        if model_name == "torch":
            self.fc.weight.data = weight
            if self.fc.bias is not None:
                self.fc.bias.data = bias
        else:
            self.fc.weight.set_value(weight)
            if self.fc.bias is not None:
                self.fc.bias.set_value(bias)
    # 查看参数
    def parameters(self):
        if model_name == "torch":
            if self.fc.bias is not None:
                return (self.fc.weight.data, self.fc.bias.data)
            else:
                return self.fc.weight.data
        else:
            return (self.fc.weight, self.fc.bias)

## Softmax层（不建议作为分类任务的输出，在损失函数中有更好的处理方法）
class Softmax_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self, dim=1):
        super(Softmax_layer, self).__init__()
        self.dim = dim

    def forward(self, x):
        if model_name == "torch":
            return F.softmax(x, dim=self.dim)
        else:
            return F.softmax(x, axis=self.dim)

## 各类激活函数层
class ELU_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(ELU_layer, self).__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        if model_name == "torch":
            return self.elu(x)
        else:
            return self.elu(x)

class Hardshrink_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(Hardshrink_layer, self).__init__()
        self.hardshrink = nn.Hardshrink()

    def forward(self, x):
        if model_name == "torch":
            return self.hardshrink(x)
        else:
            return self.hardshrink(x)

class Hardsigmoid_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(Hardsigmoid_layer, self).__init__()
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        if model_name == "torch":
            return self.hardsigmoid(x)
        else:
            return self.hardsigmoid(x)

class Hardswish_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(Hardswish_layer, self).__init__()
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        if model_name == "torch":
            return self.hardswish(x)
        else:
            return self.hardswish(x)

class Hardtanh_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self, min_val=-2, max_val=2):
        super(Hardtanh_layer, self).__init__()
        self.hardtanh = nn.Hardtanh(min_val, max_val)

    def forward(self, x):
        if model_name == "torch":
            return self.hardtanh(x)
        else:
            return self.hardtanh(x)

class LeakyReLU_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self, negative_slope=0.01):
        super(LeakyReLU_layer, self).__init__()
        self.leakyrelu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        if model_name == "torch":
            return self.leakyrelu(x)
        else:
            return self.leakyrelu(x)

class LogSigmoid_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(LogSigmoid_layer, self).__init__()
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, x):
        if model_name == "torch":
            return self.logsigmoid(x)
        else:
            return self.logsigmoid(x)

class PReLU_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self, num_parameters=1, init=0.25):
        super(PReLU_layer, self).__init__()
        self.prelu = nn.PReLU(num_parameters, init)

    def forward(self, x):
        if model_name == "torch":
            return self.prelu(x)
        else:
            return self.prelu(x)

class ReLU_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(ReLU_layer, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        if model_name == "torch":
            return self.relu(x)
        else:
            return self.relu(x)

class ReLU6_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(ReLU6_layer, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        if model_name == "torch":
            return self.relu6(x)
        else:
            return self.relu6(x)

class RReLU_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self, lower=1. / 8, upper=1. / 3):
        super(RReLU_layer, self).__init__()
        self.rrelu = nn.RReLU(lower, upper)

    def forward(self, x):
        if model_name == "torch":
            return self.rrelu(x)
        else:
            return self.rrelu(x)

class SELU_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(SELU_layer, self).__init__()
        self.selu = nn.SELU()

    def forward(self, x):
        if model_name == "torch":
            return self.selu(x)
        else:
            return self.selu(x)

class CELU_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self, gamma=1.0):
        super(CELU_layer, self).__init__()
        self.celu = nn.CELU(gamma)

    def forward(self, x):
        if model_name == "torch":
            return self.celu(x)
        else:
            return self.celu(x)

class GELU_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(GELU_layer, self).__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        if model_name == "torch":
            return self.gelu(x)
        else:
            return self.gelu(x)

class SiLU_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(SiLU_layer, self).__init__()
        self.silu = nn.SiLU()

    def forward(self, x):
        if model_name == "torch":
            return self.silu(x)
        else:
            return self.silu(x)

class Sigmoid_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(Sigmoid_layer, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if model_name == "torch":
            return self.sigmoid(x)
        else:
            return self.sigmoid(x)

class Mish_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(Mish_layer, self).__init__()
        self.mish = nn.Mish()

    def forward(self, x):
        if model_name == "torch":
            return self.mish(x)
        else:
            return self.mish(x)

class Softplus_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(Softplus_layer, self).__init__()
        self.softplus = nn.Softplus()

    def forward(self, x):
        if model_name == "torch":
            return self.softplus(x)
        else:
            return self.softplus(x)

class Softshrink_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(Softshrink_layer, self).__init__()
        self.softshrink = nn.Softshrink()

    def forward(self, x):
        if model_name == "torch":
            return self.softshrink(x)
        else:
            return self.softshrink(x)

class Softsign_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(Softsign_layer, self).__init__()
        self.softsign = nn.Softsign()

    def forward(self, x):
        if model_name == "torch":
            return self.softsign(x)
        else:
            return self.softsign(x)

class Tanh_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(Tanh_layer, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        if model_name == "torch":
            return self.tanh(x)
        else:
            return self.tanh(x)

class Tanhshrink_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self):
        super(Tanhshrink_layer, self).__init__()
        self.tanhshrink = nn.Tanhshrink()

    def forward(self, x):
        if model_name == "torch":
            return self.tanhshrink(x)
        else:
            return self.tanhshrink(x)

class Threshold_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self, th=0.1, th_hard=20):
        super(Threshold_layer, self).__init__()
        self.threshold = nn.Threshold(th, th_hard)

    def forward(self, x):
        if model_name == "torch":
            return self.threshold(x)
        else:
            return self.threshold(x)

## 暂退层
class Dropout_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self, p=0.5):
        super(Dropout_layer, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        if model_name == "torch":
            return self.dropout(x)
        else:
            return self.dropout(x)

## 连接层(在某维度上连接两输出张量)
class Concat_layer(nn.Module if model_name=='torch' else nn.Layer):
    """Concatenate a list of tensors along dimension."""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        if model_name == "torch":
            return dlmodel.cat(x, self.d)
        else:
            return dlmodel.concat(x, self.d)

## 展平层
class Flatten_layer(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten_layer, self).__init__()
        self.Flatten = nn.Flatten(start_dim, end_dim)

    def forward(self, x):
        return self.Flatten(x)

