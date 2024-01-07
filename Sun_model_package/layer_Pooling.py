from .basic_model import *

# 该内容主要为各类网络层
## 池化层相关
class MaxPool1d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    一维最大池化层
    '''
    def __init__(self, kernel_size=2, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool1d_layer, self).__init__()
        if model_name == 'torch':
            self.MaxPool1d = nn.MaxPool1d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        else:
            self.MaxPool1d = nn.MaxPool1D(kernel_size, stride, padding, return_mask=return_indices,
                                          ceil_mode=ceil_mode)

    def forward(self, x):
        return self.MaxPool1d(x)

class MaxPool2d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    二维最大池化层
    '''
    def __init__(self, kernel_size=2, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool2d_layer, self).__init__()
        if model_name == 'torch':
            self.MaxPool2d = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        else:
            self.MaxPool2d = nn.MaxPool2D(kernel_size, stride, padding, return_mask=return_indices,
                                          ceil_mode=ceil_mode)

    def forward(self, x):
        return self.MaxPool2d(x)

class MaxPool3d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    三维最大池化层
    '''
    def __init__(self, kernel_size=2, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool3d_layer, self).__init__()
        if model_name == 'torch':
            self.MaxPool3d = nn.MaxPool3d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        else:
            self.MaxPool3d = nn.MaxPool3D(kernel_size, stride, padding, return_mask=return_indices,
                                          ceil_mode=ceil_mode)

    def forward(self, x):
        return self.MaxPool3d(x)

class AvgPool1d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    一维平均池化层
    '''
    def __init__(self, kernel_size=2, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super(AvgPool1d_layer, self).__init__()
        if model_name == 'torch':
            self.AvgPool1d = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode, count_include_pad)
        else:
            self.AvgPool1d = nn.AvgPool1D(kernel_size, stride, padding, ceil_mode=ceil_mode,
                                          exclusive=count_include_pad)
            
    def forward(self, x):
        return self.AvgPool1d(x)

class AvgPool2d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    二维平均池化层
    '''
    def __init__(self, kernel_size=2, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super(AvgPool2d_layer, self).__init__()
        if model_name == 'torch':
            self.AvgPool2d = nn.AvgPool2d(kernel_size, stride, padding, ceil_mode, count_include_pad)
        else:
            self.AvgPool2d = nn.AvgPool2D(kernel_size, stride, padding, ceil_mode=ceil_mode,
                                          exclusive=count_include_pad)
            
    def forward(self, x):
        return self.AvgPool2d(x)

class AvgPool3d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    一维平均池化层
    '''
    def __init__(self, kernel_size=2, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super(AvgPool3d_layer, self).__init__()
        if model_name == 'torch':
            self.AvgPool3d = nn.AvgPool3d(kernel_size, stride, padding, ceil_mode, count_include_pad)
        else:
            self.AvgPool3d = nn.AvgPool3D(kernel_size, stride, padding, ceil_mode=ceil_mode,
                                          exclusive=count_include_pad)
            
    def forward(self, x):
        return self.AvgPool3d(x)

class BatchNorm1d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    一维batch归一化层
    '''
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None):
        super(BatchNorm1d_layer, self).__init__()
        if model_name == 'torch':
            self.BatchNorm1d = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats,
                                              device, dtype)
        else:
            self.BatchNorm1d = nn.BatchNorm1D(num_features, epsilon=eps, momentum=momentum)
    
    def forward(self, x):
        return self.BatchNorm1d(x)

class BatchNorm2d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    二维batch归一化层
    '''
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None):
        super(BatchNorm2d_layer, self).__init__()
        if model_name == 'torch':
            self.BatchNorm2d = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats,
                                              device, dtype)
        else:
            self.BatchNorm2d = nn.BatchNorm2D(num_features, epsilon=eps, momentum=momentum)
    
    def forward(self, x):
        return self.BatchNorm2d(x)

class BatchNorm3d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    三维batch归一化层
    '''
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None):
        super(BatchNorm3d_layer, self).__init__()
        if model_name == 'torch':
            self.BatchNorm3d = nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats,
                                              device, dtype)
        else:
            self.BatchNorm3d = nn.BatchNorm3D(num_features, epsilon=eps, momentum=momentum)
    
    def forward(self, x):
        return self.BatchNorm3d(x)

class AdaptiveAvgPool1d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    一维自适应平均池化层
    '''
    def __init__(self, output_size):
        super(AdaptiveAvgPool1d_layer, self).__init__()
        if model_name == 'torch':
            self.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(output_size)
        else:
            self.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1D(output_size)
    
    def forward(self, x):
        return self.AdaptiveAvgPool1d(x)

class AdaptiveAvgPool2d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    二维自适应平均池化层
    '''
    def __init__(self, output_size):
        super(AdaptiveAvgPool2d_layer, self).__init__()
        if model_name == 'torch':
            self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size)
        else:
            self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2D(output_size)
    
    def forward(self, x):
        return self.AdaptiveAvgPool2d(x)

class AdaptiveAvgPool3d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    三维自适应平均池化层
    '''
    def __init__(self, output_size):
        super(AdaptiveAvgPool3d_layer, self).__init__()
        if model_name == 'torch':
            self.AdaptiveAvgPool3d = nn.AdaptiveAvgPool3d(output_size)
        else:
            self.AdaptiveAvgPool3d = nn.AdaptiveAvgPool3D(output_size)
    
    def forward(self, x):
        return self.AdaptiveAvgPool3d(x)

class AdaptiveMaxPool1d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    一维自适应最大池化层
    '''
    def __init__(self, output_size):
        super(AdaptiveMaxPool1d_layer, self).__init__()
        if model_name == 'torch':
            self.AdaptiveMaxPool1d = nn.AdaptiveMaxPool1d(output_size)
        else:
            self.AdaptiveMaxPool1d = nn.AdaptiveMaxPool1D(output_size)
    
    def forward(self, x):
        return self.AdaptiveMaxPool1d(x)

class AdaptiveMaxPool2d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    二维自适应最大池化层
    '''
    def __init__(self, output_size):
        super(AdaptiveMaxPool2d_layer, self).__init__()
        if model_name == 'torch':
            self.AdaptiveMaxPool2d = nn.AdaptiveMaxPool2d(output_size)
        else:
            self.AdaptiveMaxPool2d = nn.AdaptiveMaxPool2D(output_size)
    
    def forward(self, x):
        return self.AdaptiveMaxPool2d(x)

class AdaptiveMaxPool3d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    一维自适应最大池化层
    '''
    def __init__(self, output_size):
        super(AdaptiveMaxPool3d_layer, self).__init__()
        if model_name == 'torch':
            self.AdaptiveMaxPool3d = nn.AdaptiveMaxPool3d(output_size)
        else:
            self.AdaptiveMaxPool3d = nn.AdaptiveMaxPool3D(output_size)
    
    def forward(self, x):
        return self.AdaptiveMaxPool3d(x)

