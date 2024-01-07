from .basic_model import *

# 该内容主要为各类网络层
## 卷积层相关
class Conv1d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    卷积层: 输入形状(batch_size, C, L)
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv1d_layer, self).__init__()
        if model_name=='torch':
            self.Conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        else:
            self.Conv1d = nn.Conv1D(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups)
        self.weight = self.Conv1d.weight
        self.bias = self.Conv1d.bias
    
    def forward(self, x):
        return self.Conv1d(x)

class Conv2d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    卷积层: 输入形状(batch_size, C, H, W)
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_layer, self).__init__()
        if model_name=='torch':
            self.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        else:
            self.Conv2d = nn.Conv2D(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups)
        self.weight = self.Conv2d.weight
        self.bias = self.Conv2d.bias
    
    def forward(self, x):
        return self.Conv2d(x)

class Con3d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    卷积层: 输入形状(batch_size, C, D, H, W)
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Con3d_layer, self).__init__()
        if model_name=='torch':
            self.Con3d = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        else:
            self.Con3d = nn.Conv3D(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups)
        self.weight = self.Con3d.weight
        self.bias = self.Con3d.bias
    
    def forward(self, x):
        return self.Con3d(x)

class ConvTranspose1d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    反卷积层: 输入形状(batch_size, C, L)
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(ConvTranspose1d_layer, self).__init__()
        if model_name=='torch':
            self.ConvTranspose1d = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                    stride, padding, output_padding, groups, bias, dilation)
        else:
            self.ConvTranspose1d = nn.ConvTranspose1D(in_channels, out_channels, kernel_size,
                                    stride, padding, output_padding, groups, dilation)
        self.weight = self.ConvTranspose1d.weight
        self.bias = self.ConvTranspose1d.bias
    
    def forward(self, x):
        return self.ConvTranspose1d(x)

class ConvTranspose2d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    反卷积层: 输入形状(batch_size, C, H, W)
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(ConvTranspose2d_layer, self).__init__()
        if model_name=='torch':
            self.ConvTranspose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                    stride, padding, output_padding, groups, bias, dilation)
        else:
            self.ConvTranspose2d = nn.ConvTranspose2D(in_channels, out_channels, kernel_size,
                                    stride, padding, output_padding, groups, dilation)
        self.weight = self.ConvTranspose2d.weight
        self.bias = self.ConvTranspose2d.bias
    
    def forward(self, x):
        return self.ConvTranspose2d(x)

class ConvTranspose3d_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    反卷积层: 输入形状(batch_size, C, D, H, W)
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(ConvTranspose3d_layer, self).__init__()
        if model_name=='torch':
            self.ConvTranspose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                    stride, padding, output_padding, groups, bias, dilation)
        else:
            self.ConvTranspose3d = nn.ConvTranspose3D(in_channels, out_channels, kernel_size,
                                    stride, padding, output_padding, groups, dilation)
        self.weight = self.ConvTranspose3d.weight
        self.bias = self.ConvTranspose3d.bias
    
    def forward(self, x):
        return self.ConvTranspose3d(x)

