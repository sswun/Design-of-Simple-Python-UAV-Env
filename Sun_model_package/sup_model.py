from .basic_model import *
from .layer_Conv import *
from .layer_Pooling import *

# 特别设计块
# 图像分类的经典卷积块
def choose_actFun(actFun_name='ReLU', param=None):
    '''
    一个好用的网络层选择器，输入函数名称str及参数dict
    函数将在Sun包中，以及torch或paddle中找对应的nn层
    '''
    try:
        if param is None or param =='None':
            actFun = eval(actFun_name+'_layer()')
        else:
            actFun = eval(actFun_name+'_layer(**param)')
    except NameError:
        try:
            if param is None or param =='None':
                actFun = eval('nn.'+actFun_name+'()')
            else:
                actFun = eval('nn.'+actFun_name+'(**param)')
        except NameError:
            raise NameError('activation function is not found')
    return actFun

class Conv2dBlock(nn.Module if model_name=='torch' else nn.Layer):
    '''
    经典卷积神经网络的基本组成部分是下面的这个序列：  
        - 带填充以保持分辨率的卷积层；  
        - 非线性激活函数，如ReLU；  
        - 汇聚层，如最大汇聚层。
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, actFun='ReLU', pooling='Max'):
        super(Conv2dBlock, self).__init__()
        self.Conv2d_layer = Conv2d_layer(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups, bias)
        self.actFun = choose_actFun(actFun)
        if pooling == 'Max':
            self.pooling = MaxPool2d_layer(kernel_size=2, stride=2, padding=0)
        elif pooling == 'Avg':
            self.pooling = AvgPool2d_layer(kernel_size=2, stride=2, padding=0)
        else:
            self.pooling = choose_actFun(pooling)

    def forward(self, x):
        x = self.Conv2d_layer(x)
        x = self.actFun(x)
        x = self.pooling(x)
        return x

class VGGBlock(nn.Module if model_name=='torch' else nn.Layer):
    '''
    VGG块:由一系列卷积层+激活层组成，最后为汇聚层
    '''
    def __init__(self, in_channels, out_channels, num_Conv2d_layer=1, actFun='ReLU', pooling='Max'):
        super(VGGBlock, self).__init__()
        self.Conv2d_layers = []
        for _ in range(num_Conv2d_layer):
            self.Conv2d_layers.append(Conv2d_layer(in_channels, out_channels, kernel_size=3, padding=1))
            self.Conv2d_layers.append(choose_actFun(actFun))
            in_channels = out_channels
        if pooling == 'Max':
            self.Conv2d_layers.append(MaxPool2d_layer(kernel_size=2, stride=2, padding=0))
        elif pooling == 'Avg':
            self.Conv2d_layers.append(AvgPool2d_layer(kernel_size=2, stride=2, padding=0))
        else:
            self.Conv2d_layers.append(choose_actFun(pooling))
        self.Conv2d_layers = nn.Sequential(*self.Conv2d_layers)
    
    def forward(self, x):
        return self.Conv2d_layers(x)

class NiNBlock(nn.Module if model_name=='torch' else nn.Layer):
    '''
    NiN块:由一个卷积层+一个线性层+一个线性层组成，线性层有卷积核大小为1的卷积层代替
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(NiNBlock, self).__init__()
        self.Conv2d_layers = []
        self.Conv2d_layers.append(Conv2d_layer(in_channels, out_channels,
                                               kernel_size=kernel_size, stride=stride, padding=padding))
        self.Conv2d_layers.append(Conv2d_layer(out_channels, out_channels, 
                                               kernel_size=1, stride=1, padding=0))
        self.Conv2d_layers.append(Conv2d_layer(out_channels, out_channels, 
                                               kernel_size=1, stride=1, padding=0))
        self.Conv2d_layers = nn.Sequential(*self.Conv2d_layers)
    
    def forward(self, x):
        return self.Conv2d_layers(x)

class InceptionBlock(nn.Module if model_name=='torch' else nn.Layer):
    '''
    Inception块:由四个分支组成
    输入参数c1--c4是每条路径的输出通道数
    由Inception结构，输入的c1,c4为1，4分支卷积层通道数，类型为int；
    输入的c2,c3为2，3分支卷积层通道数，由于有两层卷积层，因此需要两个通道数，类型为list或turple
    '''
    def __init__(self, in_channels=192, c1=64, c2=(96, 128), c3=(16, 32), c4=32):
        super(InceptionBlock, self).__init__()
        # 线路1，单1x1卷积层
        self.p1_1 = Conv2d_layer(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = Conv2d_layer(in_channels, c2[0], kernel_size=1)
        self.p2_2 = Conv2d_layer(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = Conv2d_layer(in_channels, c3[0], kernel_size=1)
        self.p3_2 = Conv2d_layer(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = MaxPool2d_layer(kernel_size=3, stride=1, padding=1)
        self.p4_2 = Conv2d_layer(in_channels, c4, kernel_size=1)
        # 激活层
        self.relu = ReLU_layer()
        # 连接层
        self.cat = Concat_layer(dimension=1)
        
    def forward(self, x):
        p1 = self.relu(self.p1_1(x))
        p2 = self.relu(self.p2_2(self.relu(self.p2_1(x))))
        p3 = self.relu(self.p3_2(self.relu(self.p3_1(x))))
        p4 = self.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return self.cat([p1, p2, p3, p4])

class ResidualBlock(nn.Module if model_name=='torch' else nn.Layer): 
    '''
    残差网络
    '''
    def __init__(self, in_channels, out_channels,
                 use_1x1conv=False, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2d_layer(in_channels, out_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = Conv2d_layer(in_channels, out_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = Conv2d_layer(in_channels,out_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = BatchNorm2d_layer(out_channels)
        self.bn2 = BatchNorm2d_layer(out_channels)
        self.relu = ReLU_layer()

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)

def conv_block(in_channels, num_channels):
    return nn.Sequential(
        BatchNorm2d_layer(in_channels), ReLU_layer(),
        Conv2d_layer(in_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module if model_name=='torch' else nn.Layer):
    '''
    稠密连接网络
    '''
    def __init__(self, num_convs, in_channels, num_channels):
        super(DenseBlock, self).__init__()
        self.cat = Concat_layer(dimension=1)
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + in_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = self.cat((X, Y))
        return X

def transitionBlock(input_channels, num_channels):
    '''
    过渡层
    '''
    return nn.Sequential(
        BatchNorm2d_layer(input_channels), ReLU_layer(),
        Conv2d_layer(input_channels, num_channels, kernel_size=1),
        AvgPool2d_layer(kernel_size=2, stride=2))
