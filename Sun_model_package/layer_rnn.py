from .basic_model import *

# 该内容主要为各类网络层
## 循环神经网络相关
# torch.nn.RNN(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh',
# bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
# class paddle.nn.SimpleRNN(input_size, hidden_size, num_layers=1, activation='tanh',
# direction='forward', dropout=0., time_major=False, weight_ih_attr=None,
# weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None)
# direction (str，可选) - 网络迭代方向，可设置为 forward 或 bidirect
class RNN_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    循环神经网络：RNN
    attention:
    The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
    '''
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh',
        bias=True, batch_first=False, dropout=0.0, bidirectional=False):
        super(RNN_layer, self).__init__()
        if model_name=='torch':
            self.RNN = nn.RNN(input_size, hidden_size, num_layers, nonlinearity,
                              bias, batch_first, dropout, bidirectional)
        else:
            self.RNN = nn.SimpleRNN(input_size, hidden_size, num_layers, activation=nonlinearity,
                                    direction='bidirect' if bidirectional else 'forward',
                                     dropout=dropout, time_major=~batch_first)
    
    def forward(self, x, h=None):
        output, final_state = self.RNN(x, h)
        return output, final_state

# torch.nn.LSTM(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
# dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
# paddle.nn.LSTM(input_size, hidden_size, num_layers=1, direction='forward', dropout=0.,
# time_major=False, weight_ih_attr=None, weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None, name=None)
class LSTM_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    循环神经网络：LSTM
    '''
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False):
        super(LSTM_layer, self).__init__()
        if model_name=='torch':
            self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, 
                                bias, batch_first, dropout, bidirectional)
        else:
            self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, 
                                direction='bidirect' if bidirectional else 'forward',
                                     dropout=dropout, time_major=~batch_first)
    
    def forward(self, x, h=None):
        output, final_state = self.LSTM(x, h)
        return output, final_state

# torch.nn.GRU(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
# dropout=0.0, bidirectional=False, device=None, dtype=None)
# paddle.nn.GRU(input_size, hidden_size, num_layers=1, direction='forward',
# dropout=0., time_major=False,
# weight_ih_attr=None, weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None, name=None)
class GRU_layer(nn.Module if model_name=='torch' else nn.Layer):
    '''
    循环神经网络：GRU
    '''
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False):
        super(GRU_layer, self).__init__()
        if model_name=='torch':
            self.GRU = nn.GRU(input_size, hidden_size, num_layers,
                              bias, batch_first, dropout, bidirectional)
        else:
            self.GRU = nn.GRU(input_size, hidden_size, num_layers,
                              direction='bidirect' if bidirectional else 'forward',
                                     dropout=dropout, time_major=~batch_first)
        
    def forward(self, x, h=None):
        output, final_state = self.GRU(x, h)
        return output, final_state
