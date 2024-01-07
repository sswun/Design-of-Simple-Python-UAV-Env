# 导入基本的模型
from .basic_model import *
from .sup_model import *
from .layer_Conv import *
from .layer_Pooling import *
from .layer_rnn import *
from .attention_mechanism import *
from copy import deepcopy

# 导入需要支持的库
import yaml

# 一些需要的函数设计
model = nn.Module if model_name=='torch' else nn.Layer
## 选择运算设备
def choose_device(device):
    '''
    输入为str:'gpu'；'cpu'
    '''
    if device.lower()=='GPU'.lower():
        return try_gpu()
    else:
        return dlmodel.device('cpu') \
            if model_name=='torch' else dlmodel.CPUPlace()

## 根据模型选择优化器（使用几个常用优化器）
def choose_optimizer(params, optimizer=['sgd', None]):
    '''
    输入params为模型的参数，optimizer为两项，第一项为优化器类型，第二项为参数
    当第二项为None时，会有默认的参数初始化
    主要为下列方法：sgd,momentum,adagrad,rmsprop,adadelta,adam
    '''
    if optimizer[0].lower()=='sgd':
        if model_name=='torch':
            if optimizer[1]==None or optimizer[1]=='None':
                optimizer[1] = {'lr': 0.01}
            return dlmodel.optim.SGD(params=params, **optimizer[1])
        else:
            if optimizer[1]==None or optimizer[1]=='None':
                optimizer[1] = {'learning_rate': 0.01}
            return dlmodel.optimizer.SGD(**optimizer[1], parameters=params)

    elif optimizer[0].lower()=='asgd':
        if model_name=='torch':
            if optimizer[1]== None or optimizer[1]=='None':
                optimizer[1] = {'lr': 0.01}
            return dlmodel.optim.ASGD(params=params, **optimizer[1])
        else:
            # paddle不便于设置该参数
            raise NotImplementedError
        
    elif optimizer[0].lower()=='momentum':
        if model_name=='torch':
            if optimizer[1]==None or optimizer[1]=='None':
                optimizer[1] = {'lr': 0.01}
            return dlmodel.optim.SGD(params=params, **optimizer[1])
        else:
            if optimizer[1]==None or optimizer[1]=='None':
                optimizer[1] = {'learning_rate': 0.01}
                return dlmodel.optimizer.Momentum(**optimizer[1], parameters=params)
    
    elif optimizer[0].lower()=='adagrad':
        if model_name=='torch':
            if optimizer[1]==None or optimizer[1]=='None':
                optimizer[1] = {'lr': 0.01}
            return dlmodel.optim.Adagrad(params=params, **optimizer[1])
        else:
            if optimizer[1]==None or optimizer[1]=='None':
                optimizer[1] = {'learning_rate': 0.01}
            return dlmodel.optimizer.Adagrad(**optimizer[1], parameters=params)
    
    elif optimizer[0].lower()=='rmsprop':
        if model_name=='torch':
            if optimizer[1]==None or optimizer[1]=='None':
                optimizer[1] = {'lr': 0.01}
            return dlmodel.optim.RMSprop(params=params, **optimizer[1])
        else:
            if optimizer[1]==None or optimizer[1]=='None':
                optimizer[1] = {'learning_rate': 0.01}
            return dlmodel.optimizer.RMSProp(**optimizer[1], parameters=params)
        
    elif optimizer[0].lower()=='adadelta':
        if model_name=='torch':
            if optimizer[1]==None or optimizer[1]=='None':
                optimizer[1] = {'lr': 0.01}
            return dlmodel.optim.Adadelta(params=params, **optimizer[1])
        else:
            if optimizer[1]==None or optimizer[1]=='None':
                optimizer[1] = {'learning_rate': 0.01}
            return dlmodel.optimizer.Adadelta(**optimizer[1], parameters=params)
        
    elif optimizer[0].lower()=='adam':
        if model_name=='torch':
            if optimizer[1]==None or optimizer[1]=='None':
                optimizer[1] = {'lr': 0.01}
            return dlmodel.optim.Adam(params=params, **optimizer[1])
        else:
            if optimizer[1]==None or optimizer[1]=='None':
                optimizer[1] = {'learning_rate': 0.01}
            return dlmodel.optimizer.Adam(**optimizer[1], parameters=params)
        
    else:
        if model_name=='torch':
            optim = eval('dlmodel.optim.'+optimizer[0])
            return optim(params=params, **optimizer[1])
        else:
            optim = eval('dlmodel.optimizer.'+optimizer[0])
            return optim(**optimizer[1], parameters=params)

## 根据模型选择损失函数
def choose_loss(loss=['MSELoss', {'reduction':'mean'}]):
    '''
    输入loss为两项，第一项为损失函数类型，第二项为参数
    '''
    if loss[1] == None or loss[1]=='None':
        loss[1] = {'reduction': 'mean'}
    lossFun = eval('dlmodel.nn.'+loss[0])
    a = loss[1]
    return lossFun(**a)

# 定义一个标准型网络
class Net(nn.Module if model_name=='torch' else nn.Layer):
    def __init__(self, netlist, layers):
        super(Net, self).__init__()
        self.netlist = netlist
        self.layers = layers
                        
    def forward(self, x):
        output = [x]
        for i, net in enumerate(self.netlist):
            if self.layers[i][0] == -1:
                output.append(net(output[-1]))
            else:
                temp_x = []
                for j in self.layers[i][0]:
                    if j != -1:
                        j1 = j+1
                        temp_x.append(output[j1])
                    else:
                        temp_x.append(output[-1])
                output.append(net(temp_x))
        return output[-1]
                

class model_set(nn.Module if model_name=='torch' else nn.Layer):
    '''将模型导入到model_set，简化训练，预测，参数修改，导入等过程'''
    def __init__(self, model=None, device=try_gpu(), optimizer=None, loss=None,
                 task_type=None):
        '''
        导入的参数model为网络net模型，可以为torch中nn.Module类型，可以为paddle中nn.Layer类型，
        且只能为未初始化的定义模型，或者为yaml形式文件，也可以先不导入model利用load_model加载模型
        虽然model可以先不导入，但是device,optimizer,loss一定要按形式导入
        
        device为模型训练时使用的设备，参数为'cpu'或'gpu',默认将尝试gpu
        
        optimizer为优化器，传入参数默认建议为一列表，列表包含两项，第一项
        为str，内容为优化器名称，比如'SGD'；第二项为dict，包含需要初始化的参数。
        也可以为自定义的优化器（父类为dlmodel.optim.Optimizer或
        dlmodel.optimizer.Optimizer）
        
        loss为损失函数，传入参数默认建议为一列表，列表包含两项，第一项
        为str，内容为损失函数名称，比如'MSELoss'；第二项为dict，
        包含需要初始化的参数。
        也可以为自定义的损失函数（父类为nn.Module或nn.Layer）
        
        task_type为任务类型:None默认训练测试方法，'classification'分类任务
        主要区别在于标签和输出的对应方式。
        '''
        super(model_set, self).__init__()
        self.model = model  # 网络模型
        self.model_label = None  # 标记当初导入模型时是哪种方式导入的
        self.device = device  # 运算设备
        self.optimizer = optimizer  # 优化器
        self.loss = loss  # 损失函数
        self.task_type = task_type # 任务类型
        
    def refresh(self):
        if self.model is None:
            return
        elif isinstance(self.model, str):
            self.refresh_str()
            return
        if model_name=='torch': # 判断导入的深度学习工具库是啥
            if isinstance(self.model, nn.Module):
                self.model_label = 'torch'
                if isinstance(self.device, str):
                    self.device = choose_device(self.device)
                self.model = self.model.to(self.device)
                if self.optimizer is not None:
                    if isinstance(self.optimizer, list):
                        self.optimizer = choose_optimizer(
                            self.model.parameters(), self.optimizer)
                    elif ~isinstance(self.optimizer, dlmodel.optim.Optimizer):
                        raise TypeError('optimizer must inherit \
                                        from torch.optim.Optimizer')
                    else:
                        raise TypeError('传入了一个错误的参数')
                else:
                    self.optimizer = choose_optimizer(
                        self.model.parameters(), optimizer=['sgd', None])
                self.optimizer.parameters = self.model.parameters()
                if self.loss is not None:
                    if isinstance(self.loss, list):
                        self.loss = choose_loss(self.loss)
                    elif ~isinstance(self.loss, nn.Module):
                        raise TypeError('loss must inherit \
                                        from nn.Module')
                    else:
                        raise TypeError('传入了一个错误的参数')
                else:
                    self.loss = choose_loss()
                    
        else:
            if isinstance(self.model, nn.Layer):
                self.model_label = 'paddle'
                if isinstance(self.device, str):
                    self.device = choose_device(self.device)
                self.model = self.model.to(self.device)
                if self.optimizer is not None:
                    if isinstance(self.optimizer, list):
                        self.optimizer = choose_optimizer(
                            self.model.parameters(), self.optimizer)
                    elif ~isinstance(self.optimizer, dlmodel.optimizer.Optimizer):
                        raise TypeError('optimizer must inherit \
                                        from torch.optim.Optimizer')
                    else:
                        raise TypeError('传入了一个错误的参数')
                else:
                    self.optimizer = choose_optimizer(
                        self.model.parameters(), optimizer=['sgd', None])
                self.optimizer.parameters = self.model.parameters()
                if self.loss is not None:
                    if isinstance(self.loss, list):
                        self.loss = choose_loss(self.loss)
                    elif ~isinstance(self.loss, nn.Layer):
                        raise TypeError('loss must inherit \
                                        from nn.Module')
                    else:
                        raise TypeError('传入了一个错误的参数')
                else:
                    self.loss = choose_loss()
    
    def refresh_str(self):
        '''
        通过yaml文件初始化
        '''
        if isinstance(self.model, str):  # 此时，model只有可能是
            # yaml形式文件的路径位置
            self.model_label = 'yaml'
            f = open(self.model, encoding='UTF-8')
            model_dict = yaml.load(f, Loader=yaml.FullLoader)
            # 设置device
            self.device = choose_device(model_dict['device'])
            # 设置loss
            self.loss = choose_loss(model_dict['loss'])
            # 设置网络
            layers = model_dict['layer']
            if model_name == 'torch':
                netlist = nn.ModuleList()
            else:
                netlist = nn.LayerList()
            for layer_select in layers: # 提取网络层放入
                if layer_select[3]!='None':
                    net_layer = [eval(layer_select[2]+'(*layer_select[3])')
                                for i in range(layer_select[1])]
                else:
                    net_layer = [eval(layer_select[2]+'()')
                                for i in range(layer_select[1])]
                net_Sequential = nn.Sequential(*net_layer)
                netlist.append(net_Sequential)
            # 根据前向传播定义模型
            self.model = Net(netlist, layers).to(self.device)
            # 设置optimizer
            self.optimizer = choose_optimizer(self.model.parameters(), 
                                            model_dict['optimizer'])
    
    def train(self, data_iter, num_epochs=5):
        '''
        通过数据迭代器训练模型，通用训练方法
        '''
        self.model.train()
        self.optimizer.parameters = self.model.parameters()
        if self.task_type == 'classification':
            self.train_classify(data_iter, num_epochs)
            return
        for i in range(num_epochs):
            for X, y in data_iter:
                if model_name == 'torch':
                    X, y = X.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                else:
                    X, y = X._to(self.device), y._to(self.device)
                    self.optimizer.clear_grad()
                out = self.model(X)
                y = y.reshape(out.shape)
                l = self.loss(out, y)
                l.mean().backward()
                self.optimizer.step()
            print(f'loss: {l}, {i} sec/epoch')
    
    def train_classify(self, data_iter, num_epochs=5):
        '''
        通过数据迭代器训练模型，分类专用训练方法
        '''
        for i in range(num_epochs):
            for X, y in data_iter:
                if model_name == 'torch':
                    X, y = X.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                else:
                    X, y = X._to(self.device), y._to(self.device)
                    self.optimizer.clear_grad()
                out = self.model(X)
                l = self.loss(out, y)
                l.backward()
                self.optimizer.step()
            print(f'loss: {l}, {i} sec/epoch')
    
    def test(self, data_iter, isreturn=False, max_time=10):
        '''
        通过数据迭代器测试模型，通用测试方法
        '''
        self.model.eval()
        if self.task_type == 'classification':
            if isreturn:
                return self.test_classify(data_iter, isreturn, max_time)
            else:
                self.test_classify(data_iter, isreturn, max_time)
                return
        out1 = []
        i = 0
        for X, y in data_iter:
            i += 1
            if i > max_time:
                break
            if model_name == 'torch':
                X, y = X.to(self.device), y.to(self.device)
            else:
                X, y = X._to(self.device), y._to(self.device)
            out = self.model(X)
            y = y.reshape(out.shape)
            l = self.loss(out, y)
            out1.append((out, y))
            print(f'loss: {l}')
        if isreturn:
            return out1
    
    def test_classify(self, data_iter, isreturn=False, max_time=10):
        '''
        通过数据迭代器测试模型，分类专用测试方法
        '''
        out1 = []
        i = 0
        for X, y in data_iter:
            i += 1
            if i > max_time:
                break
            if model_name == 'torch':
                X, y = X.to(self.device), y.to(self.device)
            else:
                X, y = X._to(self.device), y._to(self.device)
            out = self.model(X)
            l = self.loss(out, y)
            out1.append((out, y))
            print(f'loss: {l}')
        if isreturn:
            return out1

    def predict(self, X):
        """
        给定输入X，输出预测值，通用预测方法
        """
        if model_name == 'torch':
            X = X.to(self.device)
        else:
            X = X._to(self.device)
        with dlmodel.no_grad():
            return self.model(X)
    
    def save_yaml(self, path):
        '''
        保存模型结构为yaml，注意此处至少简单保存了模型的信息，并不能作为初始化调用的yaml文件
        '''
        f = open(path,'w')
        out_dict = {}
        if self.device == try_gpu():
            out_dict.update({'device':'GPU'})
        out_dict.update({'optimizer':type(self.optimizer).__name__})
        out_dict.update({'loss':type(self.loss).__name__})
        out_dict.update({'layers':self.model.layers})
        yaml.dump(out_dict, f)
        f.close()
    
    def save_model(self, path, is_only_weights=False):
        '''
        保存模型为.pt文件
        '''
        if is_only_weights:
            dlmodel.save(self.model.state_dict(), path)
            return
        dlmodel.save(self.model, path)
        
    def load_model(self, path, is_only_weights=False):
        '''
        加载模型为.pt文件
        '''
        if is_only_weights:
            self.model.load_state_dict(f=path)
        self.model = dlmodel.load(f=path)
    
    def function_apply(self,fun):
        '''
        调用自定义的函数对该类属性进行操作，传入参数必须为model_set属性
        属性为model，device，optimizer，loss
        '''
        fun(self.model, self.device, self.optimizer, self.loss)
        return
