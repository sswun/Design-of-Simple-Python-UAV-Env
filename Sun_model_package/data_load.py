from .basic_model import *
import numpy as np
import struct
import random
from PIL import Image
import os
# 本内容为加载数据集的一些方法
if model_name == 'torch':
    from torch.utils.data import DataLoader
else:
    from paddle.io import DataLoader

# 从idx-ubyte转换为图片及对应标签
def idx_ubyte2img(imag_path, file_size, label_path, label_size, object_path):
    '''
    对于minst分类任务数据集，其格式一般为idx-type，利用该函数将格式变为图片
    imag-path原图片所在地址,file-size为文件大小
    label-path图片标签对应地址,label-size为文件大小
    object-path图片导出地址
    '''
    data_file = imag_path
    # for minist-train, It's 47040016B, but we should set to 47040000B,so as test
    data_file_size = file_size
    data_file_size = str(data_file_size - 16) + 'B'
    data_buf = open(data_file, 'rb').read()
    magic, numImages, numRows, numColumns = struct.unpack_from(
        '>IIII', data_buf, 0)
    datas = struct.unpack_from(
        '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
    datas = np.array(datas).astype(np.uint8).reshape(
        numImages, 1, numRows, numColumns)

    label_file = label_path
    # for minist-train, It's 60008B, but we should set to 60000B,so as test
    label_file_size = label_size
    label_file_size = str(label_file_size - 8) + 'B'
    
    label_buf = open(label_file, 'rb').read()
    
    magic, numLabels = struct.unpack_from('>II', label_buf, 0)
    labels = struct.unpack_from(
        '>' + label_file_size, label_buf, struct.calcsize('>II'))
    labels = np.array(labels).astype(np.int64)
    
    datas_root = object_path
    if not os.path.exists(datas_root):
        os.mkdir(datas_root)
    
    for i in range(10):
        file_name = datas_root + os.sep + str(i)
        if not os.path.exists(file_name):
            os.mkdir(file_name)
    
    for ii in range(numLabels):
        img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
        label = labels[ii]
        file_name = datas_root + os.sep + str(label) + os.sep + \
                    'mnist_train_' + str(ii) + '.png'
        img.save(file_name)

# 从图像分类任务的文件夹中取训练集和测试集数据为地址，保存到txt中
def class_imgdata2txt(train_ratio, rootdata, path_train, path_test):
    '''
    图像分类任务数据集一般是有多个类文件夹，每个类文件夹下是图片，该函数将数据集划分为训练集测试集并返回地址
    为txt文件。
    train_ratio为训练集比例；rootdata为数据集位置，path-train为导出的训练集txt文件位置，
    path-test为导出的测试集txt位置。
    '''
    # 设置训练集，测试集比例
    test_ratio = 1 - train_ratio  # 测试集比例
    train_list, test_list = [], []
    data_list = []
    class_flag = -1  # 种类数量标记
    # 遍历rootdata文件夹，将文件名添加到data_list中
    # os.walk每次会生成一个元组(root,dirs,files)，生成次数取决于path路径下子目录个数
    # root为当前正在遍历的文件夹相对地址
    # dirs为list，返回当前正在遍历的文件夹中所有目录名字
    # files为list，返回文件夹中所有文件名
    for a, b, c in os.walk(rootdata):
        for i in range(len(c)):
            data_list.append(os.path.join(a, c[i]))
        # 将data_list中前train_ratio个文件名添加到train_list中，
        # 其余的添加到test_list中
        for i in range(0, int(len(c) * train_ratio)):
            train_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'
            train_list.append(train_data)

        for i in range(int(len(c) * train_ratio), len(c)):
            test_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'
            test_list.append(test_data)
        class_flag += 1

    # 打乱数据
    random.shuffle(train_list)
    random.shuffle(test_list)

    # 将train_list、test_list中数据保存到txt文件中
    with open(path_train, 'w', encoding='UTF-8') as f:
        for train_img in train_list:
            f.write(str(train_img))
    with open(path_test, 'w', encoding='UTF-8') as f:
        for test_img in test_list:
            f.write(test_img)

# 对于图像分类任务，从每一个类别文件夹加载数据到迭代器的方法
# 图像标准化
transform_BZ= transforms.Normalize(
    mean=[0.5, 0.5, 0.5],# 取决于数据集
    std=[0.5, 0.5, 0.5]
)

# 定义一个LoadData类，继承自Dataset类
class LoadData(dlmodel.utils.data.Dataset if model_name == 'torch' else dlmodel.io.Dataset):
    
    # 初始化函数，txt_path为txt文件路径，train_flag表示是否为训练集
    def __init__(self, txt_path, resize, train_flag=True, device=try_gpu()):
        # 获取图片信息，并将图片信息存入imgs_info中
        self.resize = float(resize)
        self.device = device
        self.imgs_info = self.get_images(txt_path)
        # 记录训练集标志
        self.train_flag = train_flag
        # 训练集变换
        self.train_tf = transforms.Compose([
                # 将图片缩放到224大小
                transforms.Resize(round(self.resize)),
                # 随机水平翻转
                transforms.RandomHorizontalFlip(),
                # 随机垂直翻转
                transforms.RandomVerticalFlip(),
                # 将图片转换为tensor
                transforms.ToTensor(),
                # 转换为BZ格式
                transform_BZ
            ])
        # 验证集变换
        self.val_tf = transforms.Compose([
                # 将图片缩放到224大小
                transforms.Resize(round(self.resize)),
                # 将图片转换为tensor
                transforms.ToTensor(),
                # 转换为BZ格式
                transform_BZ
            ])

    # 获取图片信息
    def get_images(self, txt_path):
        # 打开txt文件，并将文件中的每一行按照\t分割
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
        # 返回图片信息
        return imgs_info

    # 将图片填充到self.resize大小
    def padding_black(self, img):
        # 获取图片的宽度和高度
        w, h  = img.size
        # 计算缩放比例
        scale = self.resize / max(w, h)
        # 将图片缩放到self.resize大小
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        # 获取图片的宽度和高度
        size_fg = img_fg.size
        # 设置背景图片的宽度和高度
        size_bg = round(self.resize)
        # 创建背景图片
        img_bg = Image.new("RGB", (size_bg, size_bg))
        # 将图片粘贴到背景图片上
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        # 将背景图片赋值给img
        img = img_bg
        # 返回图片
        return img

    # 获取图片和标签
    def __getitem__(self, index):
        # 获取图片路径和标签
        img_path, label = self.imgs_info[index]
        # 打开图片
        img = Image.open(img_path)
        # 将图片转换为RGB格式
        img = img.convert('RGB')
        # 将图片填充到self.resize大小
        img = self.padding_black(img)
        # 如果是训练集，则使用训练集变换，否则使用验证集变换
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        if model_name == 'torch':
            img = img.to(self.device)
        else:
            img = img._to(self.device)
        # 将标签转换为int类型
        label = int(label)
        # 返回图片和标签
        return img, label

    # 获取图片数量
    def __len__(self):
        # 返回图片数量
        return len(self.imgs_info)
    
    # 将方法应用于传入的图片
    def img2tensor(self, path_img):
        # 打开图片
        img = Image.open(path_img)
        # 将图片转换为RGB格式
        img = img.convert('RGB')
        # 将图片填充到self.resize大小
        img = self.padding_black(img)
        # 如果是训练集，则使用训练集变换，否则使用验证集变换
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        # 返回图片和标签
        return img

# 数据预处理（归一化）
# 我们定义一个类，实现二维原始tensor数据在各个特征上进行归一化，反归一化
class MinMaxScaler(object):
    def __init__(self, data=None):
        self.data = data
        
    def reinit(self, data):
        self.data = data
        
    def minmax_data(self):
        if self.data is not None:
            self.data_min = self.data.min(axis=0)
            self.data_max = self.data.max(axis=0)
            if model_name == 'torch':
                self.data_norm = (self.data - self.data_min.values) \
                            / (self.data_max.values - self.data_min.values)
            else:
                self.data_norm = (self.data - self.data_min) / (self.data_max - self.data_min)   
            return self.data_norm
        else:
            print("Error: MinMaxScaler.data is None")
            return None
            
    def revert_data(self, data_norm):
        if model_name == 'torch':
            data_revert = data_norm * \
                (self.data_max.values - self.data_min.values) \
                    + self.data_min.values
        else:
            data_revert = data_norm * \
                (self.data_max - self.data_min) \
                    + self.data_min
        return data_revert
    
    def revert_data_test(self, data_test):
        if model_name == 'torch':
            data_revert = data_test * \
                (self.data_max.values[0] - self.data_min.values[0]) \
                    + self.data_min.values[0]
        else:
            data_revert = data_test * \
                (self.data_max[0] - self.data_min[0]) \
                    + self.data_min[0]
        return data_revert

# 定义一个LoadData类，继承自Dataset类，适用于时间序列预测等任务
class LoadData_alreadyset(dlmodel.utils.data.Dataset if model_name == 'torch' else dlmodel.io.Dataset):
    '''
    将数据加载为数据集，适用于已经将变量加载到工作取后，将工作区中数据打包为数据集的方法
    '''
    def __init__(self, data_feature, data_label, device=try_gpu()):
        super(LoadData_alreadyset, self).__init__()
        self.data_feature = data_feature
        self.data_label = data_label
        self.device = device
    
    def __getitem__(self, index):
        if model_name == 'torch':
            feature = self.data_feature[index].to(self.device)
            label = self.data_label[index].to(self.device)
        else:
            feature = self.data_feature[index]._to(self.device)
            label = self.data_label[index]._to(self.device)
        return feature, label
    
    def __len__(self):
        return self.data_feature.shape[0]