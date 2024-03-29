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

from tqdm import tqdm
import collections

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

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()[0]
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return dlmodel.tensor(advantage_list, dtype=dlmodel.float)


import torch
import copy
def generate_combinations(n:int = 3, element_list:list = [-1,0,1]) -> list:
    '''
    生成一个列表组合，输入n为列表元素个数，element_list为列表元素内容
    '''
    result = []
    if n == 1:
        for i in element_list:
            result.append([i])
        return result
    elif n < 1:
        raise ValueError('n must be greater than 0')
    elif type(n) != int:
        raise TypeError('n must be int')
    else:
        for i in element_list:
            for comb in generate_combinations(n-1, element_list):
                if type(comb) != list:
                    result.append([i] + [comb])
                else:
                    result.append([i] + comb)
        return result

def E_message_process(temp_position:list, env_discretizied:np.ndarray, positons_tochoose_raw:list):
    '''
    处理temp_position处的E信息
    '''
    positons_tochoose = copy.deepcopy(positons_tochoose_raw)
    for i in range(len(positons_tochoose_raw)):
        positons_tochoose[i] = [temp_position[j] + positons_tochoose_raw[i][j] for j in range(3)]
    while temp_position in positons_tochoose:
        positons_tochoose.remove(temp_position)
    num_ways_out = 0.
    for i in range(len(positons_tochoose)):
        if abs(round(env_discretizied[positons_tochoose[i][0], positons_tochoose[i][1], positons_tochoose[i][2]])) < 1e-5:
            num_ways_out += 1
    return num_ways_out / len(positons_tochoose)

def Is_within_restricted_range(position:list, shape_env:tuple):
    '''
    判断位置postion是否在shape_env范围中
    '''
    for i in range(len(position)):
        if position[i] < 0 or position[i] >= shape_env[i]:
            return False
    return True

def EDBmessage_get(env_discretizied:np.ndarray, target_position:list, now_position:list, B_message:np.ndarray):
    '''
    获得当前位置的EDB信息，其中
    env_discretizied为离散化后的空间信息
    target_position为目标位置
    now_position为当前位置
    B_message为蚂蚁走过的位置信息
    '''
    if type(env_discretizied) == list:
        env_discretizied = np.array(env_discretizied)
    elif type(env_discretizied) == torch.Tensor:
        env_discretizied = env_discretizied.to('cpu').numpy()
        
    if type(env_discretizied) != np.ndarray:
        raise TypeError('env_discretizied must be np.ndarray')
    
    shape_env = env_discretizied.shape
    size = len(shape_env)

    positons_tochoose_raw = generate_combinations(n=3, element_list=[-1,0,1])
    positons_tochoose = copy.deepcopy(positons_tochoose_raw)
    for i in range(len(positons_tochoose_raw)):
        positons_tochoose[i] = [now_position[j] + positons_tochoose_raw[i][j] for j in range(3)]
    while now_position in positons_tochoose:
        positons_tochoose.remove(now_position)

    EDB_message = np.zeros((3, len(positons_tochoose)))
    for i in range(3):
        for j in range(len(positons_tochoose)):
            temp_position_0 = positons_tochoose[j]
            if not Is_within_restricted_range(temp_position_0, shape_env):
                continue
            if i == 0:
                # 处理E信息
                EDB_message[i, j] = E_message_process(temp_position_0, env_discretizied, positons_tochoose_raw)
            elif i == 1:
                # 处理D信息
                EDB_message[i, j] = math.sqrt(math.pow(temp_position_0[0] - target_position[0], 2) + 
                                            math.pow(temp_position_0[1] - target_position[1], 2))
            else:
                # B信息
                EDB_message[i, j] = B_message[temp_position_0[0], temp_position_0[1], temp_position_0[2]]
    # 对EDB_message每一行归一化处理
    for i in range(2):
        temp_max = np.max(EDB_message[i, :])
        temp_min = np.min(EDB_message[i, :])
        if round(temp_max - temp_min) < 1e-5:
            continue
        else:
            EDB_message[i, :] = (EDB_message[i, :] - temp_min) / (temp_max - temp_min)
    return EDB_message
