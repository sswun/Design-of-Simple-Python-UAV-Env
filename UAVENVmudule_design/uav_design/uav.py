# 无人机类设计
import math
import numpy as np
import copy
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from IPython import display
import matplotlib
import time

import sys
sys.path.append("..")
from ..env_design import *

# 常用函数设计
def distance2(point1:list = [0., 0., 0.], point2:list = [1., 1., 1.]) -> float:
    '''
    计算两个向量点之间的欧式距离
    '''
    if type(point1) == list and type(point2) == list:
        if len(point1) == len(point2):
            return math.sqrt(sum([(point1[i] - point2[i]) ** 2 for i in range(len(point1))]))
        else:
            raise Exception("The length of two points are not equal!")
    else:
        try:
            return math.sqrt(sum([(point1[i] - point2[i]) ** 2 for i in range(len(point1))]))
        except:
            raise Exception("The input of function distance2 should be a 1D list!")

def is_samepositionlist_same(point1, point2) -> bool:
    '''
    判断两个向量点是否相同，包括位置
    '''
    if len(point1) == len(point2):
        for i in range(len(point1)):
            if point1[i] != point2[i]:
                return False
        return True
    return False

def scatter_3d(x:list, y:list, z:list, ax1 = None):
    # 散点图
    if ax1 is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # s：marker标记的大小
        # c: 颜色  可为单个，可为序列
        # depthshade: 是否为散点标记着色以呈现深度外观。对 scatter() 的每次调用都将独立执行其深度着色。
        # marker：样式
        ax.scatter(xs=x, ys=y, zs=z, zdir='z', s=30, c="g", depthshade=True, cmap="jet", marker=".")
        plt.show()
    else:
        ax1.scatter(xs=x, ys=y, zs=z, zdir='z', s=30, c="g", depthshade=True, cmap="jet", marker=".")
        plt.show()

def scatter_dynamic_3d(x:list, y:list, z:list, dt=0.2, ax1 = None):
    # 创建一个图形和轴
    if ax1 is None:
        for i in range(len(x)):
            fig = plt.figure(1)
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim(min(x), max(x))
            ax.set_ylim(min(y), max(y))
            ax.set_zlim(min(z), max(z))
            xx = x[i]
            yy = y[i]
            zz = z[i]
            ax.scatter(xs=xx, ys=yy, zs=zz, zdir='z', s=30, c="g", depthshade=True, cmap="jet", marker=".")
            ax.set_title('Points Traveled')
            time.sleep(dt)
            plt.show()
            plt.close()
            display.clear_output(wait=True)
    else:
        for i in range(len(x)):
            ax1.scatter(xs=x[i], ys=y[i], zs=z[i], zdir='z', s=30, c="g", depthshade=True, cmap="jet", marker=".")
            ax1.set_title('Points Traveled')
            time.sleep(dt)
            plt.show()
            plt.close()
            display.clear_output(wait=True)

def transpose_list(matrix):
    # 使用 * 操作符和 zip 函数将内层列表展开，然后进行转置
    transposed = [list(row) for row in zip(*matrix)]
    return transposed

class UAV:
    '''
    本设计无人机模型参考并简化自：https://zhuanlan.zhihu.com/p/349306054
    无人机类的设计：
    无人机类属性：
        name：无人机名称；
        type：类型；
        max_speed：最大速度；
        max_range：最大搜索范围；
        max_altitude：最大高度；
        max_power：最大电源；
        max_load：最大负载重量
    '''
    def __init__(self, name:str = '001', 
                        type:int = 0, 
                        max_speed:float = 10., 
                        max_range:float = 100., 
                        max_altitude:float = 1000., 
                        max_power:float = 1e4, 
                        max_load:float = 1e3):
        '''
        初始化属性设置：
        无人机类属性：
            name：无人机名称；
            type：类型；
            max_speed：最大速度；
            max_range：最大搜索范围；
            max_altitude：最大高度；
            max_power：最大电源；
            max_load：最大负载重量
        '''
        self.name = name
        self.type = type
        self.max_speed = max_speed
        self.max_range = max_range
        self.max_altitude = max_altitude
        self.max_power = max_power
        self.max_load = max_load
        self.properties = {'name':self.name,
                           'type':self.type,
                           'speed':0.,
                           'range':self.max_range,
                           'altitude':0.,
                           'power':self.max_power,
                           'load':0.
                           } 
    
    def reinit_max_properties(self, name:str = None,
                        type:int = None, 
                        max_speed:float = None, 
                        max_range:float = None, 
                        max_altitude:float = None, 
                        max_power:float = None, 
                        max_load:float = None):
        '''
        重新初始化属性设置：
        无人机类属性：
            name：无人机名称；
            type：类型；
            max_speed：最大速度；
            max_range：最大搜索范围；
            max_altitude：最大高度；
            max_power：最大电源；
            max_load：最大负载重量
        '''
        if name is not None:
            self.name = name
        if type is not None:
            self.type = type
        if max_speed is not None:
            self.max_speed = max_speed
        if max_range is not None:
            self.max_range = max_range
        if max_altitude is not None:
            self.max_altitude = max_altitude
        if max_power is not None:
            self.max_power = max_power
        if max_load is not None:
            self.max_load = max_load
    
    def reset_Current_Properties(self, name:str = None,
                        type:int = None, 
                        speed:float = None, 
                        range:float = None, 
                        altitude:float = None, 
                        power:float = None, 
                        load:float = None):
        '''
        改变当前无人机的属性设置：
        无人机类属性：
            name：无人机名称；
            type：类型；
            speed：速度；
            range：搜索范围；
            altitude：高度；
            power：电量；
            load：负载
        '''
        self.properties = {'name':self.name if name is None else name,
                           'type':self.type if type is None else type,
                           'speed':0. if speed is None else speed,
                           'range':self.max_range if range is None else range,
                           'altitude':0. if altitude is None else altitude,
                           'power':self.max_power if power is None else power,
                           'load':0. if load is None else load
                           } 
        
    def get_info_max(self):
        print("无人机名称：", self.name)
        print("无人机类型：", self.type)
        print("无人机最大速度：", self.max_speed)
        print("无人机最大射程：", self.max_range)
        print("无人机最大飞行高度：", self.max_altitude)
        print("无人机最大电力容量：", self.max_power)
        print("无人机最大负载能力：", self.max_load)

    def get_info_json_max(self):
        return {
            "name": self.name,
            "type": self.type,
            "max_speed": self.max_speed,
            "max_range": self.max_range,
            "max_altitude": self.max_altitude,
            "max_power": self.max_power,
            "max_load": self.max_load,
        }
    def get_Current_Properties(self):
        return self.properties
    
class Conventional_UAV(UAV):
    '''
    常规无人机类的设计——继承自UAV：
    本设计将无人机抽象为理想飞行器，无人机电量消耗只与垂直升降、平行飞行有关，且为距离的一次函数
    无人机类属性：
        name：无人机名称；
        type：类型；
        max_speed：最大速度；
        max_range：最大搜索范围；
        max_altitude：最大高度；
        max_power：最大电源；
        max_load：最大负载重量
        k_up：上升电量消耗
        k_down：下降电量消耗
        k_around：平行飞电量消耗
        dt：模拟飞行时时间微元
    '''
    def __init__(self, name:str = '001', 
                        type:int = 0, 
                        max_speed:float = 10., 
                        max_range:float = 100., 
                        max_altitude:float = 1000., 
                        max_power:float = 1e4, 
                        max_load:float = 1e3,
                        k_up:float = 0.01,
                        k_down:float = 0.001,
                        k_around:float = 0.005,
                        dt:float = 0.01,
                        weight:float = 1):
        super().__init__(name, type, max_speed, max_range, max_altitude, max_power, max_load)
        self.properties['k_up'] = k_up
        self.properties['k_down'] = k_down
        self.properties['k_around'] = k_around
        self.properties['dt'] = dt
        self.properties['weight'] = weight
        # 记录无人机旅行对应数据变化
        self.points_traveled = {'time':[0],
                                'position':[[0, 0, 0]],
                                'power':[self.properties['power']],
                                'load':[0],
                                'speed':[0]}
    
    def reset_UAV(self, init_position:list = None):
        self.points_traveled = {'time':[0],
                                'position':[[0, 0, 0]] if init_position is None else init_position,
                                'power':[self.max_power],
                                'load':[0],
                                'speed':[0]}
        self.properties["speed"] = 0.
        self.properties["altitude"] = 0.
        self.properties["power"] = self.max_power
        self.properties["load"] = 0.
    
    def is_front_obstacle(self, obstacle:object = None) -> bool:
        '''
        基本动作函数：
        判断飞行前方是否有障碍物
        '''
        if obstacle is None:
            return False
        temp_point = copy.deepcopy(self.points_traveled["position"][-1])
        if type(obstacle) == solid_base:
            return obstacle.point_in_solid(temp_point)
        elif type(obstacle) == solid_env:
            return obstacle.point_in_solid_env(temp_point)
        else:
            raise TypeError("The type of obstacle is not supported!")
    
    def scatter_points_traveled(self, isdynamic:bool = True, dt:float = None, obstacle:object = None):
        '''
        绘制无人机飞行过的轨迹
        '''
        positions = transpose_list(self.points_traveled['position'])
        if obstacle is None:
            if isdynamic:
                if dt is None:
                    scatter_dynamic_3d(positions[0], positions[1], positions[2], dt=self.properties['dt'])
                else:
                    scatter_dynamic_3d(positions[0], positions[1], positions[2], dt=dt)
            else:
                scatter_3d(positions[0], positions[1], positions[2])
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            if type(obstacle) == solid_base:
                obstacle.show_Solid(ax1=self.ax)
            elif type(obstacle) == solid_env:
                obstacle.show_solid_env(ax1=self.ax)
                if isdynamic:
                    if dt is None:
                        scatter_dynamic_3d(positions[0], positions[1], positions[2], dt=self.properties['dt'],
                                                   ax1=self.ax)
                    else:
                        scatter_dynamic_3d(positions[0], positions[1], positions[2], dt=dt, ax1=self.ax)
                else:
                    scatter_3d(positions[0], positions[1], positions[2], ax1=self.ax)
    
    def power_consume(self, direction):
        # 计算从当前点飞行一个向量direction，总的电量消耗
        consume_all_pre = 0
        if direction[2] < 0:
            consume_all_pre += self.properties['k_down']*abs(direction[2])* \
            (self.properties['weight'] + self.properties['load'])
        else:
            consume_all_pre += self.properties['k_up']*abs(direction[2])* \
                (self.properties['weight'] + self.properties['load'])
        consume_all_pre += self.properties['k_around']*np.linalg.norm(direction[:2])* \
            (self.properties['weight'] + self.properties['load'])
        
        if consume_all_pre > self.properties['power']:
            raise ValueError('power is not enough to finish the travel')
        return consume_all_pre
    
    def travel_as_direction(self, direction:list = None, speed:float = None, time:float = None, obstacle:object = None):
        '''
        基本动作函数：
        按照指定的方向进行飞行，飞行一段时间time，或者dt，按照速度speed飞行
        '''
        dist_norm = np.linalg.norm(np.array(direction))
        if dist_norm < 1e-5:
            raise ValueError('direction is zero!')
        # 方向向量归一化
        direction_norm = direction / dist_norm
        if time is None:
            time = self.properties['dt']
        
        # 速度范围限制
        if speed is None:
            speed = self.max_speed
        elif speed > self.max_speed:
            Warning('speed is too large, set to max_speed')
            speed = self.max_speed
        elif speed <= 0:
            raise ValueError('speed must be positive')
        
        # 电量限制
        if self.properties['power'] <= 0:
            raise ValueError('power must be positive')
        
        # 开始模拟
        remain_time = time
        current_position = copy.deepcopy(self.points_traveled['position'][-1])
        while remain_time>0:
            
            # 迭代计算
            remain_time -= self.properties['dt']
            self.points_traveled['time'].append(self.points_traveled['time'][-1] + self.properties['dt'])
            travel_temp_d = self.properties['dt'] * speed * direction_norm
            current_position += travel_temp_d
            dt_consume = self.power_consume(travel_temp_d)
            self.points_traveled['power'].append(self.points_traveled['power'][-1] - dt_consume)
            self.points_traveled['load'].append(self.properties['load'])
            self.points_traveled['speed'].append(speed)
            self.points_traveled['position'].append(copy.deepcopy(current_position).tolist())
            
            # 更新当前状态点
            self.properties['power'] = self.points_traveled['power'][-1]
            self.properties['load'] = self.points_traveled['load'][-1]
            self.properties['altitude'] = self.points_traveled['position'][-1][2]
            self.properties['speed'] = self.points_traveled['speed'][-1]
            # 判断前方是否有障碍物
            if self.is_front_obstacle(obstacle=obstacle):
                print('obstacle crashed, break')
                break
            # 判断是否超过最大高度限制
            if self.properties['altitude'] > self.max_altitude:
                print('exceed max altitude, break')
                break
        
    
    def travel_point2point(self, start_point:list = None, end_point:list = [100, 100, 10],
                           speed:float = None, obstacle:object = None):
        '''
        基本动作函数：
        从一点飞行到另一点
        '''
        if start_point is None:
            start_point = self.points_traveled['position'][-1]
        direction = np.array(end_point) - np.array(start_point) 
        # 判断海拔是否超出最大限度
        temp_altitude = max([start_point[2], end_point[2]])
        if temp_altitude > self.max_altitude:
            raise ValueError('The altitude is too high')
        
        # 得到起点终点距离
        distance = np.linalg.norm(direction)
        if is_samepositionlist_same(start_point, end_point) or distance <= 1e-5:
            raise ValueError('start_point and end_point is same or nearly same')
        
        # 方向向量归一化
        direction_norm = direction / distance
        
        # 速度范围限制
        if speed is None:
            speed = self.max_speed
        elif speed > self.max_speed:
            Warning('speed is too large, set to max_speed')
            speed = self.max_speed
        elif speed <= 0:
            raise ValueError('speed must be positive')
        
        # 电量限制
        if self.properties['power'] <= 0:
            raise ValueError('power must be positive')
        
        # 计算总的电量消耗
        consume_all_pre = self.power_consume(direction=direction)
        
        # 计算飞行时间与单位时间消耗的电量
        time_fly = distance / speed
        dt_consume = consume_all_pre / time_fly * self.properties['dt']
        
        # 开始模拟
        remain_dis = distance
        current_position = copy.deepcopy(start_point)
        while remain_dis>0:
            
            # 迭代计算
            self.points_traveled['time'].append(self.points_traveled['time'][-1] + self.properties['dt'])
            travel_temp_d = self.properties['dt'] * speed * direction_norm
            current_position += travel_temp_d
            self.points_traveled['power'].append(self.points_traveled['power'][-1] - dt_consume)
            self.points_traveled['load'].append(self.properties['load'])
            self.points_traveled['speed'].append(speed)
            self.points_traveled['position'].append(copy.deepcopy(current_position).tolist())
            remain_dis -= np.linalg.norm(travel_temp_d)
            
            # 更新当前状态点
            self.properties['power'] = self.points_traveled['power'][-1]
            self.properties['load'] = self.points_traveled['load'][-1]
            self.properties['altitude'] = self.points_traveled['position'][-1][2]
            self.properties['speed'] = self.points_traveled['speed'][-1]
            # 判断前方是否有障碍物
            if self.is_front_obstacle(obstacle=obstacle):
                print('obstacle crashed!, break')
                break
            # 判断是否超过最大高度限制
            if self.properties['altitude'] > self.max_altitude:
                print('exceed max altitude, break')
                break
    
    def travel_as_points(self, points:list, speed:float | list, obstacle:object = None):
        '''
        无人机从当前位置出发，按照给定的点列points前进
        speed要么为float，要么为list，若为speed，则无人机在整个旅程中按照speed设置的速度大小前进，速度不变；
        若为list，则其size为len(points)，在前往每一个点时按照对应的速度前进
        '''
        for i in range(len(points)):
            if type(speed)==float:
                self.travel_point2point(end_point=points[i], speed=speed, obstacle=obstacle)
            else:
                self.travel_point2point(end_point=points[i], speed=speed[i], obstacle=obstacle)

            