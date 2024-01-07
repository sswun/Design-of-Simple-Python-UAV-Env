# 环境类设计
import math
import numpy as np
import copy
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

range = np.arange

# 实用函数设计
# 判断某点是否在包络体中
def is_point_inside_hull(point, hull):
    """
    判断点是否在凸包中
    """
    # 使用Delaunay三角剖分来判断点是否在凸包中
    tri = Delaunay(hull.points)
    simplex = tri.find_simplex(point)
    return simplex >= 0

def point_is_in_solid(point, feature_information):
    '''
    判断一个点是否在物体内部
    '''
    point = np.array(point)
    if feature_information["type"] == "sphere":
        return np.linalg.norm(point - np.array(feature_information["feature_points"][0])) \
            <= feature_information["feature_points"][1][0]
    
    elif feature_information["type"] == "cube":
        if point[0] >= feature_information["feature_points"][0][0] - \
            float(feature_information["feature_points"][1][0]) / 2 and \
            point[0] <= feature_information["feature_points"][0][0] + \
            float(feature_information["feature_points"][1][0]) / 2 and \
            point[1] >= feature_information["feature_points"][0][1] - \
            float(feature_information["feature_points"][1][0]) / 2 and \
            point[1] <= feature_information["feature_points"][0][1] + \
            float(feature_information["feature_points"][1][0]) / 2 and \
            point[2] >= feature_information["feature_points"][0][2] - \
            float(feature_information["feature_points"][1][0]) / 2 and \
            point[2] <= feature_information["feature_points"][0][2] + \
            float(feature_information["feature_points"][1][0]) / 2:
            return True
        else:
            return False
    
    elif feature_information["type"] == "spheroid":
        if (point[0] - feature_information["feature_points"][0][0]) ** 2 / feature_information["feature_points"][1][0] ** 2 + \
            (point[1] - feature_information["feature_points"][0][1]) ** 2 / feature_information["feature_points"][1][1] ** 2 + \
            (point[2] - feature_information["feature_points"][0][2]) ** 2 / feature_information["feature_points"][1][2] ** 2 <= 1:
            return True
        else:
            return False
    
    elif feature_information["type"] == "cuboid":
        if point[0] >= feature_information["feature_points"][0][0] - \
            float(feature_information["feature_points"][1][0]) / 2 and \
            point[0] <= feature_information["feature_points"][0][0] + \
            float(feature_information["feature_points"][1][0]) / 2 and \
            point[1] >= feature_information["feature_points"][0][1] - \
            float(feature_information["feature_points"][1][1]) / 2 and \
            point[1] <= feature_information["feature_points"][0][1] + \
            float(feature_information["feature_points"][1][1]) / 2 and \
            point[2] >= feature_information["feature_points"][0][2] - \
            float(feature_information["feature_points"][1][2]) / 2 and \
            point[2] <= feature_information["feature_points"][0][2] + \
            float(feature_information["feature_points"][1][2]) / 2:
            return True
        else:
            return False
    
    elif feature_information["type"] == "lattice":
        # 构造凸包
        hull = ConvexHull(feature_information["feature_points"])
        # 判断测试点是否在凸包中
        if is_point_inside_hull(point, hull):
            return True
        else:
            return False
    else:
        return False

def points_inSolid_generate(feature_information, dx=0.1):
    '''
    生成物体中的散点，便于绘3D图，dx为散点间隔
    '''
    feature_points = copy.deepcopy(feature_information["feature_points"])
    x_points = []
    y_points = []
    z_points = []
    if feature_information["type"] == "sphere":
        for x in range(feature_points[0][0] - feature_points[1][0],
                      feature_points[0][0] + feature_points[1][0] + dx,
                      dx):
            for y in range(feature_points[0][1] - feature_points[1][0],
                            feature_points[0][1] + feature_points[1][0] + dx,
                            dx):
                for z in range(feature_points[0][2] - feature_points[1][0],
                                 feature_points[0][2] + feature_points[1][0] + dx,
                                 dx):
                    if point_is_in_solid([x,y,z], feature_information):
                        x_points.append(x)
                        y_points.append(y)
                        z_points.append(z)
        return [x_points, y_points, z_points]
    elif feature_information["type"] == "cube":
        for x in range(feature_points[0][0] - float(feature_points[1][0]) / 2,
                      feature_points[0][0] + float(feature_points[1][0]) / 2 + dx,
                      dx):
            for y in range(feature_points[0][1] - float(feature_points[1][0]) / 2,
                            feature_points[0][1] + float(feature_points[1][0]) / 2 + dx,
                            dx):
                for z in range(feature_points[0][2] - float(feature_points[1][0]) / 2,
                                 feature_points[0][2] + float(feature_points[1][0]) / 2 + dx,
                                 dx):
                    if point_is_in_solid([x,y,z], feature_information):
                        x_points.append(x)
                        y_points.append(y)
                        z_points.append(z)
        return [x_points, y_points, z_points]
    elif feature_information["type"] == "spheroid":
        for x in range(feature_points[0][0] - float(feature_points[1][0]),
                      feature_points[0][0] + float(feature_points[1][0]) + dx,
                      dx):
            for y in range(feature_points[0][1] - float(feature_points[1][1]),
                            feature_points[0][1] + float(feature_points[1][1]) + dx,
                            dx):
                for z in range(feature_points[0][2] - float(feature_points[1][2]),
                                 feature_points[0][2] + float(feature_points[1][2]) + dx,
                                 dx):
                    if point_is_in_solid([x,y,z], feature_information):
                        x_points.append(x)
                        y_points.append(y)
                        z_points.append(z)
        return [x_points, y_points, z_points]
    elif feature_information["type"] == "cuboid":
        for x in range(feature_points[0][0] - float(feature_points[1][0]) / 2,
                      feature_points[0][0] + float(feature_points[1][0]) / 2 + dx,
                      dx):
            for y in range(feature_points[0][1] - float(feature_points[1][1]) / 2,
                            feature_points[0][1] + float(feature_points[1][1]) / 2 + dx,
                            dx):
                for z in range(feature_points[0][2] - float(feature_points[1][2]) / 2,
                                 feature_points[0][2] + float(feature_points[1][2]) / 2 + dx,
                                 dx):
                    if point_is_in_solid([x,y,z], feature_information):
                        x_points.append(x)
                        y_points.append(y)
                        z_points.append(z)
        return [x_points, y_points, z_points]
    elif feature_information["type"] == "lattice":
        for i in range(len(feature_points)):
            x_points.append(feature_points[i][0])
            y_points.append(feature_points[i][1])
            z_points.append(feature_points[i][2])
        return [x_points, y_points, z_points]
    else:
        raise Exception("Unknown solid type")
        

class solid_base:
    '''
    本类为固体类，表示占据一定体积的刚体物体
    feature_information为物体类型，为一字典或列表，
    可以为某些标准几何体("sphere"球体， "cube"正方体，"spheroid"椭球体，"cuboid"长方体)
    如：
    {"type":"sphere","feature_points":[[x, y, z],[r]]}表示半径为r，中心位置在[x,y,z]处的球体；
    {"type":"cube","feature_points":[[x, y, z],[d]]}表示边长为d，中心位置在[x,y,z]处的正方体；
    {"type":"spheroid","feature_points":[[x, y, z],[a, b, c]]}表示特征量为[a,b,c]，中心位置在[x,y,z]处的椭球体；
    {"type":"cuboid","feature_points":[[x, y, z],[a, b, c]]}表示长宽高为[a,b,c]，中心位置在[x,y,z]处的长方体
    也可以为某点阵("lattice")
    如：
    {"type":"lattice","feature_points":[[x, y, z],...]}，此时将以feature_points中点列，构建一个包络体
    注意：此时将使用scipy.spatial中函数构造凸包
    '''
    def __init__(self, feature_information:dict = None):
        if feature_information is None:
            self.feature_information = {"type":"cube","feature_points":[[0, 0, 0],[1]]}
        elif type(feature_information) == dict:
            self.feature_information = feature_information
        else:
            raise TypeError("feature_information must be a dict")
        self.points = None
        self.fig = None
    
    def point_in_solid(self, point:list):
        '''
        判断给定点是否固体中
        返回值为bool
        '''
        if point_is_in_solid(point, self.feature_information):
            return True
        else:
            return False      
    
    def points_in_solid(self, points:list):
        '''
        判断给定点是否固体中
        返回值为bool列表，与points一一对应
        '''
        return [self.point_in_solid(point) for point in points]
    
    def points_Solid_generate(self):
        '''
        生成一个生成器，用于生成在固体中的点
        '''
        self.points = points_inSolid_generate(self.feature_information)
    
    def set_fig(self):
        '''
        设置图窗
        '''
        if self.fig is None:
            self.fig = plt.figure()
            self.ax1 = self.fig.add_subplot(1,1,1,projection = "3d")
        
    def show_Solid(self, ax1 = None):
        '''
        显示固体
        '''
        if self.points is None:
            self.points_Solid_generate()
        x_points = self.points[0]
        y_points = self.points[1]
        z_points = self.points[2]
        if ax1 is None:
            self.set_fig()
            self.ax1.scatter(x_points, y_points, z_points, c = "red", marker = "."
                        ,label = self.feature_information["type"])
            self.ax1.set_xlabel("x")
            self.ax1.set_ylabel("y")
        else:
            ax1.scatter(x_points, y_points, z_points, c = "red", marker = "."
                        ,label = self.feature_information["type"])
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
    
    def add_scatter(self, x_points, y_points, z_points, c = "red", marker = ".", ax1 = None):
        '''
        添加散点图
        '''
        if ax1 is None:
            self.set_fig()
            self.ax1.scatter(x_points, y_points, z_points, c = c, marker = marker)
        else:
            ax1.scatter(x_points, y_points, z_points, c = c, marker = marker)
    
    def add_scatter_solid(self, Obj_solid_base, c = "red", marker = ".", ax1 = None):
        '''
        添加散点图，将Obj_solid_base物体导入散点图中
        '''
        if Obj_solid_base.points is None:
            Obj_solid_base.points_Solid_generate()
        x_points = Obj_solid_base.points[0]
        y_points = Obj_solid_base.points[1]
        z_points = Obj_solid_base.points[2]
        if ax1 is None:
            self.add_scatter(x_points, y_points, z_points, c = c, marker = marker)
        else:
            ax1.scatter(x_points, y_points, z_points, c = c, marker = marker)

class solid_env:
    '''
    创建一个刚体环境容器，便于与无人机进行交互
    '''
    def __init__(self, solid_information:list):
        self.solid_list = []
        for solid_info in solid_information:
            self.solid_list.append(solid_base(solid_info))
        self.fig = None
        self.ax1 = None
    
    def set_fig(self):
        '''
        设置画布
        '''
        if self.fig is None:
            self.fig = plt.figure()
            self.ax1 = self.fig.add_subplot(111, projection='3d')
            self.ax1.set_xlabel("x")
            self.ax1.set_ylabel("y")
    
    def add_scatter(self, x_points, y_points, z_points, c = "red", marker = ".", ax1 = None):
        '''
        添加散点图
        '''
        if ax1 is None:
            self.set_fig()
            self.ax1.scatter(x_points, y_points, z_points, c = c, marker = marker)
        else:
            ax1.scatter(x_points, y_points, z_points, c = c, marker = marker)
    
    def add_scatter_solid(self, Obj_solid_base, c = "red", marker = ".", ax1 = None):
        '''
        添加散点图，将Obj_solid_base物体导入散点图中
        '''
        if Obj_solid_base.points is None:
            Obj_solid_base.points_Solid_generate()
        x_points = Obj_solid_base.points[0]
        y_points = Obj_solid_base.points[1]
        z_points = Obj_solid_base.points[2]
        if ax1 is None:
            self.add_scatter(x_points, y_points, z_points, c = c, marker = marker)
        else:
            ax1.scatter(x_points, y_points, z_points, c = c, marker = marker)
    
    def point_in_solid_env(self, point:list, isindex:bool = False):
        '''
        判断给定点是否在环境中
        若在，返回值为在的物体对应的index
        '''
        for solid in self.solid_list:
            if solid.point_in_solid(point):
                if isindex:
                    return self.solid_list.index(solid)
                else:
                    return True
        return False
    
    def points_in_solid_env(self, points:list):
        '''
        判断给定点列中每个点是否在环境中
        若在，返回值为在的物体对应的index
        '''
        return [self.point_in_solid_env(point) for point in points]
    
    def show_solid_env(self, ax1 = None):
        '''
        展示刚体环境
        '''
        self.set_fig()
        for solid in self.solid_list:
            self.add_scatter_solid(solid, ax1 = ax1)

    
