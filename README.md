# Design-of-Simple-Python-UAV-Env
Simple Python drone environment design, suitable for some simple drone task planning, reinforcement learning, etc

### 1.环境架构
利用python搭建无人机飞行环境，需要有高度、障碍物、地面状态、边界设置；可启动多架无人机；无人机具备电量、重量及动力学的基本属性。
### 2.路径规划任务的强化学习算法设计
路径规划包括避障类、搜索类任务两部分，利用到强化学习的方法策略。
### 3.算法与环境的交互实现
建立算法环境交互架构，利用不同的算法方法实现无人机在环境中的飞行路径规划等任务。

### **环境架构**
- 无人机抽象类设计
- 常规无人机设计
- 刚体障碍物设计
- 简单刚体环境设计

### **设计过程案例**
- <sup><a href="tasks_design/task0-points_flight.ipynb">无人机按照给定点飞行任务</a></sup>
- <sup><a href="tasks_design/task1-solid_simple_env.ipynb">无人机与刚体环境的简单交互</a></sup>
- <sup><a href="tasks_design/task2_SUAV_envDesign.ipynb">基于深度强化学习的单智能体决策过程方法研究————任务一：场景搭建</a></sup>
- <sup><a href="tasks_design/task3_SUAV_RL_Design.ipynb">基于深度强化学习的单智能体决策过程方法研究————任务二：强化学习过程20240122</a></sup>
