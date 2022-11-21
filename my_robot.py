import time

import numpy as np
import random

from math import cos, sin, pi, sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
random.seed(0)


class Robot:
    def __init__(self):
        self.target = np.asarray([30, 0,  100.6475143e+00])  # 机器臂要到达的目标点   可行点
        self.barrier = np.asarray([60, 0, 10])  # 障碍物中心坐标
        self.theta_min = 0.001
        self.theta_max = pi/2
        self.d_max = 150
        self.d_min = 0.1
        self.threshold = 5            # 机械臂停止阈值 末端点与目标点距离小于阈值时done=True
        self.rob_len = 50           # 机械臂的单段臂体长度
        self.theta1 = 0.001          # 弯曲1初始角度
        self.theta2 = 0.001          # 弯曲2初始角度
        self.theta3 = 0.001          # 弯曲3初始角度
        self.fai1 = 0               # 旋转1初始角度
        self.fai2 = 0               # 旋转2初始角度
        self.fai3 = 0               # 旋转3初始角度
        self.d = 0.1                  # 初始给进距离 给一点防止nan
        self.robt_obs = Robot.rob_kf(self)  # 初始化机械臂形状
        self.robt_end = self.robt_obs[-1][-1, :]    # 初始化机械臂末端点位置
        self.stp = 0            # 机械臂移动次数
        self.max_stp = 2e2      # 最大移动次数设置

    def rob_kf(self):
        r1 = self.rob_len / self.theta1
        r2 = self.rob_len / self.theta2
        r3 = self.rob_len / self.theta3
        fai2 = self.fai1 + self.fai2
        fai3 = fai2 + self.fai3
        # ----------------机器人在三维空间中弯曲段的形状点集-----------------------------------------
        point1 = np.array([])  # 移动段点集
        point2 = np.array([])  # 第一弯曲段点集合
        point3 = np.array([])  # 第二弯曲段点集合
        point4 = np.array([])  # 第三弯曲段点集合
        # ------------移动点集合50个---------------
        T0 = np.array([[0, 0, 1, -150 + self.d], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        T1 = np.array([[pow(cos(self.fai1), 2)*cos(self.theta1) + pow(sin(self.fai1), 2), cos(self.fai1)*sin(self.fai1)*cos(self.theta1)-cos(self.fai1)*sin(self.fai1), cos(self.fai1)*sin(self.theta1), r1*cos(self.fai1)*(1-cos(self.theta1))],
                [cos(self.fai1)*sin(self.fai1)*cos(self.theta1)-cos(self.fai1)*sin(self.fai1), pow(sin(self.fai1), 2)*cos(self.theta1)+pow(cos(self.fai1), 2), sin(self.fai1)*sin(self.theta1), r1*sin(self.fai1)*(1-cos(self.theta1))],
                [-cos(self.fai1)*sin(self.theta1), -sin(self.fai1)*sin(self.theta1), cos(self.theta1), r1*sin(self.theta1)],
                [0, 0, 0, 1]])

        T2 = np.array([[pow(cos(fai2), 2)*cos(self.theta2) + pow(sin(fai2), 2), cos(fai2)*sin(fai2)*cos(self.theta2)-cos(fai2)*sin(fai2), cos(fai2)*sin(self.theta2), r2*cos(fai2)*(1-cos(self.theta2))],
                [cos(fai2)*sin(fai2)*cos(self.theta2)-cos(fai2)*sin(fai2), pow(sin(fai2), 2)*cos(self.theta2)+pow(cos(fai2), 2), sin(fai2)*sin(self.theta2), r2*sin(fai2)*(1-cos(self.theta2))],
                [-cos(fai2)*sin(self.theta2), -sin(fai2)*sin(self.theta2), cos(self.theta2), r2*sin(self.theta2)],
                [0, 0, 0, 1]])
        # -------------------------------------------------
        mov_start = np.array([-150, 0, 0])
        move_end = T0[:3, -1]

        for i0 in np.linspace(mov_start, move_end, 50):
            point1 = np.append(point1, i0)
        point1 = point1.reshape(50, 3)
        # ----------弯曲1点集合--------------
        wan1_start = T0
        for i1 in np.linspace(0.0001, self.theta1, 50):
            t1 = np.array([[pow(cos(self.fai1), 2) * cos(i1) + pow(sin(self.fai1), 2),
                            cos(self.fai1) * sin(self.fai1) * cos(i1) - cos(self.fai1) * sin(self.fai1),
                            cos(self.fai1) * sin(i1), r1 * cos(self.fai1) * (1 - cos(i1))],
                           [cos(self.fai1) * sin(self.fai1) * cos(i1) - cos(self.fai1) * sin(self.fai1),
                            pow(sin(self.fai1), 2) * cos(i1) + pow(cos(self.fai1), 2), sin(self.fai1) * sin(i1),
                            r1 * sin(self.fai1) * (1 - cos(i1))],
                           [-cos(self.fai1) * sin(i1), -sin(self.fai1) * sin(i1), cos(i1), r1 * sin(i1)],
                           [0, 0, 0, 1]])
            temp1 = np.dot(wan1_start, t1)
            temp1 = temp1[:3, -1]
            point2 = np.append(point2, temp1)
        point2 = point2.reshape(50, 3)
        # ----------弯曲2点集合--------------
        wan2_start = np.dot(T0, T1)
        for i2 in np.linspace(0.0001, self.theta2, 50):
            t2 = np.array([[pow(cos(fai2), 2) * cos(i2) + pow(sin(fai2), 2),
                            cos(fai2) * sin(fai2) * cos(i2) - cos(fai2) * sin(fai2), cos(fai2) * sin(i2),
                            r2 * cos(fai2) * (1 - cos(i2))],
                           [cos(fai2) * sin(fai2) * cos(i2) - cos(fai2) * sin(fai2),
                            pow(sin(fai2), 2) * cos(i2) + pow(cos(fai2), 2), sin(fai2) * sin(i2),
                            r2 * sin(fai2) * (1 - cos(i2))],
                           [-cos(fai2) * sin(i2), -sin(fai2) * sin(i2), cos(i2), r2 * sin(i2)],
                           [0, 0, 0, 1]])
            temp2 = np.dot(wan2_start, t2)
            temp2 = temp2[:3, -1]
            point3 = np.append(point3, temp2)
        point3 = point3.reshape(50, 3)
        # ----------弯曲3点集合--------------
        wan3_start = np.dot(T0, T1)
        wan3_start = np.dot(wan3_start, T2)
        for i3 in np.linspace(0.0001, self.theta3, 50):
            t3 = np.array([[pow(cos(fai3), 2) * cos(i3) + pow(sin(fai3), 2),
                            cos(fai3) * sin(fai3) * cos(i3) - cos(fai3) * sin(fai3), cos(fai3) * sin(i3),
                            r3 * cos(fai3) * (1 - cos(i3))],
                           [cos(fai3) * sin(fai3) * cos(i3) - cos(fai3) * sin(fai3),
                            pow(sin(fai3), 2) * cos(i3) + pow(cos(fai3), 2), sin(fai3) * sin(i3),
                            r3 * sin(fai3) * (1 - cos(i3))],
                           [-cos(fai3) * sin(i3), -sin(fai3) * sin(i3), cos(i3), r3 * sin(i3)],
                           [0, 0, 0, 1]])
            temp3 = np.dot(wan3_start, t3)
            temp3 = temp3[:3, -1]
            point4 = np.append(point4, temp3)
        point4 = point4.reshape(50, 3)
        # ==========================ok===================================
        self.robt_obs = np.array([point1, point2, point3, point4],dtype=np.float32)
        self.robt_end = self.robt_obs[-1][-1, :]
        return self.robt_obs

    # ========绘制机器人一帧图像============
    def draw(self):
        plt.clf()
        plt.ion()
        ax = plt.axes(projection='3d')
        ax.view_init(ax.elev, ax.azim-1)  # 设置观察角度
        ax.plot3D(self.robt_obs[0][:, 0], self.robt_obs[0][:, 1], self.robt_obs[0][:, 2], c='k', linewidth=4)
        ax.plot3D(self.robt_obs[1][:, 0], self.robt_obs[1][:, 1], self.robt_obs[1][:, 2], c='r', linewidth=4)
        ax.plot3D(self.robt_obs[2][:, 0], self.robt_obs[2][:, 1], self.robt_obs[2][:, 2], c='g', linewidth=4)
        ax.plot3D(self.robt_obs[3][:, 0], self.robt_obs[3][:, 1], self.robt_obs[3][:, 2], c='b', linewidth=4)
        ax.scatter(self.target[0], self.target[1], self.target[2], c='r', linewidth=3.0)
        # --------------------------------画红色长方体框线---------------------------------
        cube_x, cube_y, cube_z = 0, -100, -100    # 红色框线起始坐标
        cube_dx, cube_dy, cube_dz = 150, 200, 200   # 红色框线各个边长
        cube_xx = [cube_x, cube_x, cube_x + cube_dx, cube_x + cube_dx, cube_x]
        cube_yy = [cube_y, cube_y + cube_dy, cube_y + cube_dy, cube_y, cube_y]
        kwargs = {'alpha': 1, 'color': 'red'}
        ax.plot3D(cube_xx, cube_yy, [cube_z]*5, **kwargs)
        ax.plot3D(cube_xx, cube_yy, [cube_z+cube_dz]*5, **kwargs)
        ax.plot3D([cube_x, cube_x], [cube_y, cube_y], [cube_z, cube_z+cube_dz], **kwargs)
        ax.plot3D([cube_x, cube_x], [cube_y+cube_dy, cube_y+cube_dy], [cube_z, cube_z+cube_dz], **kwargs)
        ax.plot3D([cube_x+cube_dx, cube_x+cube_dx], [cube_y+cube_dy, cube_y+cube_dy], [cube_z, cube_z+cube_dz], **kwargs)
        ax.plot3D([cube_x+cube_dx, cube_x+cube_dx], [cube_y, cube_y], [cube_z, cube_z+cube_dz], **kwargs)
        # ----------正方体障碍物绘制-------------------------------
        cube_barrier_x, cube_barrier_y, cube_barrier_z = 50, -10, 0  # 正方体障碍物的起始坐标  正方体中心（60，0，10） 外切球半径 ：10根3
        cube_barrier_dx, cube_barrier_dy, cube_barrier_dz = 20, 20, 20  # 正方体障碍物各个边长
        xx = np.linspace(cube_barrier_x, cube_barrier_x + cube_barrier_dx, 2)
        yy = np.linspace(cube_barrier_y, cube_barrier_y + cube_barrier_dy, 2)
        zz = np.linspace(cube_barrier_z, cube_barrier_z + cube_barrier_dz, 2)
        xx2, yy2 = np.meshgrid(xx, yy)
        ax.plot_surface(xx2, yy2, np.full_like(xx2, cube_barrier_z))
        ax.plot_surface(xx2, yy2, np.full_like(xx2, cube_barrier_z + cube_barrier_dz))
        yy2, zz2 = np.meshgrid(yy, zz)
        ax.plot_surface(np.full_like(yy2, cube_barrier_x), yy2, zz2)
        ax.plot_surface(np.full_like(yy2, cube_barrier_x + cube_barrier_dx), yy2, zz2)
        xx2, zz2 = np.meshgrid(xx, zz)
        ax.plot_surface(xx2, np.full_like(yy2, cube_barrier_y), zz2)
        ax.plot_surface(xx2, np.full_like(yy2, cube_barrier_y + cube_barrier_dy), zz2)
        # ----------设置状态空间大小-----------------
        ax.set_xlim(-150, 150)
        ax.set_xlabel('x')
        ax.set_ylim(-150, 150)
        ax.set_ylabel('y')
        ax.set_zlim(-150, 150)
        ax.set_zlabel('z')
        plt.ioff()






if __name__ == '__main__':
    robot = Robot()
    # robot.d = 150
    # robot.theta1 = pi/3.5
    # robot.theta2 = pi/3
    # robot.theta3 = pi/3
    # robot.fai1 = pi
    # robot.fai2 = pi
    # robot.fai3 = 0
    temp = robot.rob_kf()
    print(type(temp))
    print("末端点", temp[-1][-1, :])
    print(temp.shape)
    robot.draw()
    plt.pause(100)
