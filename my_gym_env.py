import random

import gym
import matplotlib.pyplot as plt
from gym import spaces
import stable_baselines3
import math
from stable_baselines3.common.env_checker import check_env
from math import pi, sqrt, exp, acos
import numpy as np
from my_robot import Robot


class CustomEnv(gym.Env, Robot):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(CustomEnv, self).__init__()
        super(Robot, self).__init__()
        # Define action and observation space
        # 设置机械臂的动作空间，动作为连续动作：box类型
        self.action_space = spaces.Box(low=np.asarray([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -5]),
                                        high=np.asarray([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 5]), shape=(7, ), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):  一段是50设置一段25个点
        self.observation_space = spaces.Box(low=-150, high=150,
                                            shape=(4, 50, 3), dtype=np.float32)

    def reset(self):   #重置
        Robot.__init__(self)
        return self.robt_obs  # reward, done, info can't be included

    def render(self, mode="human"):    #渲染
        self.draw()
        plt.pause(1)
        # pass

    def step(self, action):  #采取动作
        self.stp = self.stp+1
        deta_theta1 = action[0]
        deta_theta2 = action[1]
        deta_theta3 = action[2]
        deta_fai1 = action[3]
        deta_fai2 = action[4]
        deta_fai3 = action[5]
        deta_d = action[6]
        if self.theta1+deta_theta1 >= self.theta_max or self.theta1+deta_theta1 <= self.theta_min:
            pass
            # print("超出弯曲限制")
        else:
            self.theta1 = self.theta1+deta_theta1
        if self.theta2+deta_theta2 >= self.theta_max or self.theta2+deta_theta2 <= self.theta_min:
            pass
            # print("超出弯曲限制")
        else:
            self.theta2 = self.theta2+deta_theta2
        if self.theta3+deta_theta3 >= self.theta_max or self.theta3+deta_theta3 <= self.theta_min:
            pass
            # print("超出弯曲限制")
        else:
            self.theta3 = self.theta3+deta_theta3
        if self.d + deta_d >= self.d_max or self.d + deta_d <= self.d_min:
            pass
            # print("超出给进限制")
        else:
            self.d = self.d + deta_d
        self.fai1 = self.fai1+deta_fai1
        self.fai2 = self.fai2+deta_fai2
        self.fai3 = self.fai3+deta_fai3
        # ----------------------------------------------------------
        old_rob_end = self.robt_end  # 执行动作前末端点位置
        new_observation = self.rob_kf()  # 执行动作，返回机械臂执行后形状
        new_rob_end = self.robt_end  # 执行动作后末端点位置
        reward, done = self.reward_func_fw(old_rob_end, new_rob_end)
        return new_observation, reward, done, {}   # 返回新的状态、奖励、结束标志、info={}

    def reward_func_fw(self, old_rob_end, new_rob_end):
        # ------------------------计算方向奖励与位置奖励------------------------------------
        # ------------位置奖励--------------------------------
        D_eo = np.linalg.norm(new_rob_end - self.barrier)  # 末端点与障碍物距离
        D_et = np.linalg.norm(new_rob_end - self.target)  # 末端点与目标点的距离
        r1 = D_et  # 末端到目标点距离
        r2 = D_eo  # 末端到障碍物距离
        Qe = 0.5  # 末端点电荷量
        Qt = 2  # 目标点电荷量
        Qo = 1  # 障碍物电荷量
        tao = 0.0785  # 方向奖励中的超参数

        f_obstacle = (-1/sqrt(2)) * exp(-(D_eo*D_eo) / 2)  # 位置奖励中的避障项
        f_triplet = 1/D_et  # 位置奖励中目标引导项
        R_location = f_obstacle + f_triplet  # 位置奖励

        # ------------方向奖励-----------------------
        ET = self.target - new_rob_end   # 末端点指向目标点的向量
        EO = self.barrier - new_rob_end  # 末端点指向障碍点的向量
        ET_ = (Qe * Qt * ET) / (r1 * r1 * np.linalg.norm(ET))  # 目标点吸引向量
        EO_ = -(Qe * Qo * EO) / (r2 * r2 * np.linalg.norm(EO))  # 障碍物排斥向量
        EC = new_rob_end - old_rob_end  # 末端点运动后的运动向量
        fai = acos(np.dot((EO_ + ET_), EC) / (np.linalg.norm((EO_ + ET_)) * np.linalg.norm(EC)))
        R_orientation = tao - fai * pi / 180  # 方向奖励

        # -------------计算完方向奖励与位置奖励-----------------------------------------------------------
        # 障碍物坐标（60，0，10）  碰撞半径：10倍根号3  威胁半径：10倍根号3+10 感知半径：10倍根号3+15
        d_alarm = 10 * sqrt(3) + 20  # 障碍物警戒距离
        d_danger = 10 * sqrt(3)     # 障碍物危险距离
        if D_eo <= d_alarm:  # 进入障碍物警戒区
            # print("进入警戒区域")
            lamda_location = (D_eo-d_danger) / (d_alarm-d_danger)
            lamda_orientation = (d_alarm-D_eo) / (d_alarm-d_danger)
        elif D_eo <= d_danger:  # 进入危险区
            # print("进入危险区域")
            lamda_location = 0
            lamda_orientation = 1
        else:       # 处于安全区域
            # print("处于安全区域")
            lamda_location = 1
            lamda_orientation = 0
        Reward = lamda_location * R_location + lamda_orientation * R_orientation
        R = self.min_dist()  # 机械臂臂体距离障碍物的最近距离
        if np.linalg.norm(new_rob_end - self.target) < self.threshold:
            Done = True
            print("yes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return Reward+100, Done
        elif self.stp >= self.max_stp:  # 到达最大探索次数
            Done = True
            print("###达到最大探索次数###")
        elif R <= 10 * sqrt(3):
            Done = True
            Reward = Reward-100
            print("碰撞")
        else:
            Done = False
        return Reward, Done

    def min_dist(self):  # 计算蛇形臂臂体与障碍物最小距离
        temp = self.robt_obs.reshape(200, 3)
        minDist = np.linalg.norm(temp[0] - self.barrier)
        for i in temp:
            temp_dist = np.linalg.norm(i - self.barrier)
            if temp_dist < minDist:
                minDist = temp_dist
        return minDist

# --------------------------------------按照上面gym环境进行设计自己的环境-----------------------------


if __name__ == '__main__':

    env = CustomEnv()  # 用来测试环境
    print("action_space: ", env.action_space.shape)
    print("observation_space: ", env.observation_space.shape)
    # print("action_sample: ", env.action_space.sample())
    # print("observation_sample", env.observation_space.sample())
    print("action_sample_shape: ", env.action_space.sample().shape)
    print("observation_sample_shape", env.observation_space.sample().shape)
    check_env(env)  # 最后检查环境是否符合标准用
    eposides = 10
    for ep in range(eposides):
        obs = env.reset()
        done = False
        rewards = 0
        while not done:
            action = env.action_space.sample()
            print(action)
            obs, reward, done, info = env.step(action)
            print("reward:", reward)
            env.render()
            rewards += reward
        print(rewards)