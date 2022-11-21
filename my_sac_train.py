import gym
import numpy as np
import my_gym_env
from stable_baselines3 import SAC
from stable_baselines3.common.noise import ActionNoise
env = my_gym_env.CustomEnv()


model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_log/", device='cuda')
            # policy_kwargs=dict(net_arch=dict(pi=[64, 64], qf=[400, 300])))
model.learn(total_timesteps=int(1e6), log_interval=4, tb_log_name="方向奖励函数")
model.save("sac_10")

