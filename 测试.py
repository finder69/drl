import time

import gym
import numpy as np
import my_gym_env
from stable_baselines3 import SAC
env = my_gym_env.CustomEnv()

model = SAC.load("sac_10")

env.obs = env.reset()
env.render()
# action, _states = model.predict(obs, deterministic=True)
# print("action:", action)
# print("_states", _states)
while True:
    action, env.obs = model.predict(env.obs)
    env.obs, reward, done, info = env.step(action)
    print("action", action)
    print("reward", reward)
    env.render()
    if done:
      obs = env.reset()
      print("reset")
      break