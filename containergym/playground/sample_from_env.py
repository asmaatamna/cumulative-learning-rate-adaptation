from env import ContainerEnv
import os
import sys


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

config_file = "5containers_2presses_2.json"
n_rollouts = 1

env = ContainerEnv.from_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../configs/" + config_file))
observation, info = env.reset(seed=1)

for _ in range(n_rollouts):
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print("Action: {}, reward: {}".format(action, reward))

        if terminated or truncated:
            observation, info = env.reset()
            break
