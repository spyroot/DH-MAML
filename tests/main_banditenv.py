import gym
from meta_critics.running_spec import RunningSpec, RunningSpecError
from meta_critics.app_globals import get_running_mode, SpecTypes
from meta_critics.envs.bandits.bandit_bernoulli_env import *

env = gym.make('Bandit-K5-v0', k=10)

observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    _, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()