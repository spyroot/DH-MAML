from gym.wrappers import TimeLimit
import meta_critics.envs.mujoco.ant
import meta_critics.envs.mujoco.half_cheetah
# this test if env and time limit new API do work
# mus
import gym
env = gym.make('HalfCheetahVel-v4', render_mode="human")
env = TimeLimit(env, max_episode_steps=100)

observation, info = env.reset(seed=42)

for i in range(100000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if truncated:
        print(f"Pass a test at step {i}")
        break
    if terminated or truncated:
        observation, info = env.reset()
env.close()