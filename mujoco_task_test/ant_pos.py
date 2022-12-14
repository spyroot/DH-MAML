# AntPos-v4 test , general test for ant with fixed env no time limit.
# it should render and you should see ant
# Mus
import meta_critics.envs.mujoco.ant
import gym
env = gym.make('AntPos-v4', render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(100000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
env.close()
exit(1)
