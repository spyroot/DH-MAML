"""
This meta task environment for v4 Cheetah.
THere are a bunch of fixes here , it should work in any new GYm / Mujoco environment
Mus
"""
from typing import Dict, List, Any, Optional
import mujoco
import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

from gym.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv as HalfCheetahEnv_


class HalfCheetahEnv(HalfCheetahEnv_):
    def __init__(self,
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 **kwargs):
        super(HalfCheetahEnv, self).__init__(**kwargs)
        # print("TASK", task)
        print("KWARGS", kwargs)

    # def _get_obs(self):
    #     return np.concatenate([
    #         self.data.qpos.flat[1:],
    #         self.data.qvel.flat,
    #         self.get_body_com("torso").flat]).astype(np.float64).flatten()

    def viewer_setup(self):
        self.viewer.cam.type = 2
        camera_name = "track"
        camera_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera_name,
        )
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        # Hide the overlay
        self.viewer._hide_overlay = True

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        return super().reset(seed=seed)

    # def render(self, mode='human'):
    #     if mode == 'rgb_array':
    #         self._get_viewer(mode).render()
    #         # window size used for old mujoco-py:
    #         width, height = 500, 500
    #         data = self._get_viewer(mode).read_pixels(width, height, depth=False)
    #         return data
    #     elif mode == 'human':
    #         self._get_viewer(mode).render()


class HalfCheetahVelEnv(HalfCheetahEnv):
    """Half-cheetah environment with target velocity, as described in [1]. The 
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand.py

    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each 
    time step a reward composed of a control cost and a penalty equal to the 
    difference between its current velocity and the target velocity. The tasks 
    are generated by sampling the target velocities from the uniform 
    distribution on [0, 2].

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for 
        model-based control", 2012 
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """

    def __init__(self,
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 task=None, low=0.0, high=2.0, **kwargs):
        if task is None:
            task = {}

        self._task = task
        self.low = low
        self.high = high

        self._goal_vel = task.get('velocity', 0.0)
        super(HalfCheetahVelEnv, self).__init__(**kwargs)

    def step(self, action):
        """
        """
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False

        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost,
                     task=self._task)

        return observation, reward, done, False, infos

    def sample_tasks(self, num_tasks: int) -> List[dict[str, Any]]:
        """  Sample n tasks.
        :param num_tasks:
        :return:
        """
        velocities = self.np_random.uniform(self.low, self.high, size=(num_tasks,))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def reset_task(self, task) -> None:
        """ Reset task for velocity
        :param task:
        :return:
        """
        self._task = task
        self._goal_vel = task['velocity']
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        return super().reset(seed=seed)


class HalfCheetahDirEnv(HalfCheetahEnv):
    """Half-cheetah environment with target direction, as described in [1]. The 
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand_direc.py

    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each 
    time step a reward composed of a control cost and a reward equal to its 
    velocity in the target direction. The tasks are generated by sampling the 
    target directions from a Bernoulli distribution on {-1, 1} with parameter 
    0.5 (-1: backward, +1: forward).

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for 
        model-based control", 2012 
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """

    def __init__(self, task=None, **kwargs):
        """
        :param task:
        """
        if task is None:
            task = {}

        print("TASK", task)
        print("KWARGS", kwargs)

        self._task = task
        self._goal_dir = task.get('direction', 1)
        super(HalfCheetahDirEnv, self).__init__(**kwargs)

    def step(self, action):
        """
        :param action:
        :return:
        """
        x_pos_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_pos_after = self.data.qpos[0]

        forward_vel = (x_pos_after - x_pos_before) / self.dt
        forward_reward = self._goal_dir * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost,
                     task=self._task)
        return observation, reward, done, False, infos

    def sample_tasks(self, num_tasks: int) -> List[dict[str, Any]]:
        """ sample n tasks.
        :param num_tasks:
        :return:
        """
        directions = 2 * self.np_random.binomial(1, p=0.5, size=(num_tasks,)) - 1
        tasks = [{'direction': direction} for direction in directions]
        return tasks

    def reset_task(self, task) -> None:
        """ Task is direction for ant
        :param task:
        :return:
        """
        self._task = task
        self._goal_dir = task['direction']
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        return super().reset(seed=seed)
