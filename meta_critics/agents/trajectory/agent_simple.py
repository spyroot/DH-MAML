from typing import List, Any, Tuple

import torch

from meta_critics.trajectory.advantage_episode import AdvantageBatchEpisodes
from meta_critics.envs.env_vectorized_meta_task import BaseVecMetaTaskEnv
from meta_critics.envs.env_creator import env_creator
from meta_critics.policies.policy import Policy
from meta_critics.objective.reinforcement_learning import reinforce_loss
from meta_critics.running_spec import RunningSpec


class AgentSingleWorker:
    def __init__(self,
                 spec: RunningSpec,
                 observation_space,
                 action_space,
                 policy: Policy,
                 baseline):
        """
        :param spec:
        :param observation_space:
        :param action_space:
        :param policy:
        :param baseline:
        """
        super(AgentSingleWorker, self).__init__()
        print("Creating Agent")

        self.spec = spec
        self.policy = policy

        assert self.spec is not None
        assert self.policy is not None

        self.env_name = self.spec.get('env_name')
        self.env_kwargs = {}
        if hasattr(self.spec, 'env_args'):
            self.env_kwargs = self.spec.env_args

        self._device = self.spec.get('device')
        self.num_traj = self.spec.get('num_trajectory', 'meta_task')

        self.seed = self.spec.get('seed')
        if self.seed is not None:
            self.seed = self.seed * self.num_traj

        self.envs = BaseVecMetaTaskEnv([env_creator(self.env_name,
                                                    env_kwargs=self.env_kwargs,
                                                    seed=self.seed)
                                        for _ in range(self.num_traj)],
                                       observation_space=observation_space,
                                       action_space=action_space)

        self.policy = policy
        self.baseline = baseline

        self.num_grad_steps = self.spec.get('num_steps', 'meta_task')
        self.gae_lambda = self.spec.get('gae_lambda_factor', 'trainer')
        self.fast_lr = self.spec.get('fast_lr', 'meta_task')
        self.gamma = self.spec.get('gamma_factor', 'trainer')

    def collect_episodes(self, tasks: List[Any]) -> Tuple[List[AdvantageBatchEpisodes], List[AdvantageBatchEpisodes]]:
        """collect all episodes for list of tasks
        :return:
        """
        params = None
        tasks_train_episodes = []
        tasks_val_episodes = []
        for _ in tasks:
            for step in range(self.num_grad_steps):
                train_episodes = self.sample_single_task(param_dict=params)
                tasks_train_episodes.append(train_episodes)
                loss = reinforce_loss(self.policy, train_episodes, W=params)
                params = self.policy.update_params(loss, params=params, step_size=self.fast_lr, first_order=True)

            tasks_val_episodes.append(self.sample_single_task(param_dict=params))

        return tasks_train_episodes, tasks_val_episodes

    def sample_single_task(self, param_dict=None):
        """Sample episodes
        :param param_dict: dict store torch parameters
        :return: return episode with advantages.
        """
        episodes = AdvantageBatchEpisodes(batch_size=self.num_traj, gamma=self.gamma, device=self._device)
        for item in self.sample_trajectories(param_dict=param_dict):
            episodes.append(*item)

        self.baseline.fit(episodes)
        episodes.recompute_advantages(self.baseline, gae_lambda=self.gae_lambda, normalize=True)
        return episodes

    def sample_trajectories(self, param_dict=None):
        """ Sample trajectories
        :param param_dict: dict store torch parameters
        :return:
        """
        observations, info = self.envs.reset()
        with torch.no_grad():
            while True:
                if self.envs.is_terminated() or self.envs.is_truncated():
                    break
                observations_tensor = torch.from_numpy(observations)
                actions_tensor = self.policy(observations_tensor.float(), W=param_dict).collect_episodes()
                actions = actions_tensor.cpu().numpy()
                new_observations, rewards, _, _, infos = self.envs.step(actions)
                batch_ids = infos['batch_ids']

                yield observations, actions, rewards, batch_ids
                observations = new_observations
