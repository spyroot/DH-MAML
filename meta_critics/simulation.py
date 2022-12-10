import asyncio
from typing import Tuple, List

import torch

from meta_critics.envs.env_creator import env_creator
from meta_critics.ioutil.term_util import print_red
from meta_critics.policies.policy import Policy
from meta_critics.running_spec import RunningSpec
from meta_critics.trajectory.advantage_episode import AdvantageBatchEpisodes
from meta_critics.objective.reinforcement_learning import reinforce_loss
from meta_critics.envs.env_vectorized_meta_task import BaseVecMetaTaskEnv
from util import create_env_from_spec


class RemoteSimulation:
    def __init__(self,
                 observer_id: int,
                 spec: RunningSpec = None,
                 policy: Policy = None,
                 baseline=None,
                 debug=False):

        super(RemoteSimulation, self).__init__()
        # print("Creating simulation")

        self.spec = spec
        self.debug = debug
        self.policy = policy
        self.observer_id = observer_id

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
            self.seed = self.seed + observer_id + self.num_traj

        env = create_env_from_spec(self.spec)
        self.envs = BaseVecMetaTaskEnv([env_creator(self.env_name,
                                                    env_kwargs=self.env_kwargs,
                                                    seed=self.seed,
                                                    debug=self.spec.get('debug_env'))
                                        for _ in range(self.num_traj)],
                                       observation_space=env.observation_space,
                                       action_space=env.action_space,
                                       debug=self.spec.get('debug_env'))

        self.baseline = baseline

        # all specs agent required
        self.gamma = self.spec.get('gamma_factor', 'trainer')
        self.fast_lr = self.spec.get('fast_lr', 'meta_task')
        self.num_steps = self.spec.get('num_steps', 'meta_task')
        self.gae_lambda = self.spec.get('gae_lambda_factor', 'trainer')

    async def stop(self):
        pass

    async def start(self, queue, task_list, num_meta_tasks, debug=False):
        """Starts asyncio coroutine.
        :param queue:  A queue method will use to serialize trajectories.
        :param task_list: a list of task.
        :param num_meta_tasks:
        :param debug:
        :return:
        """
        try:
            for i, task in enumerate(task_list):
                self.envs.reset_task(task)
                await self.sample(queue)
        except Exception as err:
            print_red("Error in meta sample: ", err)
            raise err

    async def meta_tests(self, tasks):
        try:
            trajectory_train = []
            meta_test_episodes = []
            for i, task in enumerate(tasks):
                self.envs.reset_task(task)
                _meta_train, _meta_test = await self.meta_test()
                trajectory_train.append(_meta_train)
                meta_test_episodes.append(_meta_test)
        except Exception as err:
            print("Error in meta test", err)
            raise err

        return trajectory_train, meta_test_episodes

    async def meta_test(self) -> Tuple[List[AdvantageBatchEpisodes], AdvantageBatchEpisodes]:
        """
        :return:
        """
        params = None
        trajectory_train = []
        for step in range(self.num_steps):
            train_episodes = self.create_episodes(params=params)
            loss = reinforce_loss(self.policy, train_episodes, W=params)
            params = self.policy.update_params(loss, params=params,
                                               step_size=self.fast_lr,
                                               first_order=True)
            trajectory_train.append(train_episodes)

        return trajectory_train, self.create_episodes(params=params)

    async def sample(self, queue: asyncio.Queue) -> None:
        """
        :param queue:
        :return:
        """
        params = None
        for step in range(self.num_steps):
            train_episodes = self.create_episodes(params=params)
            await queue.put({"train": train_episodes})
            loss = reinforce_loss(self.policy, train_episodes, W=params)
            params = self.policy.update_params(loss,
                                               params=params,
                                               step_size=self.fast_lr,
                                               first_order=True)

        valid_episodes = self.create_episodes(params=params)
        await queue.put({"validation": valid_episodes})

    def create_episodes(self, params=None) -> AdvantageBatchEpisodes:
        """
        :param params:
        :return:
        """
        remap_dtype = self.spec.get("remap_types", "trajectory_sampler")
        episodes = AdvantageBatchEpisodes(batch_size=self.batch_size,
                                          gamma=self.gamma,
                                          device=self._device,
                                          remap_dtype=remap_dtype)

        for item in self.sample_trajectories(params=params):
            episodes.append(*item)

        self.baseline.fit(episodes)
        episodes.recompute_advantages(self.baseline,
                                      gae_lambda=self.gae_lambda,
                                      normalize=True)
        return episodes

    def sample_trajectories(self, params=None):
        """
        :param params:
        :return:
        """
        observations, info = self.envs.reset()
        with torch.no_grad():
            while True:
                if self.envs.is_terminated() or self.envs.is_truncated():
                    break
                if self.envs.is_done():
                    break
                observations_tensor = torch.from_numpy(observations)
                if observations_tensor is None:
                    continue

                # if self.debug:
                # if torch.all(observations_tensor == 0.0):
                # if self.debug:
                # print("Return tensor all zero and term + trunc ",
                #       np.count_nonzero(self.envs._terminateds)
                #       + np.count_nonzero(self.envs._truncateds))
                actions_tensor = self.policy(observations_tensor.float(), W=params).sample()
                actions = actions_tensor.cpu().numpy()
                new_observations, rewards, _, _, infos = self.envs.step(actions)
                batch_ids = infos['batch_ids']

                yield observations, actions, rewards, batch_ids
                observations = new_observations
