# import asyncio
# import threading
# from copy import deepcopy
# from multiprocessing.context import BaseContext
# from typing import Optional
#
# import torch
# # import torch.multiprocessing as mp
#
# from meta_critics.running_spec import RunningSpec
# from meta_critics.policies.policy import Policy
# from meta_critics.agents.trajectory.agent_async_worker import AgentWorker
# from meta_critics.trajectory.advantage_episode import AdvantageBatchEpisodes
# from meta_critics.collectors.base.samplerbasedcollector import SamplerBasedCollector
#
#
# class AsyncMultiTaskSampler(SamplerBasedCollector):
#
#     def __init__(self,
#                  ctx: BaseContext,
#                  spec: RunningSpec,
#                  policy: Policy,
#                  baseline,
#                  env=None,
#                  cv: Optional[threading.Condition] = None,
#                  debug: Optional[bool] = False):
#         """
#         :param spec:
#         :param policy:
#         :param baseline:
#         :param env:
#         :param env:
#         :param debug:
#         """
#         super(AsyncMultiTaskSampler, self).__init__(spec, policy, env=env)
#         self.debug = self.spec.get('debug_agent')
#
#         assert self.policy is not None
#         self.num_workers = self.spec.get('num_workers')
#         if self.debug:
#             print(f"AsyncMultiTaskSampler: Creating MultiTaskSampler. "
#                   f"Number of agent workers: {self.num_workers}")
#
#         if ctx is None:
#             ctx = torch.multiprocessing.get_context("spawn")
#
#         # self.task_queue = ctx.JoinableQueue()
#         # self.train_episodes_queue = ctx.SimpleQueue()
#         # self.valid_episodes_queue = ctx.SimpleQueue()
#         policy_lock = ctx.Lock()
#         worker_lock = ctx.Lock()
#
#         # cv2 = threading.Condition()
#         self.workers = [AgentWorker(worker_idx, spec,
#                                     self.env.observation_space,
#                                     self.env.action_space,
#                                     self.policy,
#                                     deepcopy(baseline),
#                                     self.task_queue,
#                                     self.train_episodes_queue,
#                                     self.valid_episodes_queue,
#                                     policy_lock,
#                                     worker_lock,
#                                     debug=self.debug)
#                         for worker_idx in range(self.num_workers)]
#
#         with worker_lock:
#             for worker in self.workers:
#                 worker.daemon = True
#                 worker.start()
#
#             self._waiting_sample = False
#             self._event_loop = asyncio.get_event_loop()
#             self._train_consumer_thread = None
#             self._valid_consumer_thread = None
#
#         with cv:
#             cv.notify_all()
#
#     def sample_tasks(self, num_tasks):
#         """
#         :param num_tasks:
#         :return:
#         """
#         return self.env.unwrapped.sample_tasks(num_tasks)
#
#     def sample_async(self, tasks, **kwargs):
#         """
#         :param tasks:
#         :param kwargs:
#         :return:
#         """
#         if self._waiting_sample:
#             raise RuntimeError('calling async while waiting.')
#         [self.task_queue.put((index, task)) for index, task in enumerate(tasks)]
#         futures = self._start_consumer_threads(tasks, num_steps=kwargs.get('num_steps', 1))
#         self._waiting_sample = True
#         return futures
#
#     def sample_wait(self, episodes_futures):
#         """
#         :param episodes_futures:
#         :return:
#         """
#         if not self._waiting_sample:
#             raise RuntimeError('Calling `sample_wait` without any prior call to `sample_async`.')
#
#         async def _wait(train_futures, valid_futures):
#             """
#             :param train_futures:
#             :param valid_futures:
#             :return:
#             """
#             train_episodes = await asyncio.gather(*[asyncio.gather(*f) for f in train_futures])
#             valid_episodes = await asyncio.gather(*valid_futures)
#             return train_episodes, valid_episodes
#
#         samples = self._event_loop.run_until_complete(_wait(*episodes_futures))
#         self._join_consumer_threads()
#         self._waiting_sample = False
#         return samples
#
#     def sample(self, tasks, **kwargs):
#         """
#         :param tasks:
#         :param kwargs:
#         :return:
#         """
#         futures = self.sample_async(tasks, **kwargs)
#         return self.sample_wait(futures)
#
#     @property
#     def train_consumer_thread(self):
#         """
#         :return:
#         """
#         if self._train_consumer_thread is None:
#             raise ValueError()
#         return self._train_consumer_thread
#
#     @property
#     def valid_consumer_thread(self):
#         """
#         :return:
#         """
#         if self._valid_consumer_thread is None:
#             raise ValueError()
#         return self._valid_consumer_thread
#
#     def _start_consumer_threads(self, tasks, num_steps=1):
#         """
#         :param tasks:
#         :param num_steps:
#         :return:
#         """
#         # Start train episodes consumer thread
#         train_episodes_futures = [[self._event_loop.create_future() for _ in tasks] for _ in range(num_steps)]
#
#         if self.debug:
#             print(f"AsyncMultiTaskSampler: Starting consumer thread for training queue.")
#
#         self._train_consumer_thread = threading.Thread(target=self._create_consumer,
#                                                        args=(None, self.train_episodes_queue, train_episodes_futures),
#                                                        kwargs={'events': self._event_loop},
#                                                        name='train-consumer')
#         self._train_consumer_thread.daemon = True
#         self._train_consumer_thread.start()
#
#         if self.debug:
#             print(f"AsyncMultiTaskSampler: Starting consumer thread for validation queue.")
#
#         # Start second consumer thread
#         valid_episodes_futures = [self._event_loop.create_future() for _ in tasks]
#         self._valid_consumer_thread = threading.Thread(target=self._create_consumer,
#                                                        args=(None,
#                                                              self.valid_episodes_queue,
#                                                              valid_episodes_futures),
#                                                        kwargs={'events': self._event_loop},
#                                                        name='valid-consumer')
#         self._valid_consumer_thread.daemon = True
#         self._valid_consumer_thread.start()
#         return train_episodes_futures, valid_episodes_futures
#
#     def _join_consumer_threads(self):
#         """
#
#         :return:
#         """
#         if self._train_consumer_thread is not None:
#             self.train_episodes_queue.put(None)
#             self.train_consumer_thread.join()
#             if self.debug:
#                 print(f"AsyncMultiTaskSampler: consumer trainer thread joined back.")
#
#         if self._valid_consumer_thread is not None:
#             self.valid_episodes_queue.put(None)
#             self.valid_consumer_thread.join()
#             if self.debug:
#                 print(f"AsyncMultiTaskSampler: consumer trainer thread validation join back.")
#
#         self._train_consumer_thread = None
#         self._valid_consumer_thread = None
#
#     def close(self):
#         """
#
#         :return:
#         """
#         if self.closed:
#             return
#
#         for _ in range(self.num_workers):
#             self.task_queue.put(None)
#
#         self.task_queue.join()
#         self._join_consumer_threads()
#         self.closed = True
#
#     def select_action(self, ob_id, state):
#         state = torch.from_numpy(state).float().unsqueeze(0)
#         probs = self.policy(state)
#         m = Categorical(probs)
#         action = m.sample()
#         self.saved_log_probs[ob_id].append(m.log_prob(action))
#         return action.item()
#
#     def sample(self, episode_id, advantages, returns, actions, observations, lengths, rewards, mask):
#         self.episods[episode_id].append(reward)
#
#     def _create_consumer(cv: threading.Condition, _queue, _fut, events=None):
#         """
#         :param queue:
#         :param futures:
#         :param loop:
#         :return:
#         """
#         if events is None:
#             events = asyncio.get_event_loop()
#
#         while True:
#             data = _queue.get()
#             if data is None:
#                 break
#             index, step, advantages, returns, actions, observations, lengths, rewards, mask, dev, batch_size = data
#             # print(advantages.requires_grad)
#             # print(advantages.returns)
#             # print(advantages.actions)
#             # print(advantages.observations)
#             # print(advantages.lengths)
#             # print(advantages.rewards)
#
#             future = _fut if (step is None) else _fut[step]
#             if not future[index].cancelled():
#                 events.call_soon_threadsafe(future[index].set_result, AdvantageBatchEpisodes(batch_size=batch_size,
#                                                                                              device=dev,
#                                                                                              advantages=torch.from_numpy(
#                                                                                                      advantages),
#                                                                                              returns=torch.from_numpy(
#                                                                                                      returns),
#                                                                                              actions=torch.from_numpy(
#                                                                                                      actions),
#                                                                                              observations=torch.from_numpy(
#                                                                                                      observations),
#                                                                                              lengths=torch.from_numpy(
#                                                                                                      lengths),
#                                                                                              rewards=torch.from_numpy(
#                                                                                                      rewards),
#                                                                                              mask=torch.from_numpy(
#                                                                                                      mask)))
