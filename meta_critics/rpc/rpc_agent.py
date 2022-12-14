#
#  Now let add async from agent
#  Mus
import asyncio
import threading
import traceback
from abc import ABC
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional, Callable

import gym
import numpy as np
import torch
import torch.distributed.rpc as rpc
from numpy import ndarray
from torch.distributed.rpc import RRef, remote
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from meta_critics.ioutil.term_util import print_red
from meta_critics.policies.policy import Policy
from meta_critics.rpc.async_logger import AsyncLogger
from meta_critics.rpc.generic_rpc_agent import GenericRpcAgent
from meta_critics.rpc.metric_receiver import MetricReceiver
from meta_critics.rpc.rpc_observer import RpcObservers
from meta_critics.rpc.shared_vars import OBSERVER_NAME
from meta_critics.running_spec import RunningSpec
from meta_critics.trajectory.advantage_episode import AdvantageBatchEpisodes
from meta_critics.models.concurrent_trpo import ConcurrentMamlTRPO
from meta_critics.base_trainer.internal.utils import to_numpy


# from aiologger import Logger
# logger = Logger.with_default_handlers()

class DistributedAgent(GenericRpcAgent, ABC):
    def __init__(self,
                 agent_policy: Policy,
                 spec: RunningSpec,
                 world_size: int,
                 env: Optional[gym.Env] = None,
                 debug: Optional[bool] = True,
                 self_logger: Optional[AsyncLogger] = None):
        """
        :param world_size:
        :param spec:
        """
        super(DistributedAgent, self).__init__(agent_policy=agent_policy,
                                               spec=spec,
                                               world_size=world_size,
                                               env=env)

        self.debug = None
        self.agent_rref = RRef(self)
        self.self_logger = self_logger

        # self.self_logger.emit("Starting Agent")

        self.ob_rrefs = []
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, RpcObservers, args=(deepcopy(spec),)))
        self.pending = len(self.ob_rrefs)

        # global specification shared by trainer, observer and agent
        self.spec = spec
        self.device = self.spec.get('device')

        self.rets = None

        # agent policy
        self.agent_policy = agent_policy

        # self lock , block modification for a policy.
        self.self_lock = threading.Lock()
        self.num_task = self.spec.get('num_meta_task', 'meta_task')
        self.queue = None

        # agent policy seq
        self.update_seq = 0

        # internal asyncio loop
        self.loop = asyncio.new_event_loop()
        self.log("Starting DistributedMetaTrainer")

    def log(self, msg):
        """
        :param msg:
        :return:
        """
        # if self.self_logger is not None:
        #     self.self_logger.emit(msg)
        # self.loop.call_soon(functools.partial(self.self_logger.emit, msg))

    def rpc_sync_grad(self, worker_id, parameters):
        """ This rpc all called by the observer
        :param worker_id:
        :param parameters:
        :return:
        """
        # print(f"Agent rpc_sync_grad grads {worker_id}")
        with self.self_lock:
            self.agent_policy.update_grads(parameters)
            self.agent_policy.to(self.device)

    def sync_policy(self, worker_id, parameters):
        """
        :param worker_id:
        :param parameters:
        :return:
        """
        with self.self_lock:
            self.update_seq += 1
            self.agent_policy.update_parameters(parameters)

    def get_self_rref(self):
        """ returns self rref
        :return:
        """
        return self.agent_rref

    def get_policy(self):
        """returns current policy.
        :return:
        """
        return self.agent_policy

    def get_policy_rrefs(self):
        """
        :return:
        """
        param_rrefs = [rpc.RRef(param) for param
                       in self.agent_policy.parameter]
        return param_rrefs

    async def sample_tasks(self):
        # await logger.debug(f"Agent {self.agent_rref} sample tasks")
        return self.env.unwrapped.sample_tasks(self.num_task)

    async def broadcast_policy(self):
        """Broadcasts to all agent policy
        :return:
        """
        # await logger.debug(f"agent {self.agent_rref} broadcast policy.")
        with self.self_lock:
            p = OrderedDict(self.agent_policy.cpu().named_parameters())
            for ob_rref in self.ob_rrefs:
                ob_rref.rpc_sync().update_observer_policy(self.agent_rref, p)

    async def shutdown_observers(self):
        """Broadcasts to all observers to shut down.
        If agent can't recover from error, torch rpc doesn't handle many issues.
        So it on keyboard event it better to send shutdown, otherwise RPC socket will
        remain open.

        Generally, if torch provide good wait to signal via UNIX signal SIG TERM etc.
        :return:
        """
        print("Agent send shutdown to observer")
        for ob_rref in self.ob_rrefs:
            f = ob_rref.rpc_sync().shutdown(self.agent_rref)
            f.wait()

    async def broadcast_grads(self):
        """Broadcasts to all agent policy,  Agent distribute policy to each observer.
        :return:
        """
        # we need detach to cpu
        # await logger.debug(f"agent {self.agent_rref} broadcasting grads")
        with self.self_lock:
            p = OrderedDict(self.agent_policy.cpu().named_parameters())
            for ob_rref in self.ob_rrefs:
                f = ob_rref.rpc_async().update_observer_grad(self.agent_rref, p)
                f.wait()

    async def broadcast_rref(self):
        """Broadcasts to all worker rref, so we avoid global function or symbols.
        not sure why torch guys doing example with global mp.Span
        separate process space.
        :return:
        """
        self.log(f"agent {self.agent_rref} broadcasting rref")
        for ob_rref in self.ob_rrefs:
            ob_rref.rpc_sync(timeout=180).update_agent_rref(self.agent_rref, self.agent_rref)

    async def distribute_tasks(self, queue, n_steps=0):
        """Distribute task to each observer. Method doesn't wait.
        It distributes tasks via async RPC and put future to a queue.
        There is separate consumer wait on this event. Note RPC
        doesn't wait
        :param queue: a queue where RPC call will put a torch.Future
        :param n_steps: number of step observer need execute policy gradient.
               By default, it just 1.
        :return:
        """
        self.queue = queue
        for ob_rref in self.ob_rrefs:
            meta_task = await self.sample_tasks()
            f = ob_rref.rpc_async().sample_episode(self.agent_rref, n_steps, meta_task)
            await queue.put((f, meta_task))

    @staticmethod
    async def returns_metric(episodes: List[AdvantageBatchEpisodes]) -> np.ndarray:
        """
        :param episodes:
        :return:
        """
        return to_numpy([episode.rewards.sum(dim=0) for episode in episodes])

    @staticmethod
    def default_metric_callback(batch_stats, episode_num, writer):
        """
        :param batch_stats:
        :param episode_num:
        :param writer:
        :return:
        """
        for metrics in batch_stats:
            tqdm_update_dict = {}
            for k in metrics.keys():
                v = metrics[k]
                if not isinstance(v, torch.Tensor):
                    loss_term = torch.stack(metrics[k]).mean().item()
                    tqdm_update_dict[k] = loss_term
                    writer.add_scalar(f"loss/{k}", loss_term.cpu(), episode_num)
                else:
                    tqdm_update_dict[k] = v.item()
                    writer.add_scalar(f"loss/{k}", v.item(), episode_num)

    @staticmethod
    async def metric_dequeue(trainer_metric_queue: asyncio.Queue,
                             writer: SummaryWriter,
                             metric_receiver: MetricReceiver,
                             i_episode: int,
                             tqdm_iter: tqdm,
                             callback: Optional[Callable] = None,
                             flush_io: Optional[bool] = False):
        """
        :param callback:  a callback that we want execute. (TODO)
        :param tqdm_iter: tqdm iterator dict note we using asyncio version.
        :param i_episode:  current episode number
        :param writer:  tensorboard writer.  todo move that to metric writer to avoid IO cost.
        :param trainer_metric_queue: a queue metrics pushed from a trainer at each step
        :param metric_receiver:
        :param flush_io:  torch does something with IO pipes during fork().
                         Generally avoid write big chunk to IO.
        :return:
        """
        # for a sake of time metric average for LS compute here.
        # TODO I have internal state metric counter to make it more generic
        # this call should call generic callback and serialize metric without hard codings
        # any specific.  for now we want to track average LS step TRPO took to push KL term
        # it useful to track for each task.
        # await logger.info("Received future from remote client.")
        while True:
            try:
                algo_metrics, reward_metrics = await trainer_metric_queue.get()
                if algo_metrics is None:
                    break

                metric_data = {}

                tqdm_update_dict = {}
                for k in algo_metrics.keys():
                    v = algo_metrics[k]
                    if isinstance(v, torch.Tensor):
                        tqdm_update_dict[k] = v.mean().item()
                        metric_data[f"loss/{k}"] = v.mean().item()
                        writer.add_scalar(f"loss/{k}", v.mean().item(), i_episode)
                    elif isinstance(v, ndarray):
                        loss_term = algo_metrics[k].mean()
                        tqdm_update_dict[k] = loss_term
                        metric_data[f"loss/{k}"] = loss_term
                        writer.add_scalar(f"loss/{k}", loss_term, i_episode)
                    else:
                        tqdm_update_dict[k] = v
                        writer.add_scalar(f"loss/{k}", v, i_episode)
                        metric_data[f"loss/{k}"] = v

                for k, v in reward_metrics.items():
                    writer.add_scalar(f"rewards_mean/{k}", v.mean(), i_episode)
                    metric_data[f"rewards_mean/{k}"] = v.mean()
                    writer.add_scalar(f"rewards_sum/{k}", v.sum(), i_episode)
                    metric_data[f"rewards_sum/{k}"] = v.sum()

                # update metric listener
                metric_receiver.update(metric_data)

                # update tbar
                tqdm_iter.set_postfix(tqdm_update_dict)

                trainer_metric_queue.task_done()

            except Exception as exp:
                print("Error in metric dequeue:", exp)
                print(traceback.print_exc())
            finally:
                if flush_io:
                    writer.flush()

                try:
                    if not trainer_metric_queue.empty():
                        trainer_metric_queue.task_done()
                except ValueError as ve:
                    pass

    @staticmethod
    async def trainer_consumer(self_agent: GenericRpcAgent,
                               episode_queue: asyncio.Queue,
                               metric_queue: asyncio.Queue,
                               _meta_learner: ConcurrentMamlTRPO,
                               device='cpu'):
        """
        :param self_agent:  We keep self, so we can await if needed on class method.
        :param episode_queue:
        :param metric_queue:
        :param _meta_learner:
        :param device:
        :return:
        """
        while True:
            try:
                # wait for an item from the agent
                (remote_futures, meta_task) = await episode_queue.get()
                if remote_futures is None:
                    break

                remote_futures.wait()
                remote_value = remote_futures.value()

                list_train = []
                list_validation = []
                tr, ty = remote_value

                total_episode_len = 0
                train_rewards = []
                validation_rewards = []

                # pack all back to GPU and send
                for i, (xs, ys) in enumerate(zip(tr, ty)):
                    total_episode_len += torch.sum(xs.lengths) + torch.sum(ys.lengths)
                    train_rewards.append(xs.rewards.clone().sum(dim=0))
                    validation_rewards.append(ys.rewards.clone().sum(dim=0).clone())

                    t = AdvantageBatchEpisodes(batch_size=xs.batch_size,
                                               advantages=xs.advantages,
                                               returns=xs.returns,
                                               actions=xs.actions,
                                               observations=xs.observations,
                                               lengths=xs.lengths,
                                               rewards=xs.rewards,
                                               mask=xs.mask,
                                               device=torch.device(device),
                                               reward_dtype=xs.reward_dtype,
                                               action_dtype=xs.action_dtype,
                                               observations_dtype=xs.observations_dtype)

                    v = AdvantageBatchEpisodes(batch_size=ys.batch_size,
                                               advantages=ys.advantages,
                                               returns=ys.returns,
                                               actions=ys.actions,
                                               observations=ys.observations,
                                               lengths=ys.lengths,
                                               rewards=ys.rewards,
                                               mask=ys.mask,
                                               device=torch.device(device),
                                               reward_dtype=ys.reward_dtype,
                                               action_dtype=ys.action_dtype,
                                               observations_dtype=ys.observations_dtype
                                               )

                    t.to_gpu()
                    v.to_gpu()

                    list_train.append(t)
                    list_validation.append(v)

                metrics = await _meta_learner.step([list_train], list_validation)
                reward_metrics = {
                    "meta_train": torch.stack(train_rewards),
                    "meta_validation": torch.stack(validation_rewards)
                }

                # episode_queue.task_done()
                await metric_queue.put((metrics, reward_metrics))
                await self_agent.broadcast_grads()
            except asyncio.CancelledError:
                # print_red(f"Canceling trainer consumer")
                break
            except ValueError as verr:
                print_red(f"Error in trainer consumer: {verr}")
                print(traceback.print_exc())
                await self_agent.shutdown_observers()
                raise verr
            except Exception as generaLexp:
                print_red(f"Error in trainer consumer: {generaLexp}")
                print(traceback.print_exc())
                await self_agent.shutdown_observers()
                raise generaLexp
            finally:
                episode_queue.task_done()

    def save(self, step) -> bool:
        """Saving model if failed will not raise exception.
        :return:
        """
        try:
            save_freq = self.spec.get('save_freq', 'trainer')
            if step > 0 and step % save_freq == 0:
                model_file_name = self.spec.get('model_state_file', 'model_files')
                print(f"Saving model to a file. {model_file_name}")
                step_checkpoint = f"{model_file_name}_{step}"

                with open(model_file_name, 'wb') as f:
                    with self.self_lock:
                        state_dict = self.agent_policy.state_dict()
                    state_dict["last_step"] = step
                    torch.save(state_dict, f)

                with open(step_checkpoint, 'wb') as f:
                    state_dict["last_step"] = step
                    torch.save(state_dict, f)

                return True
        except Exception as err:
            print("Error failed to save model:", err)

        return False
