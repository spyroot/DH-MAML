import asyncio
import sys
import threading
import traceback
from collections import OrderedDict
from typing import Optional

import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef

from meta_critics.modules.baseline import LinearFeatureBaseline
from meta_critics.policies.policy_creator import PolicyCreator
from meta_critics.rpc.async_logger import AsyncLogger
from meta_critics.running_spec import RunningSpecError, RunningSpec
from meta_critics.simulation import RemoteSimulation
from util import create_env_from_spec


class RpcObservers:
    def __init__(self, spec: RunningSpec,
                 self_logger: Optional[AsyncLogger] = None, debug: Optional[bool] = False):

        self.spec = spec
        self.self_logger = self_logger
        self.debug = False

        if not self.spec.contains("device"):
            raise ValueError("Observer: spec must container device.")

        self.device = self.spec.get('device')

        self.agent_policy = None
        self.linear_baseline = None

        self.id = rpc.get_worker_info().id - 1
        # self.update_on_task = DistributedAgent.update_on_task
        self.agent_rref = None

        self.seq = 0
        self.read_spec()

        self.lock = threading.Lock()
        self.loop = asyncio.new_event_loop()

        self.num_task = self.spec.get('num_meta_task', 'meta_task')
        self.worker_rref = RRef(self)
        self.simulation = None
        # self_logger.emit("Rpc observer")
        # self.log(green_str("Starting rpc observer."))

    def log(self, msg):
        """
        :param msg:
        :return:
        """
        # if self.self_logger is not None:
        #     self.loop.call_soon_threadsafe(functools.partial(self.self_logger.emit, msg))

    def read_spec(self):
        """
        :return:
        """
        try:
            # print(f"Observer{self.id} creating simulation.")
            # env = create_env_from_spec(self.spec)
            # policy_creator = PolicyCreator(env, self.spec)
            # self.agent_policy, is_continuous = policy_creator()
            # self.agent_policy.share_memory()
            # self.linear_baseline = LinearFeatureBaseline(env, self.device).to(self.device)
            self.num_task = self.spec.get('num_meta_task', 'meta_task')

        except RunningSpecError as r_except:
            print(f"Error:", r_except)
            sys.exit(100)

    def update_agent_rref(self, agent_rref, arg):
        """

        :param agent_rref:
        :param arg:
        :return:
        """
        # if self.debug:
        #     await logger.debug(f"Observer {self.id} update agent policy.")
        self.agent_rref = arg

    def update_observer_policy(self, agent_rref, parameters):
        """

        :param agent_rref:
        :param parameters:
        :return:
        """
        # await logger.info(f"Observer {self.id} update agent policy.")
        if self.agent_policy is None:
            env = create_env_from_spec(self.spec)
            policy_creator = PolicyCreator(env, self.spec)
            self.agent_policy, is_continuous = policy_creator()

        self.agent_policy.update_parameters(parameters)
        self.agent_policy.to(self.device)

        if self.simulation is None:
            env = create_env_from_spec(self.spec)
            self.linear_baseline = LinearFeatureBaseline(env, self.device).to(self.device)
            self.simulation = RemoteSimulation(self.id,
                                               spec=self.spec,
                                               policy=self.agent_policy,
                                               baseline=self.linear_baseline)

    @rpc.functions.async_execution
    def update_observer_grad(self, agent_rref, parameters):
        """Updates grads for local policy
        :param agent_rref:
        :param parameters:
        :return:
        """
        if self.debug:
            self.log(f"{agent_rref} updating received type {type(parameters)}")

        fut = torch.futures.Future()
        try:
            with self.lock:
                if self.agent_policy is None:
                    env = create_env_from_spec(self.spec)
                    policy_creator = PolicyCreator(env, self.spec)
                    self.agent_policy, is_continuous = policy_creator()

                self.agent_policy.update_grads(parameters)
                self.agent_policy.to(self.device)
                if self.simulation is None:
                    self.simulation = RemoteSimulation(self.id,
                                                       spec=self.spec,
                                                       policy=self.agent_policy,
                                                       baseline=self.linear_baseline)
                fut.set_result(True)
        except Exception as err:
            print("update_observer_grad error:", err)
            print(traceback.print_exc())

        return fut

    @rpc.functions.async_execution
    def shutdown(self, agent_rref):
        fut = torch.futures.Future()
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(loop.shutdown_asyncgens())
            self.simulation.stop()
            loop.close()
            fut = torch.futures.Future()
            fut.set_result(True)
        except Exception as er:
            print("Failed shutdown loop. Error", er)

        return fut

    @rpc.functions.async_execution
    def sample_episode(self, agent_rref, n_steps, task_id):
        """
        :param agent_rref:
        :param n_steps:
        :param task_id:
        :return:
        """
        future_task = []
        for _ in task_id:
            fut = torch.futures.Future()
            future_task.append(fut)
        future_task = torch.futures.collect_all(future_task)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.async_sample_episode(loop,
                                                              future_task,
                                                              agent_rref,
                                                              n_steps, task_id))
        except KeyboardInterrupt as kb:
            print("Observer recieved kb")
            loop.run_until_complete(loop.shutdown_asyncgens())
            raise kb

        # await logger.info(f"Observer {self.id} return rpc future.")
        return future_task

    async def async_sample_episode(self, loop, fut, agent_rref, n_steps, meta_task, debug=True):
        """
        :param loop:
        :param fut:
        :param agent_rref:
        :param n_steps:
        :param meta_task:
        :param debug:
        :return:
        """
        self.seq += 1
        train_queue = asyncio.Queue()

        _val = []
        _train = []

        # await logger.debug(f"Observer {self.id} received meta task goal.")

        # TODO for now num step for train and validation is same
        for step in range(n_steps):
            async def trajectory_consumer(q):
                while True:
                    try:
                        future_item = await q.get()
                        if future_item is None:
                            break

                        if "train" in future_item:
                            data = future_item["train"].clone_as_tuple()
                            _train.append(data)
                        else:
                            data = future_item["validation"].clone_as_tuple()
                            _val.append(data)

                        # q.task_done()
                    except asyncio.CancelledError as canceled_err:
                        # print_red(f"Canceling observer consumer")
                        fut.set_exception(canceled_err)
                        raise canceled_err
                    except Exception as err:
                        print(err)
                        print(traceback.format_exc())
                        fut.set_exception(err)
                        raise err
                    finally:
                        try:
                            q.task_done()
                        except ValueError as err:
                            pass

            assert len(meta_task) == self.num_task
            consumers = []
            for i in range(self.num_task):
                consumer = loop.create_task(trajectory_consumer(train_queue))
                consumers.append(consumer)

            # we start produced and wait on train queue consumer
            await self.simulation.start(train_queue, meta_task, self.num_task)
            await train_queue.join()

            for consumer in consumers:
                consumer.cancel()

            # we wait so all cancel
            await asyncio.gather(*consumers, return_exceptions=True)

        def update_agent(f):
            """We need detach parameter"""
            # print("Update_agent callback. Updating gradient parameters.")
            try:
                # print("Updating agent.")
                current_parameters = OrderedDict(self.agent_policy.cpu().named_parameters())
                # odict_keys(['sigma', 'layer1.weight', 'layer1.bias', 'layer2.weight', 'layer2.bias', 'mu.weight',
                #             'mu.bias'])
                #  print(current_parameters['layer1.weight'].grad)
                self.agent_rref.rpc_sync().rpc_sync_grad(self.worker_rref, current_parameters)
            except Exception as ex:
                print(ex)
                print(traceback.format_exc())

        # await self.logger.debug(f"Observer {self.id} sending trajectory.")
        fut.set_result((_train, _val))
        fut.add_done_callback(update_agent)
