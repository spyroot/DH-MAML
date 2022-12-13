#
#  Now let add async from agent
#  Mus
import asyncio
import os
import signal
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed.rpc as rpc
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from meta_critics.base_trainer.internal.utils import to_numpy
from meta_critics.ioutil.term_util import print_red, print_green
from meta_critics.modules.baseline import LinearFeatureBaseline
from meta_critics.policies.policy_creator import PolicyCreator
from meta_critics.models.concurrent_trpo import ConcurrentMamlTRPO
from meta_critics.rpc.async_logger import AsyncLogger
from meta_critics.rpc.metric_receiver import MetricReceiver
from meta_critics.rpc.rpc_agent import DistributedAgent
from meta_critics.running_spec import RunningSpec
from meta_critics.simulation import RemoteSimulation
from util import create_env_from_spec
from tqdm.asyncio import trange, tqdm

# import wandb
# wandb.init(project="dh-maml", entity="spyroot")
from aiologger import Logger

logger = Logger.with_default_handlers()

NUM_STEPS = 1
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"


# logging.basicConfig(level=logging.DEBUG, format="%(message)s")


def format_num(n):

    if isinstance(n, np.ndarray):
        return n
    elif isinstance(n, torch.Tensor):
        return n

    f = '{0:.3g}'.format(n)
    f = f.replace('+0', '+')
    f = f.replace('-0', '-')
    n = str(n)
    return f if len(f) < len(n) else n


def resole_primary_dir(path_to_dir: str, create_if_needed: Optional[bool] = False) -> str:
    """Resolve log dir and create if needed
    :param create_if_needed:
    :param path_to_dir:
    :return:
    """
    log_dir = path_to_dir.strip()
    if log_dir.startswith("~"):
        log_dir = str(Path.home()) + log_dir[1:]

    log_file_path = Path(log_dir).expanduser().resolve()

    if not log_file_path.exists():
        print(f"{log_dir} not found.")
        if create_if_needed:
            os.makedirs(log_file_path, exist_ok=True)
        else:
            raise FileNotFoundError(f"Error: dir {log_dir} not found.")

    if not log_file_path.is_dir():
        print(f"{log_dir} must be directory.")
        raise FileNotFoundError(f"Error {log_dir} must be directory not a file.")

    return str(log_file_path)


def resole_primary_from_spec(spec: RunningSpec, create_if_needed: Optional[bool] = False) -> str:
    """

    :param spec:
    :param create_if_needed:
    :return:
    """
    if spec.contains("create_dir"):
        create_if_needed = spec.get("create_dir")
    return resole_primary_dir(spec.get("log_dir"),
                              create_if_needed=create_if_needed)


class DistributedMetaTrainer:
    def __init__(self, spec: RunningSpec,
                 world_size: int,
                 self_logger: Optional[AsyncLogger] = None):
        """

        :param spec:
        :param world_size:
        """
        self.last_episode = None
        self.tf_writer = None
        self.meta_learner = None
        self.is_continuous = None
        self.agent_policy = None
        self.num_batches = None
        self.tqdm_iter = None
        self.env = None

        self.self_logger = self_logger

        self.agent = None
        self.started = False
        self.metric_queue = None
        self.trainer_queue = None

        self.spec = spec
        self.world_size = world_size
        self.log_dir = str(Path.cwd())

        self.self_logger.emit("Test 1")
        self.loop = asyncio.new_event_loop()

    def log(self, msg: str) -> None:
        """
        Logger in separate thread and in separate event loop.
        :param msg:
        :return:
        """
        # if self.self_logger is not None:
        #     self.loop.call_soon_threadsafe(functools.partial(self.self_logger.emit, msg))
        # else:
        #     print("Self logger is none.")

    async def start(self) -> None:
        """
        :return:
        """

        self.log(f"DistributedMetaTrainer world size{self.world_size} staring.")

        self.num_batches = self.spec.get('num_batches', 'meta_task')
        self.env = create_env_from_spec(self.spec)

        policy_creator = PolicyCreator(self.env, self.spec)
        self.agent_policy, self.is_continuous = policy_creator()

        # self.agent_policy.share_memory()
        self.agent_policy.to(self.spec.get('device'))

        self.load_model()

        # model
        self.meta_learner = ConcurrentMamlTRPO(self.agent_policy, self.spec)

        if self.spec.contains("log_dir"):
            self.log_dir = resole_primary_dir(self.spec.get("log_dir"))

        print(f"Tensorboard log location. {self.log_dir}")
        self.tf_writer = SummaryWriter(log_dir=self.log_dir)

        # distributed agents
        self.agent = DistributedAgent(agent_policy=self.agent_policy, spec=self.spec,
                                      world_size=self.world_size,
                                      self_logger=self.self_logger)

        self.started = True
        print_green(f"DistributedMetaTrainer world size{self.world_size} started.")

    def load_model(self):
        """Load model, if training interrupted will load checkpoint.
        :return:
        """
        model_file_name = self.spec.get('model_state_file', 'model_files')
        path_to_model = Path(model_file_name)
        if path_to_model.exists():
            with open(model_file_name, 'rb') as f:
                state_dict = torch.load(f, map_location=torch.device(self.spec.get("device")))
                if 'last_step' in state_dict:
                    last_step = state_dict["last_step"]
                    state_dict.pop("last_step")
                    print_green(f"Detected existing model. Loading from {model_file_name} from {last_step}.")
                else:
                    print_green(f"Detected existing model. Loading from {model_file_name}.")

                self.agent_policy.load_state_dict(state_dict)

    async def meta_test(self,
                        step: int,
                        metric_receiver: Optional[MetricReceiver] = None,
                        is_meta_test: Optional[bool] = False,
                        skip_wandb: Optional[bool] = False,
                        flash_io: Optional[bool] = False,
                        num_meta_test: Optional[bool] = 10) -> None:
        """
        Perform a meta-test.  It loads new policy from saved model and test on a new environment.
        Each environment created from own seed. So agent seen environment.

        A frequency when to meta test dictated by meta_test_freq configuration in spec.
        Note this value must be larger than save model frequency.

        During meta test trainer load a policy it saved and uses new policy to perform
        a meta test.

        :return: Nothing
        """
        print_green("Starting meta test.")
        test_freq = self.spec.get('meta_test_freq', 'trainer')

        if not is_meta_test:
            if step == 0 or (step % test_freq != 0):
                return
        else:
            step = 0

        if self.tf_writer is None:
            print("Error: tensorboard is none.")
            return

        if self.spec is None:
            print("Error: trainer spec is none.")
            return

        device = self.spec.get("device")
        env = create_env_from_spec(self.spec)

        # new policy
        policy_creator = PolicyCreator(env, self.spec)
        agent_policy, _ = policy_creator()
        agent_policy.to(device)
        linear_baseline = LinearFeatureBaseline(env, device).to(device)
        simulation = RemoteSimulation(20, spec=self.spec,
                                      policy=agent_policy,
                                      baseline=linear_baseline)

        # load policy , note we perform meta test also on target device.
        model_file_name = self.spec.get('model_state_file', 'model_files')
        print(f"Loading model from {model_file_name}")
        with open(model_file_name, 'rb') as f:
            print(f"Loading model from {model_file_name}")
            state_dict = torch.load(f, map_location=torch.device(self.spec.get("device")))
            state_dict.pop("last_step")
            agent_policy.load_state_dict(state_dict)

        try:
            from tqdm.asyncio import trange, tqdm
            tqdm_iter = tqdm(range(1, num_meta_test),
                             desc=f"Meta-test in progress, dev: {self.spec.get('device')},")

            # update tbar
            tqdm_update_dict = {}
            tqdm_iter.set_postfix(tqdm_update_dict)

            if is_meta_test:
                prefix_tasks = "tasks_final_test"
                prefix_task = "task_final_test"
            else:
                prefix_tasks = "tasks_train_test"
                prefix_task = "task_train_meta_test"

            logs = {'tasks': []}
            train_returns, valid_returns = [], []
            async for tqdm_step in tqdm_iter:
                if not is_meta_test:
                    step = tqdm_step

                tasks = await self.agent.sample_tasks()
                meta_task_train, meta_tasks_val = await simulation.meta_tests(tasks)
                _meta_tasks_train = [e.rewards.sum(dim=0) for e in meta_task_train[0]]
                _meta_tasks_val = [e.rewards.sum(dim=0) for e in meta_tasks_val]

                train_returns.append(to_numpy(_meta_tasks_train))
                valid_returns.append(to_numpy(_meta_tasks_val))

                rewards_sum = rewards_std = rewards_mean = total_task = 0
                for meta_task_i, episode in enumerate(meta_tasks_val):
                    trajectory_sum = episode.rewards.sum(dim=0)
                    # print(trajectory_sum.shape)
                    # print(len(tasks))
                    # print(f"task {tasks[meta_task_i]} reward sum {trajectory_sum * (-1) / episode.lengths}")
                    rewards_sum += episode.rewards.sum(dim=0) * (-1) / episode.lengths
                    rewards_mean += episode.rewards.mean(dim=0) * (-1) / episode.lengths
                    rewards_std += episode.rewards.std(dim=0)
                    total_task += 1

                print(torch.sum(rewards_sum) / total_task)
                print(torch.sum(rewards_std) / total_task)
                print(torch.sum(rewards_mean) / total_task)

                # tqdm_update_dict["reward mean"] = format_num((torch.sum(rewards_sum) / total_task).item())
                # tqdm_update_dict["reward sum"] = format_num((torch.sum(rewards_mean) / total_task).item())
                # tqdm_update_dict["reward std"] = format_num((torch.sum(rewards_std) / total_task).item())

                metric_data = {
                    f'{prefix_tasks} reward mean': torch.sum(rewards_sum).item(),
                    f'{prefix_tasks}  reward sum': torch.sum(rewards_mean).item(),
                    f'{prefix_tasks}  reward std': torch.sum(rewards_std).item(),
                    f'{prefix_task} reward mean': (torch.sum(rewards_sum) / total_task).item(),
                    f'{prefix_task} reward sum': (torch.sum(rewards_std) / total_task).item(),
                    f'{prefix_task} std task': (torch.sum(rewards_sum) / total_task).item(),
                    'step': step,
                }

                self.tf_writer.add_scalar(f"{prefix_tasks}/mean", torch.sum(rewards_sum).item(), step)
                self.tf_writer.add_scalar(f"{prefix_tasks}/sum", torch.sum(rewards_mean).item(), step)
                self.tf_writer.add_scalar(f"{prefix_tasks}/std", torch.sum(rewards_std).item(), step)

                self.tf_writer.add_scalar(f"{prefix_task}/mean_task",
                                          (torch.sum(rewards_sum) / total_task).item(), step)
                self.tf_writer.add_scalar(f"{prefix_task}/sum_task",
                                          (torch.sum(rewards_sum) / total_task).item(), step)
                self.tf_writer.add_scalar(f"{prefix_task}/std_task",
                                          (torch.sum(rewards_sum) / total_task).item(), step)

                if skip_wandb and metric_receiver is not None:
                    metric_receiver.update(metric_data)
                tqdm_iter.set_postfix(tqdm_update_dict)

            logs['train_returns'] = np.concatenate(train_returns, axis=0)
            logs['valid_returns'] = np.concatenate(valid_returns, axis=0)

            file_name = self.spec.get("experiment_name")
            with open(f"{file_name}.npz", 'wb') as f:
                np.savez(f, **logs)

            # data = np.load(f"{file_name}.npz")
            # plt.plot(data)
            # plt.show()

        except Exception as err:
            print("Error during meta-test", err)
            traceback.print_exc()
        finally:
            print_green("Finished meta test.")
            if flash_io:
                self.tf_writer.flush()

    async def meta_train(self):
        """ Meta train a model. Meta train create two parallel asyncio, at each batch step
        method create task for agent, the agent distribute policy to all observers.
        Then meta train ask agent to distribute tasks to all observers.
        :return:
        """

        last_saved = 0
        last_trainer_queue = None
        last_metric_queue = None
        trainer_metric_consumers = None
        trainer_episode_consumers = None

        if self.started is False:
            raise ValueError("calling meta_train before trainer started.")

        assert self.tf_writer is not None
        assert self.agent_policy is not None
        assert self.meta_learner is not None
        assert self.agent is not None

        num_batches = self.spec.get('num_batches', 'meta_task')
        metric_receiver = MetricReceiver(num_batches, spec=self.spec)

        try:
            trainer_queue = asyncio.Queue()
            metric_queue = asyncio.Queue()

            last_metric_queue = metric_queue
            last_trainer_queue = trainer_queue

            agent = DistributedAgent(agent_policy=self.agent_policy,
                                     spec=self.spec, world_size=self.world_size)

            # update all worker with agent rref and set fresh policy.
            await agent.broadcast_rref()
            await agent.broadcast_policy()

            from tqdm.asyncio import trange, tqdm
            tqdm_iter = tqdm(range(1, num_batches + 1),
                             desc=f"Training in progress, dev: {self.spec.get('device')},")

            async for episode_step in tqdm_iter:
                trainer_metric_consumers = []
                trainer_episode_consumers = []
                for _ in range(self.world_size - 1):
                    # a task set to each consumer to pass data to algo when observer return data
                    train_data_consumer = asyncio.create_task(agent.trainer_consumer(agent,
                                                                                     trainer_queue,
                                                                                     metric_queue,
                                                                                     self.meta_learner,
                                                                                     device=self.spec.get('device')))

                    # a task set to each consumer to pass data after algo performed computation.
                    metric_collector_task = asyncio.create_task(agent.metric_dequeue(metric_queue,
                                                                                     self.tf_writer,
                                                                                     metric_receiver,
                                                                                     episode_step, tqdm_iter))
                    trainer_episode_consumers.append(train_data_consumer)
                    trainer_metric_consumers.append(metric_collector_task)

                await agent.distribute_tasks(trainer_queue, n_steps=NUM_STEPS)

                # two queue one for episode step and second for metric
                await trainer_queue.join()
                await metric_queue.join()

                for consumer in trainer_episode_consumers:
                    consumer.cancel()
                for consumer in trainer_metric_consumers:
                    consumer.cancel()

                await asyncio.gather(*trainer_episode_consumers, return_exceptions=True)
                await asyncio.gather(*trainer_metric_consumers, return_exceptions=True)

                if agent.save(episode_step):
                    last_saved = episode_step
                self.last_episode = episode_step

                # we perform meta test based on spec to track rewards.
                # this not a final meta test.
                await self.meta_test(episode_step, metric_receiver)

        except KeyboardInterrupt as kb:

            # shutdown metric listener
            metric_receiver.shutdown()
            del metric_receiver

            # close io
            if self.tf_writer is not None:
                self.tf_writer.close()

            if self.agent is not None and self.last_episode > last_saved:
                self.agent.save(last_saved)

            if last_trainer_queue is not None:
                for consumer in trainer_episode_consumers:
                    if not consumer.cancelled():
                        consumer.cancel()
            if last_metric_queue is not None:
                for consumer in trainer_metric_consumers:
                    if not consumer.cancelled():
                        consumer.cancel()
            raise kb

        if metric_receiver is not None:
            metric_receiver.shutdown()
            del metric_receiver

    async def stop(self):
        """
        :return:
        """
        if not self.started:
            raise ValueError("Trainer state already stopped.")

        if self.tf_writer is not None:
            self.tf_writer.close()

        # self.agent.save(self.last_episode)
        # self.started = False
        self.loop.stop()


async def worker(rank: Optional[int] = -1, world_size: Optional[int] = -1, self_logger=None):
    """ Still trying to figure out how to do it with RPC, asyncio logger.
    This is another process so one way and probably right way.  Use asyncio file and write logs.
    in distribute ways.
    TODO
    :param self_logger:
    :param rank:
    :param world_size:
    :return:
    """
    if rank < 0 or world_size < 0:
        raise ValueError("Rank or world must be a positive number.")

    # l ogger = Logger.with_default_handlers(name='meta_critic', level=logging.INFO)
    # self_logger.emit("wroker Test msg")
    # self_logger.emit("worker Test msg two")
    # loop = asyncio.get_event_loop()
    # loop.call_soon_threadsafe(functools.partial(self_logger.emit, "worker test"))
    # await logger.info(f"Observer rank {rank} starting.")


async def rpc_async_worker(rank: int, world_size: int, spec: RunningSpec) -> None:
    """Main logic for rpc async worker,  it crates N agents ,
    each agent N observers
    :param rank: rank of each worker
    :param world_size: world_size number of workers
    :param spec: running spec for a trainer
    :return: Nothing
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29519'
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    worker_name = f"worker{rank}"

    try:
        if rank == 0:
            agent_backend = rpc.TensorPipeRpcBackendOptions(num_worker_threads=16, rpc_timeout=180)
            rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size,
                         rpc_backend_options=agent_backend)
            meta_trainer = DistributedMetaTrainer(spec=spec, world_size=world_size, self_logger=AsyncLogger())

            if spec.is_train():
                await meta_trainer.start()
                await meta_trainer.meta_train()
                await meta_trainer.stop()

            if spec.is_test():
                await meta_trainer.start()
                num_batches = spec.get('num_batches', 'meta_task')
                await meta_trainer.meta_test(step=0, skip_wandb=True, is_meta_test=True)
                await meta_trainer.stop()

        else:
            observer_backend = rpc.TensorPipeRpcBackendOptions(num_worker_threads=16, rpc_timeout=180)
            rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size,
                         rpc_backend_options=observer_backend)
            await worker(rank=rank, world_size=world_size, self_logger=AsyncLogger())

        # await logger.
        print(f"Shutdown down {worker_name}")
        rpc.shutdown(graceful=True)
    except FileNotFoundError as file_not_found:
        print_red(str(file_not_found))
        rpc.shutdown(graceful=True)
    except KeyboardInterrupt as kb:
        rpc.shutdown(graceful=True)
        ask_exit()
        raise kb
    except Exception as other_exp:
        print(other_exp)
        print(traceback.print_exc())


def ask_exit():
    """
    :return:
    """
    for task in asyncio.all_tasks():
        task.cancel()


def run_worker(rank: int, world_size: int, spec: RunningSpec):
    """
    Main entry called by main routine
    :param rank:
    :param world_size:
    :param spec:
    :return:
    """
    if spec is None:
        print_red("Running spec is empty")
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # for sig in (signal.SIGINT, signal.SIGTERM):
        #     loop.add_signal_handler(sig, ask_exit)
        loop.run_until_complete(rpc_async_worker(rank, world_size, spec))
    except SystemExit as sys_exit:
        print("SystemExit")
        ask_exit()
        raise sys_exit
    except Exception as loop_err:
        print(loop_err)
        raise loop_err
    finally:
        if loop is not None and loop.is_running():
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
