#
#  Now let add async from agent
#  Mus
import asyncio
import os
import traceback
from pathlib import Path
from typing import Optional

import torch
import torch.distributed.rpc as rpc
from torch.utils.tensorboard import SummaryWriter

from meta_critics.ioutil.term_util import print_red
from meta_critics.modules.baseline import LinearFeatureBaseline
from meta_critics.policies.policy_creator import PolicyCreator
from meta_critics.models.async_trpo import ConcurrentMamlTRPO
from meta_critics.rpc.async_logger import AsyncLogger
from meta_critics.rpc.rpc_agent import DistributedAgent
from meta_critics.running_spec import RunningSpec
from meta_critics.simulation import RemoteSimulation
from util import create_env_from_spec

# import wandb
# wandb.init(project="dh-maml", entity="spyroot")
from aiologger import Logger

logger = Logger.with_default_handlers()

NUM_STEPS = 1
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"


# logging.basicConfig(level=logging.DEBUG, format="%(message)s")


def async_gather(coroutines):
    event_loop = asyncio.get_event_loop()
    coroutine = asyncio.gather(*coroutines)
    return zip(*event_loop.run_until_complete(coroutine))


def resole_primary_dir(path_to_dir: str, create_if_needed: Optional[bool] = False) -> str:
    """
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
            os.makedirs(log_file_path)
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
        self.agent = DistributedAgent(agent_policy=self.agent_policy,
                                      spec=self.spec,
                                      world_size=self.world_size,
                                      self_logger=self.self_logger)

        self.started = True
        self.log(f"DistributedMetaTrainer world size{self.world_size} started.")

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
                    print(f"Detected existing model. Loading from {model_file_name} from {last_step}.")
                else:
                    print(f"Detected existing model. Loading from {model_file_name}.")

                self.agent_policy.load_state_dict(state_dict)

    async def meta_test(self, step, flash_io=False) -> None:
        """
        Perform a meta-test.  Load new policy from saved model and test on a new environment.
        frequency when to meta test dictated by meta_test_freq configuration in spec.
        Note this value must large then save model frequency.

        :return: Nothing
        """
        test_freq = self.spec.get('meta_test_freq', 'trainer')

        if step == 0:
            return

        if step % test_freq != 0:
            return

        if self.tf_writer is None:
            print("Error: tensorboard is none.")
            return

        if self.spec is None:
            print("Error: trainer spec is none.")
            return

        device = self.spec.get("device")
        env = create_env_from_spec(self.spec)

        policy_creator = PolicyCreator(env, self.spec)
        agent_policy, _ = policy_creator()
        linear_baseline = LinearFeatureBaseline(env, device).to(device)

        simulation = RemoteSimulation(1, spec=self.spec,
                                      policy=agent_policy,
                                      baseline=linear_baseline)

        model_file_name = self.spec.get('model_state_file', 'model_files')
        with open(model_file_name, 'rb') as f:
            print(f"Loading model from {model_file_name}")
            state_dict = torch.load(f, map_location=torch.device(self.spec.get("device")))
            state_dict.pop("last_step")
            agent_policy.load_state_dict(state_dict)

        # fcntl.fcntl(sys.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

        try:
            from tqdm.asyncio import trange, tqdm
            tqdm_iter = tqdm(range(1, 10),
                             desc=f"Meta-test in progress, dev: {self.spec.get('device')},")

            # update tbar
            tqdm_update_dict = {}
            tqdm_iter.set_postfix(tqdm_update_dict)

            async for _ in tqdm_iter:
                rewards = []
                tasks = await self.agent.sample_tasks()
                _, meta_tasks = await simulation.meta_tests(tasks)

                for meta_task_i, episode in enumerate(meta_tasks):
                    rewards_sum = episode.rewards.sum().cpu().item()
                    rewards_mean = episode.rewards.mean().cpu().item()
                    rewards_std = episode.rewards.std().cpu().item()
                    # await logger.info(episode.rewards.cpu())
                    tqdm_update_dict["reward mean"] = rewards_mean
                    tqdm_update_dict["reward sum"] = rewards_sum
                    tqdm_update_dict["reward std"] = rewards_std
                    tqdm_update_dict["task"] = tasks[meta_task_i]

                    self.tf_writer.add_scalar(f"meta_test/mean", rewards_mean, step)
                    self.tf_writer.add_scalar(f"meta_test/sum", rewards_sum, step)
                    self.tf_writer.add_scalar(f"meta_test/std", rewards_std, step)
                tqdm_iter.set_postfix(tqdm_update_dict)
        except Exception as err:
            print("Error during meta-test", err)
        finally:
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

        if self.started is False:
            raise ValueError("calling meta_train before trainer started.")

        assert self.tf_writer is not None
        assert self.agent_policy is not None
        assert self.meta_learner is not None
        assert self.agent is not None

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
            num_batches = self.spec.get('num_batches', 'meta_task')
            tqdm_iter = tqdm(range(1, num_batches),
                             desc=f"Training in progress, dev: {self.spec.get('device')},")

            async for episode_step in tqdm_iter:
                trainer_metric_consumers = []
                trainer_episode_consumers = []
                for _ in range(self.world_size - 1):
                    train_data_consumer = asyncio.create_task(agent.trainer_consumer(agent,
                                                                                     trainer_queue,
                                                                                     metric_queue,
                                                                                     self.meta_learner,
                                                                                     device=self.spec.get('device')))
                    metric_collector_task = \
                        asyncio.create_task(agent.metric_dequeue(metric_queue, self.tf_writer,
                                                                 episode_step, tqdm_iter))
                    trainer_episode_consumers.append(train_data_consumer)
                    trainer_metric_consumers.append(metric_collector_task)

                await agent.distribute_tasks(trainer_queue, n_steps=NUM_STEPS)

                # two queue one for episode and second for metric
                await trainer_queue.join()
                await metric_queue.join()

                for consumer in trainer_episode_consumers:
                    consumer.cancel()
                for consumer in trainer_metric_consumers:
                    consumer.cancel()

                await asyncio.gather(*trainer_episode_consumers, return_exceptions=True)
                await asyncio.gather(*trainer_metric_consumers, return_exceptions=True)

                # result = await self.loop.run_in_executor(
                #         None, blocking_io)

                if agent.save(episode_step):
                    last_saved = episode_step
                self.last_episode = episode_step

                await self.meta_test(episode_step)

        except KeyboardInterrupt as kb:
            if self.tf_writer is not None:
                self.tf_writer.close()
            if self.agent is not None and self.last_episode > last_saved:
                self.agent.save(last_saved)
            if last_trainer_queue is not None:
                for consumer in last_trainer_queue:
                    consumer.cancel()
            if last_metric_queue is not None:
                for consumer in last_metric_queue:
                    consumer.cancel()
            raise kb

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
    """

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


async def rpc_async_worker(rank, world_size, spec: RunningSpec):
    """

    :param rank:
    :param world_size:
    :param spec:
    :return:
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29517'
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    worker_name = f"worker{rank}"

    try:
        if rank == 0:
            rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)
            meta_trainer = DistributedMetaTrainer(spec=spec, world_size=world_size, self_logger=AsyncLogger())
            await meta_trainer.start()
            await meta_trainer.meta_train()
            await meta_trainer.stop()
            # await logger.shutdown()
        else:
            # TODO here torch has issue parent might be already dead.
            # I don't import multiprocessing lib since it lead to many other issue.
            # One idea register interrupt handler and handle signals
            logger = AsyncLogger()
            # logger = Logger.with_default_handlers(name='meta_critic', level=logging.INFO)
            rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)
            await worker(rank=rank, world_size=world_size, self_logger=logger)
        # await logger.
        print(f"Shutdown down {worker_name}")
        rpc.shutdown()
    except FileNotFoundError as file_not_found:
        print_red(str(file_not_found))
        rpc.shutdown()
    except KeyboardInterrupt as kb:
        rpc.shutdown()
        raise kb
    except Exception as other_exp:
        print(other_exp)
        print(traceback.print_exc())


def run_worker(rank, world_size, spec: RunningSpec):
    """

    :param rank:
    :param world_size:
    :param spec:
    :return:
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(rpc_async_worker(rank, world_size, spec))
        # loop.close()
    except Exception as loop_err:
        print(loop_err)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
