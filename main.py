"""
Note main use mp.span hence it will fork to separate pids.
Mus
"""
import argparse
import os
import sys

import torch
import torch.multiprocessing as mp

from meta_critics.app_globals import get_running_mode, SpecTypes
from meta_critics.running_spec import RunningSpecError, RunningSpec
from meta_critics.rpc.rpc_trainer import run_worker


def main(cmd, spec):
    """
    :param cmd:
    :param spec:
    :return:
    """
    for world_size in range(cmd.workers, cmd.workers + 1):
        mp.spawn(run_worker, args=(world_size, spec), nprocs=world_size, join=True)
    print("All main done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unit test for threaded observer")
    parser.add_argument('--tune', action='store_true', required=False, help='run ray hyperparameter optimization.')
    parser.add_argument('--test', action='store_true', required=False, help="train model for task")
    parser.add_argument('--plot', action='store_true', required=False, help="test model on task")
    parser.add_argument('--train', action='store_true', required=False, help="plots test result")
    parser.add_argument('--benchmark', action='store_true', required=False, help="will measure "
                                                                                 "time of execution for different "
                                                                                 "number of threads")

    parser.add_argument('--use-cpu', action='store_true', help='if we want enforce cpu only.')
    parser.add_argument('--config', type=str, required=True, help="a path to the configuration json or yaml file.")
    parser.add_argument('--model_file', type=str, required=False, default="default.th",
                        help="a path to the a model file.")
    parser.add_argument('--is_verbose', action='store_true', required=False, help="Enable verbose out during test")

    trainer = parser.add_argument_group('trainer')
    trainer.add_argument('--num_batches', type=int, default=10,
                         help="number of batches. Default 500.")
    trainer.add_argument('--num_meta_test', type=int, default=10,
                         help="number of meta test batch perform. Default 10")
    trainer.add_argument('--num_meta_task', type=int, default=40,
                         help="number of tasks per batch Default: 40)")
    trainer.add_argument('--num_trajectory', type=int, default=20,
                         help="number of trajectory per task collect. Default 20")

    meta_test = parser.add_argument_group('Meta-Test')
    parser.add_argument('--meta_test_freq', type=int, default=20, required=False,
                        help="A frequency when we want to test")

    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--config_type', type=SpecTypes, default=SpecTypes.JSON, help='config file type.')
    misc.add_argument('--model_dir', type=str, required=False, help='a directory where we will model data.')
    misc.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    misc.add_argument('--debug_agent', action='store_true', required=False, help='Enables debug for agent.')
    misc.add_argument('--debug_task_sampler', action='store_true', required=False, help='Enables debug environment.')
    misc.add_argument('--debug_env', action='store_true', required=False, help='Enables debug environment.')
    misc.add_argument('--gamma', type=float, default=1.0, metavar='G', help='discount factor (default: 1.0)')
    misc.add_argument('--workers', type=int, default=2, help='Number of workers minimum 2. Worker 1 main Agent.')
    misc.add_argument('--num_worker_threads', type=int, default=16, help='Number of workers threads.')
    misc.add_argument('--rpc_timeout', type=int, default=180, help='RPC timeout settings.')
    misc.add_argument('--rpc_port', type=str, default="29519", help='default rpc port.')
    misc.add_argument('--disable_wandb', action='store_true', required=False, help="plots test result")

    misc = parser.add_argument_group('Miscellaneous')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()) else 'cpu')

    # torch.manual_seed(args.seed)
    if args.use_cpu:
        args.device = 'cpu'

    mode = get_running_mode(args)
    if mode is None:
        print("Please select either train/test/plot")
        sys.exit(1)

    try:
        current_dir = os.getcwd()
        running_spec = RunningSpec(args, mode, current_dir)

    except RunningSpecError as r_except:
        print(f"Error:", r_except)
        sys.exit(100)

    main(args, running_spec)
