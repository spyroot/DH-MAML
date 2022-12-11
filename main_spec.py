import argparse
import os
import sys

import torch

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
    print("All main done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unit test for threaded observer")
    parser.add_argument('--tune', action='store_true', required=False, help='run ray hyperparameter optimization.')
    parser.add_argument('--test', action='store_true', required=False, help="train model for task")
    parser.add_argument('--plot', action='store_true', required=False, help="test model on task")
    parser.add_argument('--train', action='store_true', required=False, help="plots test result")
    parser.add_argument('--use-cpu', action='store_true', help='if we want enforce cpu only.')
    parser.add_argument('--config', type=str, required=True, help="a path to the configuration json or yaml file.")
    parser.add_argument('--model_file', type=str, required=False, default="default.th",
                        help="a path to the a model file.")

    trainer = parser.add_argument_group('trainer')
    trainer.add_argument('--num_batches', type=int, default=10,
                         help="number of batches. Default 500.")
    trainer.add_argument('--num_meta_task', type=int, default=40,
                         help="number of tasks per batch Default: 40)")
    trainer.add_argument('--num_trajectory', type=int, default=20,
                         help="number of trajectory per task collect. Default 20")

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    parser.add_argument('--config_type', type=SpecTypes, default=SpecTypes.JSON, help='config file type.')
    misc.add_argument('--model_dir', type=str, required=False, help='a directory where we will model data.')
    misc.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--debug_agent', action='store_true', required=False, help='Enables debug for agent.')
    parser.add_argument('--debug_env', action='store_true', required=False, help='Enables debug environment.')
    parser.add_argument('--debug_task_sampler', action='store_true', required=False, help='Enables debug environment.')
    parser.add_argument('--gamma', type=float, default=1.0, metavar='G', help='discount factor (default: 1.0)')
    misc.add_argument('--workers', type=int, default=2, help='Number of workers minimum 2. Worker 1 main Agent.')
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
        s = running_spec.as_dict()
        print(s)

    except RunningSpecError as r_except:
        print(f"Error:", r_except)
        sys.exit(100)

    main(args, running_spec)
