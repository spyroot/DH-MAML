# old main
# meta_critics
import argparse
import os
import sys
from multiprocessing import freeze_support

import torch

from meta_critics.app_globals import get_running_mode, SpecTypes
from meta_critics.running_spec import RunningSpec, RunningSpecError


# ctx = torch.multiprocessing.get_context('spawn')
def main(cmd):
    """"
    :param args:
    :return:
    """
    mode = get_running_mode(cmd)
    if mode is None:
        print("Please select either train/test/plot")
        sys.exit(1)

    # if args.seed is not None:
    # print("Setting fixed seed for torch.")
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)

    try:
        # with Manager() as manager:
        current_dir = os.getcwd()
        running_spec = RunningSpec(args, mode, current_dir)
        App(running_spec, mode).main()
    except RunningSpecError as r_except:
        print(f"Error {r_except}")
        sys.exit(100)


if __name__ == '__main__':
    freeze_support()

    # ctx = torch.multiprocessing.get_context('spawn')
    # import meta_critics.multi.ctx import ctx
    # ctx =
    # #try:
    #    torch.multiprocessing.get_context('spawn')
    # except RuntimeError as e:
    #    print("Error", e)
    #    pass

    parser = argparse.ArgumentParser(description="Reinforcement learning with "
                                                 "Model-Agnostic Meta-Learning (MAML) - Test")

    parser.add_argument('--tune', action='store_true', required=False, help='run ray hyperparameter optimization.')
    parser.add_argument('--test', action='store_true', required=False, help="train model for task")
    parser.add_argument('--plot', action='store_true', required=False, help="test model on task")
    parser.add_argument('--train', action='store_true', required=False, help="plots test result")
    parser.add_argument('--use-cpu', action='store_true', help='if we want enforce cpu only.')

    parser.add_argument('--config', type=str, required=True, help="a path to the configuration json or yaml file.")
    parser.add_argument('--model_file', type=str, required=False, default="default.th",
                        help="a path to the a model file.")

    # Evaluation
    evaluation = parser.add_argument_group('Evaluation')
    evaluation.add_argument('--num_batches', type=int, default=10, help="number of batches (default: 10)")
    evaluation.add_argument('--num_meta_task', type=int, default=40, help="number of tasks per batch (default: 40)")

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    parser.add_argument('--config_type', type=SpecTypes, default=SpecTypes.JSON, help='config file type.')
    misc.add_argument('--model_dir', type=str, required=False, help='a directory where we store model data.')
    misc.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--debug_agent', action='store_true', required=False, help='Enables debug for agent.')
    parser.add_argument('--debug_env', action='store_true', required=False, help='Enables debug environment.')
    parser.add_argument('--debug_task_sampler', action='store_true', required=False, help='Enables debug environment.')

    # #
    # misc.add_argument('--num-workers', type=int, default=ctx.cpu_count() - 1,
    #                   help='number of workers. (default: {0})'.format(ctx.cpu_count() - 1))

    misc.add_argument('--num-workers', type=int, default=1,
                      help='')
    # misc.add_argument('--use-cuda', action='store_true', help='use cuda (default: false, use cpu). '
    #                                                           'WARNING: Full support for cuda is not guaranteed. '
    #                                                           'Using CPU is encouraged.')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()) else 'cpu')

    if args.use_cpu:
        args.device = 'cpu'

    try:
        main(args, ctx)
    except FileNotFoundError as file_error:
        print("File not found ", str(file_error))
    except AppError as spec_error:
        print("Invalid spec:", str(spec_error))
        # logger.error(f"Invalid spec: {str(spec_error)}")
        sys.exit(2)
