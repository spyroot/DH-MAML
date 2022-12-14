"""
This main running config for a trainer.

Mus
"""
from enum import Enum, auto


class SpecTypes(Enum):
    JSON = auto()
    YAML = auto()


class AppSelector(Enum):
    """
    Data type enum,  if we need use only torch or numpy to avoid
    changing data types.
    TODO evaluate option to compute everything on GPU.
    """
    TestModel = auto()
    TranModel = auto()
    PlotModel = auto()
    TrainTestModel = auto()
    TrainTestPlotModel = auto()
    CheckSpec = auto()


def get_running_mode(args) -> AppSelector:
    """
    :param args:
    :return:
    """
    mode = None
    if args.check_specs:
        mode = AppSelector.CheckSpec
    if args.test:
        mode = AppSelector.TestModel
    if args.train:
        mode = AppSelector.TranModel
    if args.plot:
        mode = AppSelector.PlotModel
    if args.test and args.train:
        mode = AppSelector.TrainTestModel
    if args.test and args.train and args.plot:
        mode = AppSelector.TrainTestPlotModel

    return mode
