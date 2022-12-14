import os
from enum import Enum
from typing import Optional
import numpy as np

# todo remove this and add generic logger
from loguru import logger
from .call_interface import Callback
from .const import ReduceMode


class CheckpointBest(Callback):
    """
    Save best model every epoch based on loss
    Args:
        save_dir (str): path to folder where to save the model

        save_name (str): name of the saved model. can additionally
            add epoch and metric to model save name

        monitor (str): quantity to monitor. Implicitly prefers validation metrics over train. One of:
            `loss` or name of any metric passed to the runner.

        mode (str): one of "min" of "max". Whether to decide to save based
            on minimizing or maximizing loss

        This increases checkpoint size 2x times.
        verbose (bool): If `True` reports each time new best is found
    """

    def __init__(self,
                 save_dir: Optional[str] = None,
                 save_name: Optional[str] = "model_{ep}_{metric:.2f}.chpn",
                 monitor: Optional[str] = "loss", mode: Optional[str] = "min", verbose=True) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.save_name = save_name
        self.monitor = monitor

        mode = ReduceMode(mode)
        if mode == ReduceMode.MIN:
            self.best = np.inf
            self.monitor_callback = np.less
        elif mode == ReduceMode.MAX:
            self.best = -np.inf
            self.monitor_callback = np.greater
        self.verbose = verbose

    def on_begin(self):
        """
        :return:
        """
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self) -> None:
        """
        Called at the end of epoch.  If value that callback monitor changed
        based on monitor predicated.
        :return:
        """
        current = self.get_monitor_value()
        current_epoch = self.trainer.state.epoch
        if self.monitor_callback(current, self.best):
            if self.verbose:
                print("")
                print(f"Epoch {current_epoch:2d}: best {self.monitor} "
                      f"improved from {self.best:.4f} to {current:.4f}")
                logger.info(f"Epoch {current_epoch:2d}: best {self.monitor} "
                            f"improved from {self.best:.4f} to {current:.4f}")

            self.best = current
            self.trainer.save()

    def _save_checkpoint(self, path):
        """
        :param path:
        :return:
        """
        self.trainer.save()

    def get_monitor_value(self):
        """
        :return:
        """
        value = None
        if self.monitor == "loss":
            return self.metric.epoch_average_loss()
        else:
            value = self.metric.get_metric_value(self.monitor)

        if value is None:
            raise ValueError(f"CheckpointSaver can't find {self.monitor} value to monitor.")

        return value
