import json
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from meta_critics.running_spec import RunningSpec
from meta_critics.app_globals import AppSelector
from meta_critics.trainer import MetaTrainer


class AppError(Exception):
    """Base class for other exceptions"""
    pass


class App:
    def __init__(self, running_spec: RunningSpec, mode: AppSelector):
        """

        :param running_spec:
        :param mode:
        """
        self.trainer = None
        self._running_spec = running_spec

        # app mode train / test etc.
        self._mode = mode
        #
        self._env = None
        #
        self._sampler = None
        self._policy = None
        self._device = None
        self._inference_result_file = None

    def main(self, ctx):
        """
        main logic of app
        :return:
        """
        if self._running_spec.is_train():
            print("Training model.")
            self._create_env()
            self.trainer = MetaTrainer(ctx, self._env, self._running_spec)
            self.trainer.meta_train()

        if self._is_test():
            print("Testing model.")
            self._create_env()
            self.trainer = MetaTrainer(ctx, self._env, self._running_spec)
            self.trainer.meta_test()


    def _is_test(self):
        return self._mode == AppSelector.TestModel \
               or self._mode == AppSelector.TrainTestModel \
               or self._mode == AppSelector.TrainTestPlotModel

    def _is_train(self):
        return self._mode == AppSelector.TranModel \
               or self._mode == AppSelector.TrainTestModel \
               or self._mode == AppSelector.TrainTestPlotModel

    def _is_plot(self):
        return self._mode == AppSelector.PlotModel \
               or self._mode == AppSelector.TrainTestPlotModel

    def _create_env(self) -> bool:
        """Create environment.
        :return: True if environment created.
        """
        self._env_args = None
        if hasattr(self._running_spec, 'env_args'):
            self._env_args = self._running_spec.env_args

        if hasattr(self._running_spec, 'env_name'):
            print(f"Creating environment {self._running_spec.env_name}")
            if self._env_args is None:
                self._env = gym.make(self._running_spec.env_name)
            else:
                self._env = gym.make(self._running_spec.env_name, **self._env_args)
            self._env.close()

        return True

    def _save_model(self):
        """
        :return:
        """
        print("##### Saving")
        with open(self._running_spec.get("model_state_file", root="model_files"), 'wb') as f:
            torch.save(self._policy.state_dict(), f)
