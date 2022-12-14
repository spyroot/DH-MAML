"""
This running state of trainer.  It will register all config as attributes  so technically each
could be overwritten from shell.
"""
import argparse
import json
import os
import warnings
from collections.abc import MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import yaml

from meta_critics.app_globals import AppSelector, SpecTypes


class RunningSpecError(Exception):
    """Base class for other exceptions"""
    pass


class RunningSpec:
    # all mandatory keys in config.
    mandatory_keys = ['num_batches', 'num_meta_test', 'num_meta_task', 'num_trajectory', 'meta_test_freq',
                      'gamma', 'workers', 'experiment_name', 'env_name', 'wandb.entity', 'wandb.project',
                      'trajectory_sampler.remap_types',
                      'model_root', 'log_dir', 'create_dir', 'trainer.use_wandb', 'trainer.type', 'trainer.save_freq',
                      'trainer.meta_test_freq', 'trainer.use_gae', 'trainer.gamma_factor', 'trainer.gae_lambda_factor',
                      'trainer.use_discount_gamma', 'trainer.resume', 'trainer.num-workers',
                      'policy_network.hidden_sizes', 'policy_network.activation', 'meta_task.num_batches',
                      'meta_task.num_meta_task',
                      'meta_task.num_trajectory', 'meta_task.num_steps', 'meta_task.fast_lr', 'meta_task.first_order',
                      'model.name', 'model.model_type', 'model.max_kl', 'model.cg_iters', 'model.cg_damping',
                      'model.ls_max_steps', 'model.ls_backtrack_ratio'
                      ]

    def __init__(self, settings, mode, current_dir: Optional[str] = "."):
        """

        :param settings:  setting a dict or namespace from args parser.
        :param mode:  a current mode train / test etc.
        :param current_dir:  a current dir used to resolve path to a /config/ directory.
        """
        self._debug = False
        self._mode = mode
        self._settings = settings
        self._current_dir = current_dir

        self._model_dir = None
        self._train_config = None
        self._test_config = None
        self._running_config = None
        self._read_spec()

    def _resolve_model_dir(self) -> str:
        """Resolve model directory.
        :return:
        """
        if self._model_dir is not None:
            return self._model_dir

        if 'model_dir' in self._settings and self._settings.model_dir is not None:
            p = Path(self._settings.model_dir.strip()).expanduser().resolve()
            if p.is_file():
                raise RunningSpecError(f"{str(p)} is a file. --model_dir must a directory.")
            self._model_dir = str(p)
            return self._model_dir

    def _resolve_config_file(self) -> str:
        """ Resolves full path to a config file and updates
        internal state.
        :return: Nothing. Will raise exception if 'config' key not present.
        """
        if "config" not in self._settings:
            raise RunningSpecError("Setting must contain path to a config file.")

        p = Path(self._settings.config).expanduser()
        #p = p.resolve()
        if not p.is_file():
            raise RunningSpecError(f"Resolve_config_file can't find {self._settings.config}, "
                                   f"please provide a full path.")
        else:
            self._settings.config = str(p)

    def _resolve_model_files(self, suffix="") -> None:
        """Resolve all paths to all model files, if prefix indicate will add
        :param suffix:
        :return:
        """
        # model rooted
        p = Path(self._model_dir)
        # model and optimizer
        self.update("model_state_file", str(p / "model_state.th"), root="model_files")
        self.update("model_opt_state_file", str(p / "model_opt_state.th"), root="model_files")
        # metric and log traces
        self.update("metric_file", str(p / "metric.json"), root="model_files")
        self.update("trace_file", str(p / "model_trace.log"), root="model_files")
        # tensorboard and hyperparameter files.
        self.update("tensorboard_dir", str(p / "tensorboard"), root="model_files")
        self.update("hp_file", str(p / "hp.json"), root="model_files")
        # debug and performance log files
        self.update("performances_file", str(p / "performances.log"), root="model_files")

    def _read_spec(self) -> None:
        """Main entry point for config runner.
        :return:
        """
        # resolve mandatory attributes
        self._resolve_model_dir()
        self._resolve_config_file()

        # load all specs
        self._load_model_spec()
        self._load_model_dir()
        # resolve all model files
        self._resolve_model_files()

    @staticmethod
    def load_json(_filename: str, is_strict: Optional[bool] = False):
        """Load from json a spec file.
        :param is_strict: will throw error.
        :param _filename: path to a file.
        :return:
        """
        if _filename is None or len(_filename.strip()) == 0:
            raise RunningSpecError("Invalid file name.")

        _filename = _filename.strip()
        try:
            with open(_filename, 'r') as f:
                output_dict = json.load(f)
                return output_dict
        except json.decoder.JSONDecodeError as j_err:
            if is_strict:
                print(f"Check file, json file {_filename} has error. ", str(j_err))
                raise j_err
            return None

    @staticmethod
    def load_yaml(_filename: str, is_strict: Optional[bool] = False):
        """Load config file from a yaml file.
        :param is_strict: will throw error.
        :param _filename: path to a file.
        :return: config file.
        """
        if _filename is None or len(_filename.strip()) == 0:
            raise RunningSpecError("Invalid file name")

        _filename = _filename.strip()
        try:
            with open(_filename, 'r') as f:
                output_dict = yaml.load(f, Loader=yaml.FullLoader)
                return output_dict
        except yaml.YAMLError as y_err:
            if is_strict:
                print("Check file with yaml linter it has error. ", str(y_err))
                raise y_err
            return None

    def _load_model_dir(self):
        """ Read model directory.
        :return:
        """
        if self._model_dir is not None:
            p = Path(self._model_dir.strip()).expanduser()
        else:
            if 'experiment_name' not in self._running_config:
                raise RunningSpecError("Neither \"experiment_name\" nor a \"model_dir\" argument specified.")
            if self._current_dir is None:
                raise RunningSpecError("Current directory unknown.")

            p = Path(self._current_dir).expanduser() / self._moder_root / self.experiment_name

        p = p.resolve()

        if p.is_file():
            raise FileExistsError(f"{self._model_dir} is a file.")

        if os.access(self._current_dir, os.W_OK) is not True:
            RunningSpecError(f"You don't have access to write {self._current_dir}.")

        if not os.path.exists(p):
            print(f"Creating directory {self._moder_dir} for a model.")
            os.makedirs(p, exist_ok=True)

        # register internal attr
        self._model_dir = str(p)
        self.update('model_dir', self._model_dir)
        return True

    def get_model_dir(self):
        return self._model_dir

    def _load_model_spec(self):
        """Load model configuration files to running state.
        :return:
        """
        test_config = None
        train_config = None

        if (self.is_test() or self.is_plot()) and not self.is_train():
            if SpecTypes.JSON == self._settings.config_type:
                test_config = self.load_json(self._settings.config)

            if SpecTypes.YAML == self._settings.config_type or test_config is None:
                test_config = self.load_yaml(self._settings.config)
            else:
                raise RunningSpecError("Unknown config file format. (use yaml or json)")

            if test_config is None:
                raise RunningSpecError("Failed to parse config file.")

            # update running config from args first
            # and then overwrite from spec
            self.update_running_config(self._settings)
            self.update_running_config(test_config)

        if self.is_train():
            if SpecTypes.JSON == self._settings.config_type:
                train_config = self.load_json(self._settings.config)

            if train_config is None or SpecTypes.YAML == self._settings.config_type:
                train_config = self.load_yaml(self._settings.config)
            else:
                raise RunningSpecError(f"Unknown {self._settings.config} config file format. (use yaml or json)")

            if train_config is None:
                raise RunningSpecError("Failed to parse config file.")

            # update running config from args first
            # and then overwrite from spec
            self.update_running_config(self._settings)
            self.update_running_config(train_config)

        if self.contains("model_root"):
            self._moder_root = self.get("model_root")
            self.update('model_root', self.get("model_root"))

        # self.update_running_config(test_config)
        self._test_config = test_config
        self._train_config = train_config

    def _model_plot(self):
        """
        :return:
        """
        p = Path(self._get_spec_val("model_dir")).expanduser()
        p = p.resolve()
        data = np.load(p / self._get_spec_val("task_inference_file"))
        plt.plot(data)
        plt.show()

    def _get_spec_val(self, k: str, root: Optional[str] = None):
        """
        :param k:
        :return:
        """
        _config = self._running_config
        if root is not None:
            if root not in self._running_config:
                msg = f"{root} configuration not present in current spec."
                raise RunningSpecError(msg)
            _config = self._running_config[root]

        k = k.lower().strip().replace("-", "_")
        if k in _config:
            return _config[k]
        else:
            msg = f"Parameter {k} is not present in current {root} running config."
            raise RunningSpecError(msg)

    def show(self):
        """Show configuration file.
        :return:
        """
        if self._running_config is None:
            warnings.warn("Error running config is empty.")
            return
        return yaml.dump(self._running_config)

    def update(self, k: str, val, root: Optional[str] = None) -> bool:
        """Updates spec and register all spec key as attributes.
        :param root:
        :param self:
        :param k:
        :param val:
        :return:
        """
        if root is None:
            self._running_config[k] = val.strip()
            if isinstance(val, str):
                setattr(self, k, deepcopy(val.strip()))
            else:
                setattr(self, k, val)
            return True

        if root is not None:
            if root not in self._running_config:
                self._running_config[root] = {}
                setattr(self, root, {})
            section = self._running_config[root]
            if isinstance(val, str):
                section[k] = val.strip()
            else:
                section[k] = val

    def _update(self, from_argparse: argparse.Namespace):

        for k, v in self._running_config.items():
            self._running_config

    def update_running_config(self, config=None):
        """Updates running config it takes either generic dict or settings from args.
        Each config registered via setattr.
        :return: Nothing.
        """
        if self._running_config is None:
            self._running_config = {}

        if config is not None:
            if isinstance(config, argparse.Namespace):
                self._running_config.update(vars(self._settings))
            else:
                self._running_config.update(config)
        else:
            if self._settings is not None \
                    and isinstance(self._settings, argparse.Namespace):
                self._running_config.update(vars(self._settings))

        # don't serialize
        if self._running_config is not None and \
                'config_type' in self._running_config:
            self._running_config.pop('config_type')

        # register all config as attributes
        for k in self._running_config:
            new_key = k.lower().strip().replace("-", "_")
            setattr(self, new_key, self._running_config[k])

    def is_test(self):
        return self._mode == AppSelector.TestModel \
            or self._mode == AppSelector.TrainTestModel \
            or self._mode == AppSelector.TrainTestPlotModel

    def is_train(self):
        return self._mode == AppSelector.TranModel \
            or self._mode == AppSelector.TrainTestModel \
            or self._mode == AppSelector.TrainTestPlotModel

    def is_plot(self):
        return self._mode == AppSelector.PlotModel \
            or self._mode == AppSelector.TrainTestPlotModel

    def __str__(self):
        return str(self.show())

    def get(self, param, root: Optional[str] = None):
        if self._debug:
            print(f"Get value {param} {root}")
        return self._get_spec_val(param, root)

    def contains(self, k: str, root: Optional[str] = None):
        """Return true if spec has a given section
        :param root:  a root section of config.
        :param k:
        :return: If spec contains a givne key.
        """
        _config_section = self._running_config
        if root is not None:
            if root not in self._running_config:
                return False
            _config_section = self._running_config[root]

        k = k.lower().strip().replace("-", "_")
        if _config_section is None:
            raise RunningSpecError("Configuration file is none.")

        if k in _config_section:
            return True

        return False

    def _check_skip(self, skip_words: List[str], k: str) -> bool:
        for w in skip_words:
            if w in k:
                return False
        return True

    def _as_dict(self, _iter, _final_dict, skip_words):
        """
        :param _iter:
        :param _final_dict:
        :return:
        """
        for k, v in _iter.items():
            if isinstance(v, dict):
                self._as_dict(v, _final_dict, skip_words)
            else:
                if self._check_skip(skip_words, k):
                    _final_dict[k] = v
                    continue

    def as_dict(self, skip_config: Optional[List[str]] = ['debug', 'file', 'dir', 'train', 'tune', 'plot']):
        """ Filter config and output only dict with key that we need, wandb for example
        :return:
        """
        _final_dict = {}
        self._as_dict(self._running_config, _final_dict, skip_config)
        return _final_dict

    def flatten_running_config(self):
        """ Return running config as flatten dict.
        :return:
        """
        if self._running_config is None:
            return {}
        return self._flatten_dict(self._running_config)

    @staticmethod
    def _flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.'):
        """ Flatten config
        :param d:
        :param parent_key:
        :param sep:
        :return:
        """
        return dict(RunningSpec._flatten_dict_gen(d, parent_key, sep))

    @staticmethod
    def _flatten_dict_gen(d, parent_key, sep):
        """ Internal
        :param d:
        :param parent_key:
        :param sep:
        :return:
        """
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                yield from RunningSpec._flatten_dict(v, new_key, sep=sep).items()
            else:
                yield new_key, v

    def check_running_config(self):
        """ Filter config and output only dict with key that we need, wandb for example
        :return:
        """
        current_keys = self.flatten_running_config()
        for k in self.mandatory_keys:
            if current_keys.get(k) is None:
                raise RunningSpecError(f"Please make sure spec contains {k} configuration.")
        return True

    @staticmethod
    def check_spec_file(path_to_file):
        """ Filter config and output only dict with key that we need, wandb for example
        :return:
        """
        spec_dict = RunningSpec.load_yaml(path_to_file)
        current_keys = RunningSpec._flatten_dict(spec_dict)
        for k in RunningSpec.mandatory_keys:
            if current_keys.get(k) is None:
                raise RunningSpecError(f"Please make sure spec contains {k} configuration.")
        return True
