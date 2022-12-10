from typing import List

import numpy as np
import torch

from meta_critics.trajectory.advantage_episode import AdvantageBatchEpisodes
from meta_critics.base_trainer.torch_tools.torch_utils import weighted_mean, to_numpy
from meta_critics.policies.policy import Policy


def reinforce_loss(policy: Policy, episodes: AdvantageBatchEpisodes, W=None) -> torch.Tensor:
    """
    :param W:
    :param policy:
    :param episodes:
    :return:
    """

    # print("reinforce_loss episodes.observations", episodes.observations.shape)
    # print("reinforce_loss episodes.actions", episodes.actions.shape)
    # t = episodes.observation_shape
    # val = episodes.observations.view((-1, *episodes.observation_shape)).float()
    # pi = policy(val, W=W)
    try:
        # print("episodes obs", episodes.observations)
        # print("episodes act", episodes.actions)
        # print("episodes act", episodes.lengths)
        # print("Computing reinforce_loss {type}", episodes)
        pi = policy(episodes.observations.view((-1, *episodes.observation_shape)).float(), W=W)
        # episodes.require_grad()
        # episodes.to_gpu()
        log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))
        log_probs = log_probs.view(torch.max(episodes.lengths), episodes.batch_size)
        losses = -weighted_mean(log_probs * episodes.advantages, lengths=episodes.lengths)
    except Exception as ex:
        # print("ac ## ", episodes.actions)
        # print("ob ##", episodes.observations)
        # print("lt ##", episodes.lengths)
        print("error")
        raise ex

    return losses.mean()

#
# def value_iteration(transitions, rewards, gamma=0.95, theta=1e-5):
#     """
#
#     :param transitions:
#     :param rewards:
#     :param gamma:
#     :param theta:
#     :return:
#     """
#     rewards = np.expand_dims(rewards, axis=2)
#     values = np.zeros(transitions.shape[0], dtype=np.float32)
#     delta = np.inf
#     while delta >= theta:
#         q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
#         new_values = np.max(q_values, axis=1)
#         delta = np.max(np.abs(new_values - values))
#         values = new_values
#
#     return values


def value_iteration_finite_horizon(transitions, rewards, horizon=10, gamma=0.95):
    """

    :param transitions:
    :param rewards:
    :param horizon:
    :param gamma:
    :return:
    """
    rewards = np.expand_dims(rewards, axis=2)
    values = np.zeros(transitions.shape[0], dtype=np.float32)
    for k in range(horizon):
        q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
        values = np.max(q_values, axis=1)

    return values


def get_task_returns(tasks, episodes):
    ret = [-np.linalg.norm(np.array(episode.observations.cpu()) - np.expand_dims(tasks[taskIdx][1]['goal'], 0),
                           axis=2).sum(0) for taskIdx, episode in enumerate(episodes)]
    return to_numpy(ret)


def get_returns(episodes: List[AdvantageBatchEpisodes]) -> np.ndarray:
    """
    Stack rewards.
    :param episodes:
    :return:
    """
    return to_numpy([episode.rewards.sum(dim=0) for episode in episodes])
