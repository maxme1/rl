from typing import Sequence, Any, Callable

import numpy as np
import gym
from torch.nn import Module

from dpipe.train import Policy, ValuePolicy
from .memory import Memory
from .play import create_episode


def sample_and_populate(env: gym.Env, memory: Memory, step: Callable, sample_size: int, max_size: int = np.inf,
                        wrap_episode: Callable = None, **kwargs):
    while True:
        if memory.size < max_size:
            memory.add_episode(create_episode(env, step, wrap_episode, **kwargs))
        else:
            yield memory.sample(sample_size)


class HardUpdateAgent(Policy):
    def __init__(self, agent: Module, target: Module, frequency: int):
        self.frequency = frequency
        self.target = target
        self.agent = agent

    def epoch_finished(self, epoch: int, train_losses: Sequence, metrics: dict = None):
        if (epoch + 1) % self.frequency == 0:
            self.target.load_state_dict(self.agent.state_dict())


class DetachValue(Policy):
    def __init__(self, value: ValuePolicy):
        self.value = value

    def epoch_started(self, epoch: int):
        self.value.epoch_started(epoch)

    def train_step_started(self, epoch: int, iteration: int):
        self.value.train_step_started(epoch, iteration)

    def train_step_finished(self, epoch: int, iteration: int, loss: Any):
        self.value.train_step_finished(epoch, iteration, loss)

    def validation_started(self, epoch: int, train_losses: Sequence):
        self.value.validation_started(epoch, train_losses)

    def epoch_finished(self, epoch: int, train_losses: Sequence, metrics: dict = None):
        self.value.epoch_finished(epoch, train_losses, metrics)
