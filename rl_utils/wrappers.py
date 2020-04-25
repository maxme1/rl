from typing import Union

from dpipe.im.shape_ops import pad_to_shape
from rl_utils.memory import Episode, EpisodeBase
import numpy as np


class EpisodeWrapper(EpisodeBase):
    def __init__(self, episode: Union[Episode, 'EpisodeWrapper']):
        self.episode = episode

    def step(self, action, new_state, reward, done):
        return self.episode.step(action, new_state, reward, done)

    @property
    def done(self):
        return self.episode.done

    def __len__(self):
        return len(self.episode)

    def state(self, index):
        return self.episode.state(index)

    def action(self, index):
        return self.episode.action(index)

    def reward(self, index):
        return self.episode.reward(index)


class SlidingWindow(EpisodeWrapper):
    def __init__(self, episode: Union[Episode, 'EpisodeWrapper'], window_size, padding_values=0):
        super().__init__(episode)
        self.padding_values = padding_values
        self.window_size = window_size

    def state(self, index):
        if index < 0:
            index += len(self)
        states = self.episode.states(max(0, index - self.window_size + 1), index + 1)
        states = pad_to_shape(np.asarray(states), self.window_size, 0, self.padding_values, 1)
        return states


class ChangeState(EpisodeWrapper):
    def __init__(self, episode: Union[Episode, 'EpisodeWrapper'], change_state):
        super().__init__(episode)
        self.change_state = change_state

    def state(self, index):
        return self.change_state(self.episode.state(index))
