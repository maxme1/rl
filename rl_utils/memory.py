import time
from abc import ABC, abstractmethod
from threading import Lock
from typing import Iterable

import numpy as np


class EpisodeBase(ABC):
    done: bool

    @abstractmethod
    def step(self, action, new_state, reward, done):
        pass

    @abstractmethod
    def state(self, index):
        pass

    @abstractmethod
    def action(self, index):
        pass

    @abstractmethod
    def reward(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @staticmethod
    def _slice(method, start, stop, step):
        return [method(i) for i in range(start, stop, step)]

    def states(self, start=0, stop=None, step=1):
        if stop is None:
            stop = len(self)

        return self._slice(self.state, start, stop, step)

    def actions(self, start=0, stop=None, step=1):
        if stop is None:
            stop = len(self) - 1

        return self._slice(self.action, start, stop, step)

    def rewards(self, start=0, stop=None, step=1):
        if stop is None:
            stop = len(self) - 1

        return self._slice(self.reward, start, stop, step)

    def sample(self, size: int, min_size: int = None):
        if min_size is None:
            min_size = size
        assert min_size <= size

        min_start = min_size - size
        max_start = len(self) - size + 1
        start = np.random.randint(min_start, max_start)
        stop = start + size
        start = max(0, start)

        assert stop <= len(self)
        done = stop >= len(self) and self.done

        s = self.states(start, stop)
        a = self.actions(start, stop - 1)
        r = self.rewards(start, stop - 1)
        assert len(s) >= min_size
        assert len(a) == len(r) == len(s) - 1
        return s, a, r, done


class Episode(EpisodeBase):
    def __init__(self, initial_state):
        self._states, self._actions, self._rewards, self.done = [initial_state], [], [], False

    def step(self, action, new_state, reward, done):
        assert not self.done
        self._states.append(new_state)
        self._actions.append(action)
        self._rewards.append(reward)
        self.done = done

    def __len__(self):
        return len(self._states)

    def state(self, index):
        return self._states[index]

    def action(self, index):
        return self._actions[index]

    def reward(self, index):
        return self._rewards[index]


class Memory(ABC):
    # populate the memory
    @abstractmethod
    def add_episode(self, episode: Episode):
        pass

    @abstractmethod
    def full(self) -> bool:
        pass

    def sample(self, size: int, min_size=None, wait=True):
        return self.sample_episode(wait).sample(size, min_size)

    @abstractmethod
    def sample_episode(self, wait=True) -> Episode:
        pass

    @abstractmethod
    def episodes(self) -> Iterable[Episode]:
        pass


class ExpirationMemory(Memory):
    def __init__(self, max_size, max_fraction=1):
        self.max_fraction = max_fraction
        self.max_size = max_size
        self._episodes = {}
        self._lock = Lock()

    def add_episode(self, episode):
        with self._lock:
            self._episodes[episode] = 0

    def sample_episode(self, wait=True):
        if self.size == 0:
            if not wait:
                raise ValueError('Mem is empty')

            while self.size == 0:
                time.sleep(1)

        with self._lock:
            episodes = list(self._episodes.items())

            weights = np.array([len(episode) / (counts or 1) for episode, counts in episodes])
            idx = np.random.choice(len(weights), p=weights / weights.sum())
            episode, _ = episodes[idx]

            self._episodes[episode] += 1
            if self._episodes[episode] / len(episode) >= self.max_fraction:
                self._episodes.pop(episode)

            return episode

    def episodes(self):
        with self._lock:
            return list(self._episodes)

    @property
    def size(self):
        with self._lock:
            return sum(map(len, self._episodes.keys()))

    def full(self) -> bool:
        return self.size >= self.max_size
