from abc import ABC, abstractmethod
from collections import deque
from typing import Deque

import numpy as np


class Episode:
    def __init__(self, initial_state):
        self.states, self.actions, self.rewards, self.done = [initial_state], [], [], False

    def step(self, action, new_state, reward, done, info):
        assert not self.done
        self.states.append(new_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.done = done

    def get_slice(self, start, size=1):
        assert start < len(self.actions)
        stop = start + size
        done = stop >= len(self.actions) and self.done
        return self.states[start: stop + 1], self.actions[start: stop], self.rewards[start: stop], done

    def get_entry(self, start):
        s, a, r, d = self.get_slice(start)
        return s, a[0], r[0], d

    # TODO: rewrite sampling
    # def sample_index(self):
    #     return np.random.randint(0, len(self.actions))


class Memory(ABC):
    @abstractmethod
    def last_episode(self) -> Episode:
        pass

    @abstractmethod
    def step(self, action, new_state, reward, done, info):
        pass

    @abstractmethod
    def new_episode(self, state):
        pass

    @abstractmethod
    def sample_episode(self):
        pass

    @abstractmethod
    def last_state(self):
        pass

    @abstractmethod
    def empty(self):
        pass


class DequeMemory(Memory):
    def __init__(self, max_episodes=None):
        self._episodes: Deque[Episode] = deque(maxlen=max_episodes)

    def last_episode(self):
        return self._episodes[-1]

    def step(self, action, new_state, reward, done, info):
        self.last_episode().step(action, new_state, reward, done, info)

    def new_episode(self, state):
        self._episodes.append(Episode(state))

    def sample_episode(self):
        return self._episodes[np.random.randint(0, len(self._episodes))]

    def last_state(self):
        return self.last_episode().states[-1]

    def empty(self):
        return not self._episodes


class FrameLimitMemory(DequeMemory):
    def __init__(self, max_frames):
        super().__init__()
        self.max_frames = max_frames
        self.frames = 0

    def _step(self):
        self.frames += 1
        if self.frames >= self.max_frames and len(self._episodes) > 1:
            episode = self._episodes.popleft()
            self.frames -= len(episode.states)

    def step(self, action, new_state, reward, done, info):
        super().step(action, new_state, reward, done, info)
        self._step()

    def new_episode(self, state):
        super().new_episode(state)
        self._step()
