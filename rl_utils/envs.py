# some wrappers ideas were taken from
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

import numpy as np
import gym
from dpipe.im.utils import identity


class LambdaWrapper(gym.Wrapper):
    def __init__(self, env, change_state=identity, change_reward=identity, change_action=identity):
        super().__init__(env)
        self.change_action = change_action
        self.change_reward = change_reward
        self.change_state = change_state

    def reset(self, *args, **kwargs):
        return self.change_state(self.env.reset(**kwargs))

    def step(self, action):
        state, reward, done, info = self.env.step(self.change_action(action))
        return self.change_state(state), self.change_reward(reward), done, info


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop = env.unwrapped.get_action_meanings().index('NOOP')

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop)
            if done:
                obs = self.env.reset()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super().__init__(env)
        self.fire_action = env.unwrapped.get_action_meanings().index('FIRE')

    def reset(self):
        self.env.reset()
        s, _, done, _ = self.env.step(self.fire_action)
        if done:
            s = self.env.reset()
        return s


class FireAfterLoss(gym.Wrapper):
    def __init__(self, env=None):
        super().__init__(env)
        self.fire_action = env.unwrapped.get_action_meanings().index('FIRE')
        self.lives = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives:
            state, rew, done, info = self.env.step(self.fire_action)
            reward += rew
        self.lives = lives
        return state, reward, done, info

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(self.fire_action)
        if done:
            obs = self.env.reset()
        self.lives = self.unwrapped.ale.lives()
        return obs


class EpisodicLife(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.noop = env.unwrapped.get_action_meanings().index('NOOP')

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(self.noop)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FrameSkip(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Return only every `n_frames`-th frame"""
        super().__init__(env)
        self.n_frames = n_frames

    def step(self, action):
        total_reward = 0
        for _ in range(self.n_frames):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return state, total_reward, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames):
        super().__init__(env)
        self._frames = ()
        self.num_frames = n_frames

    def reset(self):
        self._frames = (self.env.reset(),) * self.num_frames
        return self._frames

    def step(self, action):
        s, reward, done, info = self.env.step(action)
        self._frames = self._frames[1:] + (s,)
        return self._frames, reward, done, info
