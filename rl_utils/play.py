from typing import Union

import gym
from scipy.special import softmax

from dpipe.train import ValuePolicy
import numpy as np

from .memory import Episode


def eps_greedy(eps: Union[ValuePolicy, float], n_actions, **kwargs):
    if not isinstance(eps, ValuePolicy):
        eps = ValuePolicy(eps)

    def decorator(func):
        def wrapper(*args, **kw):
            if np.random.binomial(1, eps.value):
                return np.random.randint(n_actions)

            return func(*args, **kw, **kwargs)

        return wrapper

    return decorator


def temperature_sample(temperature: Union[ValuePolicy, float] = 1, **kwargs):
    if not isinstance(temperature, ValuePolicy):
        temperature = ValuePolicy(temperature)

    def decorator(func):
        def wrapper(*args, **kw):
            weights = func(*args, **kw, **kwargs)

            t = temperature.value
            if t == 0:
                return weights.argmax()

            p = softmax(weights / t)
            return np.random.choice(len(p), p=p)

        return wrapper

    return decorator


def create_episode(env: gym.Env, take_action, wrap=None, **kwargs):
    episode, done = Episode(env.reset()), False
    if wrap is not None:
        episode = wrap(episode)

    while not done:
        action = take_action(episode.state(-1), **kwargs)
        state, reward, done, _, = env.step(action)
        episode.step(action, state, reward, done)

    return episode
