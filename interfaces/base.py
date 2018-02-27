import gym
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

from .memory import Memory


def to_var(array, volatile=False):
    # TODO: are there cases when i don't need cuda?
    return Variable(torch.from_numpy(np.asarray(array)).cuda(), volatile=volatile)


def make_step(env: gym.Env, agent: nn.Module, memory: Memory, get_action, prepare_last_state):
    if memory.empty() or memory.last_episode().done:
        memory.new_episode(env.reset())
    else:
        batch = prepare_last_state(memory)
        predict = agent(batch)
        a = get_action(predict)
        state, reward, done, info = env.step(a)
        # TODO: probably a function not a method is better
        memory.step(a, state, reward, done, info)


def tb_logger(memory: Memory, log_rewards, log_disc_rewards, gamma):
    episode = memory.last_episode()
    if episode.done:
        log_rewards(sum(episode.rewards))
        temp = 0
        for r in reversed(episode.rewards):
            temp = temp * gamma + r
        log_disc_rewards(temp)
