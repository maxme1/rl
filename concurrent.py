import threading

from functools import partial
import os
from queue import Queue, Empty

import gym
from tensorboard_easy import Logger
import numpy as np

from interfaces.base import make_step, tb_logger
from interfaces.counter import Counter, linear_decay
from interfaces.dqn import DQN, get_action, epsilon_greedy_action, prepare_train_batch, calculate_loss, prepare_batch
from interfaces.memory import FrameLimitMemory, Memory
# from wrappers import wrap_dqn
from oai_wrappers import LambdaObservation, wrap_dqn
import torch


def make_history(queue: Queue, env, agent, mem: Memory, prepare_last_state, get_action, logger):
    while True:
        try:
            queue.get_nowait()
            break
        except Empty:
            make_step(env, agent, mem, get_action, prepare_last_state)
            logger(mem)


log_path = os.path.expanduser('~/rl_stuff/pong/stack')
env: gym.Env = wrap_dqn(gym.make('PongNoFrameskip-v4'))

agent = DQN(4, env.action_space.n).cuda()
optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)

eps = Counter(partial(
    linear_decay, start_value=1, end_value=.02, steps=10 ** 5
), stop=10 ** 5)

mem = FrameLimitMemory(1e4)
logger = Logger(log_path)
log_rewards = logger.make_log_scalar('rewards')
log_disc_rewards = logger.make_log_scalar('discounted_rewards')

train_batch = partial(prepare_train_batch, batch_size=50, stack_size=4)
calculate_loss = partial(calculate_loss, gamma=.99)
get_action = partial(epsilon_greedy_action, get_eps=eps, sampler=env.action_space.sample, get_action=get_action)
logger = partial(tb_logger, log_rewards=log_rewards, log_disc_rewards=log_disc_rewards, gamma=.99)
prepare_batch = partial(prepare_batch, size=4)

model_path = os.path.join(log_path, 'model')
queue = Queue()
p = threading.Thread(target=make_history,
                     args=(queue, env, agent, mem, prepare_batch, get_action, logger))
p.start()

try:
    while True:
        if not mem.empty():
            loss = calculate_loss(agent, mem, train_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
except KeyboardInterrupt:
    queue.put(None)
    torch.save(agent.state_dict(), model_path)
    # save config
    p.join()
