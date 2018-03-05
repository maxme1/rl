import argparse
from os.path import join as jp
import threading
from functools import partial
from queue import Queue, Empty

import torch
from resource_manager import read_config
from tensorboard_easy import Logger

from rl_utils.interfaces.base import make_step, tb_logger
from rl_utils.interfaces.memory import Memory
from dpipe.torch.model import set_lr


def make_history(queue: Queue, env, agent, mem: Memory, prepare_last_state, get_action, logger):
    while True:
        try:
            queue.get_nowait()
            break
        except Empty:
            make_step(env, agent, mem, get_action, prepare_last_state)
            logger(mem)


parser = argparse.ArgumentParser()
parser.add_argument('experiment_path')
args = parser.parse_args()

config_path = jp(args.experiment_path, 'resources.config')
log_path = jp(args.experiment_path, 'logs')
model_path = jp(args.experiment_path, 'model')
rm = read_config(config_path)

logger = Logger(log_path)
log_rewards = logger.make_log_scalar('rewards')
log_disc_rewards = logger.make_log_scalar('discounted_rewards')
log_lr = logger.make_log_scalar('lr')
logger = partial(tb_logger, log_rewards=log_rewards, log_disc_rewards=log_disc_rewards, gamma=rm.gamma)

optimizer = rm.optimizer
agent = rm.agent
memory = rm.memory

queue = Queue()
p = threading.Thread(target=make_history,
                     args=(queue, rm.env, agent, memory, rm.prepare_batch, rm.get_action, logger))
p.start()

try:
    while True:
        if memory.total_steps >= rm.min_steps:
            loss = rm.calculate_loss(agent, memory, rm.train_batch)

            if rm.get_lr is not None:
                set_lr(optimizer, rm.get_lr(memory))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
except KeyboardInterrupt:
    queue.put(None)
    torch.save(agent.state_dict(), model_path)
    p.join()
