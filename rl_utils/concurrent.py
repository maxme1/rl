import argparse
import os
import threading
from functools import partial
from queue import Queue, Empty

import torch
from resource_manager import read_config
from tensorboard_easy import Logger

from rl_utils.interfaces.base import make_step, tb_logger
from rl_utils.interfaces.memory import Memory


def make_history(queue: Queue, env, agent, mem: Memory, prepare_last_state, get_action, logger):
    while True:
        try:
            queue.get_nowait()
            break
        except Empty:
            make_step(env, agent, mem, get_action, prepare_last_state)
            logger(mem)


parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('experiment_path')
args = parser.parse_args()

rm = read_config(args.config_path)
os.makedirs(args.experiment_path)
rm.save_config(os.path.join(args.experiment_path, 'resources.config'))
log_path = os.path.join(args.experiment_path, 'logs')
model_path = os.path.join(args.experiment_path, 'model')
os.makedirs(model_path)

logger = Logger(log_path)
log_rewards = logger.make_log_scalar('rewards')
log_disc_rewards = logger.make_log_scalar('discounted_rewards')
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
        # TODO: dangerous
        if not memory.empty():
            loss = rm.calculate_loss(agent, memory, rm.train_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
except KeyboardInterrupt:
    queue.put(None)
    torch.save(agent.state_dict(), model_path)
    p.join()
