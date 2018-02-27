import torch
import numpy as np
from wrappers import *
from utils import *
from training import *
import gym

from tensorboard_easy import Logger
from torch import nn

class AC(nn.Module):
    def __init__(self, num_frames, actions):
        super().__init__()
        
        view = 64 * 5 * 4
        units = 200
        self.main_path = nn.Sequential(
            nn.Conv2d(num_frames, 32, 3, stride=2), 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2), 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            View(-1, view),
            nn.Linear(view, units),
            nn.ReLU(inplace=True),
        )
        
        self.action = nn.Linear(units, actions)
        self.value = nn.Linear(units, 1)
        
    def forward(self, x):
        x = self.main_path(x)
        return nn.functional.softmax(self.action(x)), self.value(x)


def train(episodes, index, passed, shared_model, optimizer):
    torch.manual_seed(np.random.randint(0, 1000))

    env = MarkovWrapper(RescaleWrapper(gym.make('PongNoFrameskip-v4'), (105, 80)), 15)
    env.seed(np.random.randint(0, 1000))
    
    agent = ActorCritic(AC(env.num_frames, env.action_space.n).cuda())
    agent.model.train()
    
    with Logger(f'pong/logs{index}') as logger:
        for i in range(episodes):
            total_r = async_actor_critic(env, agent, shared_model, optimizer, n_steps=60, max_grad=50)
            logger.log_scalar(f'rewards', total_r, i + passed)