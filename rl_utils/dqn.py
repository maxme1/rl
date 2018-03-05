import torch.nn.functional
from gym import spaces
from torch import nn
from torch.autograd import Variable
import numpy as np
import cv2

from rl_utils.interfaces.memory import Memory
from rl_utils.interfaces.base import to_var
from rl_utils.utils import View
from rl_utils.wrappers import EpisodicLifeEnv, NoopResetEnv, MaxAndSkipEnv, FireResetEnv, LambdaObservation, \
    LambdaReward, FireAfterLoss


def get_action(predict: Variable):
    return predict.data.cpu().numpy().argmax()


def epsilon_greedy_action(x, get_eps, sampler, get_action=get_action):
    if np.random.binomial(1, get_eps()):
        return sampler()
    return get_action(x)


def exponential_lr(memory: Memory, initial, multiplier, step_length, floordiv=True):
    if floordiv:
        exp = memory.total_episodes // step_length
    else:
        exp = memory.total_episodes / step_length
    return initial * multiplier ** exp


def prepare_batch(mem: Memory, size):
    episode = mem.last_episode()
    state = episode.states[-size:]
    state = np.concatenate(state)
    state = complete_stack(state, size)
    return to_var(state[None].astype('float32'))


def complete_stack(stack, size):
    if len(stack) < size:
        # TODO: pad
        stack = np.concatenate([np.zeros((size - len(stack), *stack.shape[1:])), stack])
    return stack


def sample_stack(mem, stack_size):
    while True:
        episode = mem.sample_episode()
        if len(episode.actions) > 0:
            break

    idx = np.random.randint(0, max(len(episode.actions) - stack_size, 1))
    s, a, r, d = episode.get_slice(idx, stack_size)
    s = np.concatenate(s)
    begin = complete_stack(s[:-1], stack_size)
    end = complete_stack(s[1:], stack_size)
    return [begin, end], [a[-1]], [r[-1]], d


def prepare_train_batch(mem: Memory, batch_size, stack_size):
    states, *other = [*zip(*[sample_stack(mem, stack_size) for _ in range(batch_size)])]
    states = np.float32(states)
    return (states.reshape(-1, *states.shape[2:]), *other)


def calculate_loss(agent: nn.Module, mem: Memory, prepare_batch, gamma):
    states, actions, rewards, done = prepare_batch(mem)
    b, e = to_var(states[::2]), to_var(states[1::2], volatile=True)
    actions = to_var(actions)
    rewards = to_var(rewards).float().squeeze()
    done = torch.ByteTensor(done).cuda()

    begin, end = agent(b).gather(1, actions), agent(e).max(1)[0]
    end[done] = 0
    end.volatile = False

    return nn.functional.mse_loss(begin, rewards + gamma * end)


class DQN(nn.Module):
    def __init__(self, input_channels, n_actions, n_features=3136):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            View(n_features),
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)


def to_84(state):
    state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114
    state = cv2.resize(state, (84, 110), interpolation=cv2.INTER_AREA)
    # TODO: remove and add global/pyramid pooling
    state = state[18:102, :].reshape(1, 84, 84)
    return state.astype('uint8')


def wrap_dqn(env):
    """Apply a common set of wrappers for Atari games."""
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = LambdaObservation(env, to_84, spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8))
    env = LambdaReward(env, np.sign)
    return env


def wrap_breakout(env):
    assert 'NoFrameskip' in env.spec.id
    env = FireAfterLoss(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = LambdaObservation(env, to_84, spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8))
    env = LambdaReward(env, np.sign)
    return env
