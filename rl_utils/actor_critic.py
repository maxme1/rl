import torch
import torch.nn.functional as functional
from torch.autograd import Variable
from torch import nn

from rl_utils.interfaces.base import to_var
from rl_utils.utils import View


def calculate_loss(agent, memory, prepare_batch, gamma, entropy_weight, value_weight):
    states, actions, rewards, done = prepare_batch(memory)
    # TODO: add func for all this:
    b, e = to_var(states[::2]), to_var(states[1::2], volatile=True)
    actions = to_var(actions)
    rewards = to_var(rewards).float()
    done = torch.ByteTensor(done).cuda()

    prob_logits, value = agent(b)
    prob = functional.softmax(prob_logits, -1)
    log_prob = functional.log_softmax(prob_logits, -1)
    entropy = -(log_prob * prob).sum(1, keepdim=True)
    log_prob = log_prob.gather(1, actions)

    final_values = agent(e)[1]
    final_values[done] = 0
    final_values.volatile = False

    cumulative_reward = final_values * gamma + rewards
    value_loss = functional.mse_loss(value, cumulative_reward)

    delta_t = cumulative_reward.data - value.data
    policy_loss = - log_prob * Variable(delta_t) - entropy_weight * entropy

    return policy_loss.mean() + value_weight * value_loss


def get_action(predict: Variable):
    # predict == (prob_logits, value)
    return functional.softmax(predict[0], -1).multinomial().data.cpu()[0]


class ActorCritic(nn.Module):
    def __init__(self, input_channels, n_actions, n_features=3136):
        super().__init__()

        self.main_path = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            View(n_features),
            nn.Linear(n_features, 512),
            nn.ReLU(),
        )

        self.probs = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = self.main_path(x)
        return self.probs(x), self.value(x)
