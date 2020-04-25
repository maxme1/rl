import numpy as np
import torch
from torch.nn import functional

from dpipe.im.checks import check_shape_along_axis
from dpipe.itertools import lmap
from dpipe.torch import to_var, to_np, set_params
from rl_utils.memory import Memory
from rl_utils.utils import discount_rewards


def q_update(states, actions, rewards, done, *, gamma, agent, target_agent=None, optimizer, max_grad_norm=None,
             norm_type='inf', **optimizer_params):
    check_shape_along_axis(actions, rewards, axis=1)
    n_steps = actions.shape[1]
    assert n_steps > 0
    assert states.shape[1] == n_steps + 1

    agent.train()
    if target_agent is None:
        target_agent = agent
    else:
        target_agent.eval()

    # first and last state
    start, stop = states[:, 0], states[:, -1]
    # discounted rewards
    rewards = discount_rewards(np.moveaxis(rewards, 1, 0), gamma)
    actions = actions[:, [0]]
    gamma = gamma ** n_steps

    start, stop, actions, rewards, done = to_var(start, stop, actions, rewards, done, device=agent)

    predicted = agent(start).gather(1, actions).squeeze(1)
    with torch.no_grad():
        values = target_agent(stop).detach().max(1).values
        expected = (1 - done.to(values)) * values * gamma + rewards

    loss = functional.mse_loss(predicted, expected)
    set_params(optimizer, **optimizer_params)
    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm, norm_type)

    optimizer.step()
    return to_np(loss)


@torch.no_grad()
def get_q_values(state, agent):
    return to_np(agent(to_var(state[None], device=agent)))[0]


def describe_dqn(memory: Memory, agent, gamma: float = 1):
    state = memory.sample_episode().state(0)
    rewards = [e.rewards() for e in memory.episodes()]

    sizes = np.array(lmap(len, rewards))
    discounted = np.array([discount_rewards(r, gamma) for r in rewards])
    summed = np.array([discount_rewards(r, 1) for r in rewards])
    q_values = get_q_values(state, agent)

    return {
        'mean reward': discounted.mean(), 'rewards': discounted[None],
        'mean reward sum': summed.mean(), 'rewards sum': summed[None],
        'mean size': sizes.mean(), 'sizes': sizes[None],
        'q_max': q_values.max(), 'q_min': q_values.min(),
    }
