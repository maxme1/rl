from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from torch import nn


class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


def generate_episode(env, policy, max_len=None):
    s = env.reset()
    states, actions, rewards = [s], [], []
    i = 0
    done = False
    while not done and (max_len is None or i < max_len):
        i += 1
        a = policy(s)
        s, r, done, info = env.step(a)

        actions.append(a)
        rewards.append(r)
        states.append(s)
    return states, actions, rewards


def animate(data, output_name=None, fps=30, size=None, writer='imagemagick', cmap=None, vmin=None, vmax=None):
    fig = plt.figure(figsize=size)
    im = plt.imshow(data[0], animated=True, cmap=cmap, vmin=vmin, vmax=vmax)

    def update(i):
        im.set_data(data[i])
        if vmin is None or vmax is None:
            im.autoscale()
        return im,

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.axis('off')

    animation = FuncAnimation(fig, func=update, frames=len(data), interval=20, blit=True)
    if output_name is not None:
        animation.save(output_name, fps=fps, writer=writer)
    return animation
