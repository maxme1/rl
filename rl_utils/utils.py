import numpy as np
from torch import nn

from dpipe.train import TBLogger


def discount_rewards(rewards, gamma):
    reward = 0
    powers = 1
    for value in rewards:
        reward += value * powers
        powers *= gamma

    return reward


class HistLogger(TBLogger):
    def value(self, name, value, step):
        if np.asarray(value).ndim > 1:
            self.logger.log_histogram(name, value, step)
        else:
            super().value(name, value, step)


def linear_trend(x, start_value, stop_value, duration, clip=True):
    if clip and x >= duration:
        return stop_value

    a = (stop_value - start_value) / duration
    return x * a + start_value


def wrap_resnet(model, channels):
    model.conv1 = nn.Conv2d(channels, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    return model

# def expiration_pool(iterable: Iterable, pool_size: int, repetitions: int):
#     assert pool_size > 0
#     assert repetitions > 0
#
#     def sample_value():
#         idx = random.randrange(len(freq))
#         value, counter = freq[idx]
#
#         freq[idx] = value, counter + 1
#         if counter + 1 >= repetitions:
#             del freq[idx]
#
#         return value
#
#     def add_value(value):
#         freq.append([value, 0])
#
#     freq = []
#     for v in iterable:
#         add_value(v)
#         yield sample_value()
#
#         while len(freq) == pool_size:
#             yield sample_value()
#
#     while len(freq):
#         yield sample_value()
