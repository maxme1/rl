import numpy as np


# TODO: don't need the counter anymore
class Counter:
    def __init__(self, func, start=0, stop=None):
        self.func = func
        self.arg = start
        self.end = stop or np.inf

    def __call__(self):
        result = self.func(self.arg)
        if self.arg < self.end:
            self.arg += 1
        return result


def linear_decay(i, start_value, end_value, steps):
    return start_value + (end_value - start_value) * i / steps


def periodic_linear(i, start_value, end_value, steps):
    i = i % steps
    return linear_decay(i, start_value, end_value, steps)


def periodic_decay(i, start_value, end_value, steps, multiplier):
    i = i % steps
    start_value = start_value * multiplier ** (i // steps)
    return linear_decay(i, start_value, end_value, steps)
