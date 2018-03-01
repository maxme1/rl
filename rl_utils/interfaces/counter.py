import numpy as np


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
