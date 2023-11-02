import jittor as jt
from jittor import nn


class HuberLoss(nn.Module):
    def __init__(self, delta):
        self.delta = delta
        self.delta_quad = 0.5 * delta**2

    def execute(self, x, target):
        rel = jt.abs(x - target)
        sqr = 0.5 / self.delta * rel * rel
        return jt.ternary((rel > self.delta), rel - 0.5 * self.delta, sqr)
