import numpy as np
import jittor as jt
from jittor import nn


def img2mse(x, y):
    return jt.mean((x - y) ** 2)


def mse2psnr(x):
    return -10.0 * jt.log(x) / jt.log(jt.array(np.array([10.0])))


class MSELoss(nn.Module):
    def __init__(self):
        pass

    def execute(self, x, target):
        return img2mse(x, target)
