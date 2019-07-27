import itertools as it

import torch as t


def xor():
    x = t.tensor([*it.product([0, 1], [0, 1])]).float()
    y = t.tensor([a == b for a, b in x]).long()
    return x, y
