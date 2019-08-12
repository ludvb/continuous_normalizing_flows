from functools import reduce

import torch as t

from .utility import intersperse


class HyperLinear(t.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._activations = t.nn.Linear(in_size, out_size)
        self._bias = t.nn.Linear(1, out_size, bias=False)
        self._gate = t.nn.Sequential(
            t.nn.Linear(1, out_size),
            t.nn.Sigmoid(),
        )

    def forward(self, y, time):
        time = time.unsqueeze(0)
        return (
            self._bias.to(time)(time)
            +
            self._gate.to(time)(time) * self._activations.to(y)(y)
        )


class StackedHyperLinear(t.nn.Module):
    def __init__(
            self,
            in_size,
            out_size=None,
            hidden_size=64,
            hidden_n=2,
            activation_fnc=t.tanh,
    ):
        if out_size is None:
            out_size = in_size

        super().__init__()

        self._layers = t.nn.ModuleList([
            HyperLinear(in_size, hidden_size),
            *[HyperLinear(hidden_size, hidden_size) for _ in range(hidden_n)],
            HyperLinear(hidden_size, out_size),
        ])
        self._activation = activation_fnc

    def forward(self, y, time):
        return reduce(
            lambda y, f: f(y),
            intersperse(
                self._activation,
                map(lambda m: lambda y: m(y, time), self._layers),
            ),
            y,
        )
