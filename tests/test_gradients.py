import cnf

import pytest

import tests.datasets as datasets

import torch as t


@pytest.mark.fix_rng
def test_gradients():
    x, Y = datasets.xor()
    x = x.double().requires_grad_()

    t0 = t.tensor(0.0).double().requires_grad_()
    t1 = t.tensor(1.0).double().requires_grad_()
    w = t.nn.Parameter(t.randn(2, 2).double())
    b = t.nn.Parameter(t.randn(2).double())

    def f(w, b, x, t0, t1):
        class Dynamics(t.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = w
                self.b = b

            def forward(self, x, time):
                x = t.tanh(x)
                x = x @ w + b
                return x

        y = cnf.ode(Dynamics(), x, t0, t1, atol=1e-8)
        return t.nn.functional.cross_entropy(y, Y)

    t.autograd.gradcheck(f, (w, b, x, t0, t1))
