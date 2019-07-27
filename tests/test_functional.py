import cnf

import numpy as np

import pytest

import torch as t

import tests.datasets as datasets


@pytest.mark.fix_rng
def test_classifiction():
    x, Y = datasets.xor()
    dyn = cnf.dynamics.StackedHyperLinear(2)
    optim = t.optim.Adam(dyn.parameters(), lr=.01)
    for _ in range(100):
        optim.zero_grad()
        y = cnf.ode(dyn, x, atol=1e-2)
        loss = t.nn.functional.cross_entropy(y, Y)
        loss.backward()
        optim.step()
    assert loss.item() < 0.1


@pytest.mark.fix_rng
@pytest.mark.slow
def test_density_matching():
    target_distr1 = t.distributions.MultivariateNormal(
        t.tensor([1., -2.]),
        t.tensor([
            [ 1.00, -0.99],
            [-0.99,  1.00],
        ]),
    )
    target_distr2 = t.distributions.MultivariateNormal(
        t.tensor([2., 2.]),
        t.tensor([
            [0.8, 0.3],
            [0.3, 0.8],
        ]),
    )

    def lp(x):
        return (
            np.log(0.5)
            + t.logsumexp(
                t.stack([
                    target_distr1.log_prob(x),
                    target_distr2.log_prob(x),
                ]),
                dim=0,
            )
        )

    base_distr = t.distributions.MultivariateNormal(
        t.tensor(0.).expand(2),
        t.diag(t.tensor(1.).expand(2)),
    )

    dyn = cnf.dynamics.StackedHyperLinear(2)
    optim = t.optim.Adam(dyn.parameters(), lr=.01)

    for _ in range(200):
        optim.zero_grad()

        # D(q||p)
        y0 = base_distr.sample([100])
        y1, dlq = cnf.flow(dyn, y0, atol=1e-2)
        lq = base_distr.log_prob(y0) + dlq
        kl1 = (lq - lp(y1)).mean()

        # D(p||q)
        y1 = t.cat([
            target_distr1.sample([50]),
            target_distr2.sample([50]),
        ])
        y0, dlq = cnf.flow(dyn, y1, t0=1.0, t1=0.0, atol=1e-2)
        lq = base_distr.log_prob(y0) - dlq
        kl2 = (lp(y1) - lq).mean()

        js = .5 * kl1 + .5 * kl2

        js.backward()
        optim.step()

    assert js.item() < 0.5
