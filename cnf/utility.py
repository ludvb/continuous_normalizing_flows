from typing import Callable, List, Iterable, Tuple

import numpy as np

import torch as t


def flatten(
        xs: Iterable[t.Tensor],
) -> Tuple[t.Tensor, Callable[[t.Tensor], t.Tensor]]:
    xs = list(xs)
    shapes = [x.shape for x in xs]

    def _unflatten(x: t.Tensor) -> List[t.Tensor]:
        ret = []
        for shape in shapes:
            cur_length = np.prod(shape, dtype=int)
            if cur_length > len(x):
                raise ValueError('input has invalid shape')
            cur = x[:cur_length]
            if len(shape) > 0:
                cur = cur.reshape(*shape)
            else:
                cur = cur.squeeze()
            ret.append(cur)
            x = x[cur_length:]
        if len(x) > 0:
            raise ValueError('input has invalid shape')
        return ret

    return (
        t.cat([
            x.flatten() if x.ndimension() >= 1 else x.unsqueeze(0)
            for x in xs]),
        _unflatten,
    )


def jacobian(y, x):
    if y.ndimension() > 2:
        raise NotImplementedError('Can only compute Jacobian for y in R^n')

    while y.ndimension() < 2:
        y = y.unsqueeze(0)

    with t.enable_grad():
        return (
            t.stack([
                t.autograd.grad(yi, x, t.ones_like(yi), create_graph=True)[0]
                for yi in y.transpose(0, 1).reshape(-1, len(y))
            ])
            .transpose(0, 1)
        )


def intersperse(y, xs):
    yield next(xs)
    for x in xs:
        yield y
        yield x
