from copy import deepcopy

from typing import Any, Dict, Optional, Tuple

from scipy.integrate import ode as ode_

import torch as t

from .utility import flatten, jacobian


class ODE(t.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            dynamics: t.nn.Module,
            parameters: t.Tensor,
            y0: t.Tensor,
            t0: float,
            t1: float,
            integrator: str,
            integrator_params: Dict[str, Any],
    ):
        ctx.dynamics = dynamics
        ctx.parameters = parameters.detach().clone()
        ctx.t0, ctx.t1 = t0, t1
        ctx.dtype, ctx.device = y0.dtype, y0.device
        ctx.integrator, ctx.integrator_params = \
            integrator, deepcopy(integrator_params)

        y0, unflatten_y = flatten((y0,))
        y0 = y0.detach().cpu().numpy()

        def _dynamics(time, y):
            y = t.tensor(y, device=ctx.device, dtype=ctx.dtype)
            y, = unflatten_y(y)
            f = dynamics(y, time)
            f, _ = flatten((f,))
            f = f.cpu().numpy()
            return f

        y1 = (
            ode_(_dynamics)
            .set_integrator(ctx.integrator, **ctx.integrator_params)
            .set_initial_value(y0, t=ctx.t0)
            .integrate(t=ctx.t1)
        )
        y1 = t.tensor(y1, device=ctx.device, dtype=ctx.dtype)
        y1, = unflatten_y(y1)

        ctx.save_for_backward(y1)

        return y1

    @staticmethod
    def backward(ctx, dldy1: t.Tensor):
        dynamics = ctx.dynamics
        dtype, device = ctx.dtype, ctx.device

        y1, = ctx.saved_tensors

        a1, unflatten_a = flatten((y1, dldy1, t.zeros_like(ctx.parameters)))
        a1 = a1.detach().cpu().numpy()

        def _dynamics(time, a):
            a = t.tensor(a, device=device, dtype=dtype)
            y, dldy, _dldparam = unflatten_a(a)
            y = y.requires_grad_()
            with t.enable_grad():
                fy = dynamics(y, time)
                fdldy, *fdldparam = t.autograd.grad(
                    fy, (y, *dynamics.parameters()), -dldy)
            f, _ = flatten((fy, fdldy, *fdldparam))
            f = f.cpu().numpy()
            return f

        a0 = (
            ode_(_dynamics)
            .set_integrator(ctx.integrator, **ctx.integrator_params)
            .set_initial_value(a1, t=ctx.t1)
            .integrate(t=ctx.t0)
        )
        y0, dldy0, dldparam0 = unflatten_a(a0)

        return (
            None,
            # ^ dynamics: t.nn.Module
            t.tensor(dldparam0, device=device, dtype=dtype),
            # ^ parameters: t.Tensor
            t.tensor(dldy0, device=device, dtype=dtype),
            # ^  y0: t.Tensor
            None,
            # ^  t0: float
            None,
            # ^  t1: float
            None,
            # ^  integrator: str
            None,
            # ^  integrator_params: Dict[str, Any]
        )


def ode(
        dynamics: t.nn.Module,
        y0: t.Tensor,
        t0: float = 0.0,
        t1: float = 1.0,
        integrator: Optional[str] = None,
        **integrator_params: Any,
) -> t.Tensor:
    if integrator is None:
        integrator = 'dopri5'
    if integrator_params is None:
        integrator_params = {}

    params, _ = flatten(dynamics.parameters())

    return ODE.apply(
        dynamics, params, y0, t0, t1, integrator, integrator_params)


def flow(
        dynamics: t.nn.Module,
        y0: t.Tensor,
        *args,
        **kwargs,
) -> Tuple[t.Tensor, t.Tensor]:
    class flow_dynamics(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.add_module('dynamics', dynamics)

        def forward(self, y, time):
            y = y[:, :-1]
            y = y.requires_grad_()
            with t.enable_grad():
                dydt = dynamics(y, time)
            dlpdt = t.stack([-t.trace(x) for x in jacobian(dydt, y)])
            return t.cat([dydt, dlpdt.unsqueeze(1)], 1)

    if len(y0.shape) not in [1, 2]:
        raise NotImplementedError()
    if len(y0.shape) == 1:
        y0 = y0.unsqueeze(0)
    y0 = t.cat([y0, t.zeros(len(y0), 1).to(y0)], 1)

    y = ode(flow_dynamics(), y0, *args, **kwargs)

    return y[:, :-1], y[:, -1]
