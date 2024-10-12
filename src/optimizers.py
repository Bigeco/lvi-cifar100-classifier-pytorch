from typing import Iterable
import torch
from torch.optim._multi_tensor import SGD


class SAMSGD(SGD):
    def __init__(self,
                 params: Iterable[torch.Tensor],
                 lr: float,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False,
                 rho: float = 0.05,
                 ):
        if rho <= 0:
            raise ValueError(f"Invalid neighborhood size: {rho}")
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        # todo: generalize this
        if len(self.param_groups) > 1:
            raise ValueError("Not supported")
        self.param_groups[0]["rho"] = rho

    @torch.no_grad()
    def step(self,
             closure
             ) -> torch.Tensor:
        closure = torch.enable_grad()(closure)
        loss = closure().detach()

        for group in self.param_groups:
            grads = []
            params_with_grads = []

            rho = group['rho']

            for p in group['params']:
                if p.grad is not None:
                    grads.append(p.grad.clone().detach())
                    params_with_grads.append(p)
            device = grads[0].device

            grad_norm = torch.stack([g.detach().norm(2).to(device) for g in grads]).norm(2)
            epsilon = grads
            torch._foreach_mul_(epsilon, rho / grad_norm)

            torch._foreach_add_(params_with_grads, epsilon)
            closure()
            torch._foreach_sub_(params_with_grads, epsilon)

        super().step()
        return loss