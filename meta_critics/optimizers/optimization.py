from typing import Optional, Callable
import torch


def conjugate_gradient(f_Ax: Callable, b: torch.Tensor,
                       cg_iters: Optional[int] = 10, residual_tol=1e-10) -> torch.Tensor:
    """
    :param f_Ax:
    :param b:
    :param cg_iters:
    :param residual_tol:
    :return:
    """
    p = b.clone().detach()
    r = b.clone().detach()
    x = torch.zeros_like(b)
    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr.item() < residual_tol:
            break

    # x = x.detach()
    return x.detach()
