import numpy as np
from typing import List, Sequence, Any # typechecking with mypy
from numpy.typing import NDArray # typechecking with mypy
from scipy.linalg import sqrtm, expm


def expct(dm: NDArray, _op: NDArray) -> float:
    return np.real(np.trace(np.matmul(dm, _op)))


def get_state_label(dm, sz_ops: list[NDArray], adag: NDArray=None):
    label_str = r'$\vert$' + (
        r'$\uparrow$'
        if abs(expct(dm, sz_ops[0]) - 1) < 0.5
        else r'$\downarrow$'
    )
    if adag is not None:
        label_str += str(round(expct(dm, adag)))
    if len(sz_ops) > 1:
        for _szop in sz_ops[1:]:
            if abs(expct(dm, _szop) - 1) < 0.5:  # up = 1 = ex
                label_str += r'$\uparrow$' 
            else:
                label_str += r'$\downarrow$'
    label_str += r'$\rangle$'
    return label_str


def dm_fidelity(true, approx, inv=False):
    sqrt_true = sqrtm(true)
    fidelity = (np.trace(sqrtm(np.matmul(sqrt_true, approx @ sqrt_true)))) ** 2
    return 1 - fidelity if inv else fidelity