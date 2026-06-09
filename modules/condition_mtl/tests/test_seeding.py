import random

import numpy as np
import torch

from modules.condition_mtl.seeding import set_global_seed


def test_seed_makes_torch_reproducible():
    set_global_seed(0)
    a = torch.rand(5)
    set_global_seed(0)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_seed_makes_python_numpy_reproducible():
    set_global_seed(123)
    r1, n1 = random.random(), np.random.rand()
    set_global_seed(123)
    r2, n2 = random.random(), np.random.rand()
    assert r1 == r2
    assert n1 == n2
