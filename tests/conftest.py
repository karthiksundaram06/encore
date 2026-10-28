import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def small_trie():
    from encore.sid.rqkmeans import add_dedup_level
    from encore.sid.trie import SIDTrie

    rng = np.random.default_rng(0)
    codes = rng.integers(0, 4, size=(60, 2))
    full, D = add_dedup_level(codes, rng.random(60))
    return SIDTrie(full, [4, 4, D])
