from sentbias.weat import truncate_dicts


NUM_TRIALS = 1000


def test_truncate_dicts_deterministic():
    (d1, d2) = truncate_dicts(
        {1: 0, 3: 1, 5: 0}, {0: 1, 1: 1, 2: 0, 3: 1, 4: 1},
        deterministic=True)
    assert len(d1) == 3
    assert len(d2) == 3
    for _ in range(NUM_TRIALS):
        assert truncate_dicts(
            {1: 0, 3: 1, 5: 0}, {0: 1, 1: 1, 2: 0, 3: 1, 4: 1},
            deterministic=True) == (d1, d2)
