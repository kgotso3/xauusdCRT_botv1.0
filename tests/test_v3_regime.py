import pandas as pd

from ml.v3_walkforward import DEFAULT_DEVELOPMENT_CUTOFF, V3WalkForwardConfig, iter_walkforward_slices


def test_v3_development_cutoff_is_frozen_forward_boundary():
    assert DEFAULT_DEVELOPMENT_CUTOFF == pd.Timestamp("2026-09-01T00:00:00Z")


def test_walkforward_slices_are_chronological_and_non_overlapping():
    cfg = V3WalkForwardConfig(train_size=100, validation_size=20, step_size=20)
    folds = list(iter_walkforward_slices(180, cfg))
    assert len(folds) == 4
    for _, train_slice, val_slice in folds:
        assert train_slice.stop <= val_slice.start
        assert (train_slice.stop - train_slice.start) == 100 or train_slice.start == 0
        assert (val_slice.stop - val_slice.start) == 20
