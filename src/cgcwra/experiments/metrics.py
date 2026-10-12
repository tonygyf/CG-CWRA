"""Metrics used in CG-CWRA experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def accuracy(predictions: Iterable[int], labels: Iterable[int]) -> float:
    """Compute exact-match accuracy."""

    pred = np.asarray(list(predictions))
    target = np.asarray(list(labels))
    if target.size == 0:
        return 0.0
    return float((pred == target).mean())


def cw_entity_pair_accuracy(
    predictions: Iterable[int], labels: Iterable[int], cross_window_flags: Iterable[bool]
) -> float:
    """Accuracy on the subset that requires cross-window entity association."""

    subset_pred = []
    subset_target = []
    for pred, target, flag in zip(predictions, labels, cross_window_flags):
        if flag:
            subset_pred.append(pred)
            subset_target.append(target)
    if not subset_target:
        return 0.0
    return accuracy(subset_pred, subset_target)

