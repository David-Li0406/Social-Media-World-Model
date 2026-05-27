"""Sanity tests for ranking metrics."""
import math

from smwm.metrics.ranking import pairwise_accuracy, spearman


def test_spearman_perfect_scale():
    # The user's example: same order at very different scales.
    rho = spearman([100, 200, 300], [10, 20, 30])["spearman_rho"]
    assert math.isclose(rho, 1.0)


def test_spearman_reversed():
    rho = spearman([300, 200, 100], [10, 20, 30])["spearman_rho"]
    assert math.isclose(rho, -1.0)


def test_pairwise_accuracy_perfect():
    assert pairwise_accuracy([100, 200, 300], [10, 20, 30]) == 1.0


def test_pairwise_accuracy_reversed():
    assert pairwise_accuracy([300, 200, 100], [10, 20, 30]) == 0.0
