"""
Tests for experiments/journal_tests.py: the statistics of the paper
(exact permutation test, Holm correction, Cliff's delta) and the list of
pre-specified tests, which must match the plan header (F1 9, F2 6, F3 3).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO))

jt = pytest.importorskip("journal_tests")


def test_holm_matches_step_down():
    # sorted p 0.01, 0.03, 0.04, 0.2 -> x4, x3, x2, x1, then running maximum
    assert jt.holm([0.01, 0.04, 0.03, 0.2]) == pytest.approx([0.04, 0.09, 0.09, 0.2])
    assert jt.holm([0.5, 0.9]) == [1.0, 1.0]


def test_cliffs_delta_extremes():
    assert jt.cliffs_delta(np.array([1, 2, 3]), np.array([4, 5, 6])) == 1.0
    assert jt.cliffs_delta(np.array([4, 5, 6]), np.array([1, 2, 3])) == -1.0


def test_permutation_minimum_p_for_5_vs_5():
    a, b = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
    assert math.comb(10, 5) == 252
    assert jt.perm_test(a, b) == pytest.approx(1 / 252)                 # one-sided, a lower
    assert jt.perm_test(a, b, two_sided=True) == pytest.approx(2 / 252)
    assert jt.perm_test(b, a) == pytest.approx(1.0)                     # wrong direction


def test_pre_specified_tests_match_the_plan():
    tests = jt.preregistered()
    fams = [t.family for t in tests]
    assert (fams.count("F1"), fams.count("F2"), fams.count("F3")) == (9, 6, 3)
    assert [t for t in tests if t.two_sided] == [
        jt.Test("F3", "con", "nca-pool-ps", "locality-ps", "p95", two_sided=True)]
    assert len({t.id for t in tests}) == 18
