import numpy as np
import pytest
from numpy.testing import assert_array_equal

from fast_append_array import FastAppendArray, FastAppendArrayView


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    return rng.random([4, 3])


@pytest.fixture
def cols():
    return list("ABC")


@pytest.fixture
def faa(data, cols):
    return FastAppendArray(cols, data)


@pytest.fixture(scope="function")
def faav(faa):
    return FastAppendArrayView(faa, {"X": "C", "Y": "A"})


def test_columns(faav):
    assert faav.columns == ["X", "Y"]


def test_getitem_single_colname_and_slice(faa, faav):
    res = faav[:2, "X"]
    assert res.shape == (2,)
    assert_array_equal(res, faa[:2, "C"])


def test_getitem_single_colname(faa, faav):
    res = faav["X"]
    assert res.shape == (4,)
    assert_array_equal(res, faa["C"])


def test_getitem_collist(faa, faav):
    res = faav[["X", "Y"]]
    assert_array_equal(res.a, faa[["C", "A"]].a)
    assert res.columns == ["X", "Y"]


def test_getitem_single_list(faa, faav):
    res = faav[[1, 0]]
    assert res.shape == (2, 2)
    assert res.columns == ["X", "Y"]
    assert_array_equal(res.a, faa[[1, 0], ["C", "A"]].a)


def test_getitem_singleidx(faa, faav):
    res = faav[2]
    assert res.columns == ["X", "Y"]
    resa = res.a
    assert_array_equal(resa, faa[2, ["C", "A"]].a)
    assert res.shape == (1, 2)


def test_getitem_2dim(faa, faav):
    res = faav[:3, ["X", "Y"]]
    assert res.shape == (3, 2)
    assert_array_equal(res.a, faa[:3, ["C", "A"]])
    assert res.columns == ["X", "Y"]


def test_getitem_cols_and_idx_raises(faav):
    with pytest.raises(TypeError):
        faav["X", 1]

    with pytest.raises(TypeError):
        faav[["X"], 1]


def test_getitem_slice(faa, faav):
    res = faav[:2]
    assert res.shape == (2, 2)
    assert res.columns == ["X", "Y"]
    assert_array_equal(res.a, faa[:2, ["C", "A"]])
