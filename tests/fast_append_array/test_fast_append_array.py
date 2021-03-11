import numpy as np
import pytest
from numpy.testing import assert_array_equal

from fast_append_array import FastAppendArray


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


def test_getitem_single_colname_and_slice(data, cols, faa):
    res = faa[:2, "B"]
    assert res.shape == (2,)
    assert_array_equal(res, data[:2, 1])


def test_getitem_single_colname(data, cols, faa):
    res = faa["B"]
    assert res.shape == (4,)
    assert_array_equal(res, data[:, 1])


def test_getitem_collist(data, cols, faa):
    res = faa[["A", "C"]]
    assert_array_equal(res.a, data[:, [0, 2]])
    assert res.columns == ["A", "C"]


def test_getitem_single_list(data, cols, faa):
    res = faa[[1, 2]]
    assert res.shape == (2, 3)
    assert res.columns == cols
    assert_array_equal(res.a, data[[1, 2], :])


def test_getitem_singleidx(data, cols, faa):
    res = faa[2]
    assert res.columns == cols
    assert_array_equal(res.a, data[[2]])
    assert res.shape == (1, 3)


def test_getitem_2dim(data, cols, faa):
    res = faa[:3, ["A", "B"]]
    assert res.shape == (3, 2)
    assert_array_equal(res.a, data[:3, :-1])
    assert res.columns == ["A", "B"]


def test_getlist_and_cols(data, faa):
    res = faa[[1, 0], ["A", "B"]]
    assert res.shape == (2, 2)
    assert_array_equal(res.a, data[[1, 0], :-1])
    assert res.columns == ["A", "B"]


def test_getrow_and_col(data, faa):
    res = faa[3, "A"]
    assert isinstance(res, float)
    assert res == data[3, 0]


def test_getrow_and_cols(data, faa):
    res = faa[3, ["A", "B"]]
    assert res.shape == (1, 2)
    assert_array_equal(res.a, data[[3], :-1])
    assert res.columns == ["A", "B"]


def test_getitem_cols_and_idx_raises(faa):
    with pytest.raises(TypeError):
        faa["A", 1]

    with pytest.raises(TypeError):
        faa[["A"], 1]


def test_getitem_slice(faa, cols, data):
    res = faa[:2]
    assert res.shape == (2, 3)
    assert res.columns == cols
    assert_array_equal(res.a, data[:2, :])


@pytest.mark.parametrize(
    "indices,expected",
    [
        ([], slice(0, 0)),
        ([42], slice(42, 43)),
        ([4, 5, 6], slice(4, 7, 1)),
        ([4, 6, 8], slice(4, 9, 2)),
        ([4, 6, 9], [4, 6, 9]),
        ([6, 4, 2], slice(6, 1, -2)),
        ([4, 2, 0], [4, 2, 0]),
    ],
)
def test_optimize_indexing(indices, expected):
    res = FastAppendArray.optimize_indexing(indices)
    assert res == expected
