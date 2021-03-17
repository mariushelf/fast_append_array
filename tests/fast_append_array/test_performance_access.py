import numpy as np
import pandas as pd
import pytest

from fast_append_array import FastAppendArray


@pytest.fixture
def array():
    rng = np.random.default_rng(42)
    return rng.random([1_000_000, 26])


@pytest.fixture
def cols():
    return list("abcdefghijklmnopqrstuvwxyz")


@pytest.fixture
def df(array, cols):
    return pd.DataFrame(array, columns=cols)


@pytest.fixture
def faa(array, cols):
    return FastAppendArray(cols, array)


@pytest.mark.thirdparty
@pytest.mark.parametrize(
    "mode",
    ["pd", "np", "faa", "faa_a"],
)
def test_access_single_array_col(array, df, faa, mode, benchmark):
    if mode == "pd":
        benchmark(lambda: df["m"])
    elif mode == "np":
        benchmark(lambda: array[:, 13])
    elif mode == "faa":
        benchmark(lambda: faa["m"])
    elif mode == "faa_a":
        benchmark(lambda: faa.a[:, 13])


@pytest.mark.parametrize(
    "mode",
    ["pd", "np", "faa", "faa_a"],
)
def test_access_element(array, df, faa, mode, benchmark):
    s = (1000, 2)
    if mode == "pd":
        benchmark(lambda: df.iloc[s])
    elif mode == "np":
        benchmark(lambda: array[s])
    elif mode == "faa":
        benchmark(lambda: faa[1000, "c"])
    elif mode == "faa_a":
        benchmark(lambda: faa.a[s])


@pytest.mark.parametrize(
    "mode",
    ["pd", "np", "faa", "faa_a"],
)
def test_slice_rows(array, df, faa, mode, benchmark):
    s = slice(1000, 2000)
    if mode == "pd":
        benchmark(lambda: df.iloc[s])
    elif mode == "np":
        benchmark(lambda: array[s])
    elif mode == "faa":
        benchmark(lambda: faa[s])
    elif mode == "faa_a":
        benchmark(lambda: faa.a[s])


def test_array_access(faa, benchmark):
    def f():
        faa.a

    benchmark(f)
