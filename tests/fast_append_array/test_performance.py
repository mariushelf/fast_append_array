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


# @pytest.mark.thirdparty
# @pytest.mark.parametrize(
#     "mode",
#     ["pd", "np", "faa", "faa_a"],
# )
# def test_access_single_array_col(array, df, faa, mode, benchmark):
#     if mode == "pd":
#         benchmark(lambda: df["m"])
#     elif mode == "np":
#         benchmark(lambda: array[:, 13])
#     elif mode == "faa":
#         benchmark(lambda: faa["m"])
#     elif mode == "faa_a":
#         benchmark(lambda: faa.a[:, 13])


# def test_array_access(faa, benchmark):
#     def f():
#         faa.a
#
#     benchmark(f)


@pytest.mark.parametrize(
    "nrows,ncols",
    [
        (100, 10),
        (10_000, 10),
        (20_000, 10),
        (20_000, 20),
        (100_000, 10),
        (100_000, 20),
        (1_000_000, 20),
        (100_000, 200),
    ],
)
def test_append(nrows, ncols, benchmark):
    all_cols = list("abcdefghijklmnopqrstuvwxyz")
    cols = all_cols[:ncols]
    row = {c: 0 for c in cols}

    def append():
        faa = FastAppendArray(cols=cols)
        for _ in range(nrows):
            faa.append_dict(row)
        return faa

    faa = benchmark.pedantic(append, warmup_rounds=0, iterations=3, rounds=1)
    assert len(faa) == nrows


@pytest.mark.thirdparty
@pytest.mark.parametrize(
    "nrows,ncols",
    [
        (100, 10),
        (10000, 10),
        (20000, 10),
        (20000, 20),
        # (100000, 10),
        # (100000, 20),
        # (100000, 200),
    ],
)
def test_append_np_concat(nrows, ncols, benchmark):
    row = np.zeros([1, ncols])

    def append():
        a = np.zeros([0, ncols])
        for _ in range(nrows):
            a = np.concatenate([a, row], axis=0)
        return a

    a = benchmark.pedantic(append, warmup_rounds=0, iterations=1, rounds=1)
    assert len(a) == nrows


@pytest.mark.thirdparty
@pytest.mark.parametrize(
    "nrows,ncols",
    [
        (100, 10),
        (10000, 10),
        (20000, 10),
        (20000, 20),
        # (100000, 10),
        # (100000, 20),
        # (100000, 200),
    ],
)
def test_append_np_append(nrows, ncols, benchmark):
    row = np.zeros([1, ncols])

    def append():
        a = np.zeros([0, ncols])
        for _ in range(nrows):
            a = np.append(a, row, axis=0)
        return a

    a = benchmark.pedantic(append, warmup_rounds=0, iterations=1, rounds=1)
    assert len(a) == nrows


@pytest.mark.thirdparty
@pytest.mark.parametrize(
    "nrows,ncols",
    [
        (100, 10),
        (1000, 10),
        # (10000, 10),
        # (10000, 20),
        # (100000, 10),
        # (100000, 20),
        # (100000, 200),
    ],
)
def test_append_pd(nrows, ncols, benchmark):
    all_cols = list("abcdefghijklmnopqrstuvwxyz")
    cols = all_cols[:ncols]
    row = pd.DataFrame([{c: 0 for c in cols}])

    def append():
        df = pd.DataFrame([], columns=cols)
        for _ in range(nrows):
            df = df.append(row)
        return df

    df = benchmark.pedantic(append, warmup_rounds=0, iterations=3, rounds=1)
    assert len(df) == nrows
