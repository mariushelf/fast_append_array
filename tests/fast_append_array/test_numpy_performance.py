import numpy as np
import pytest


@pytest.fixture
def rng():
    rng = np.random.default_rng(42)
    return rng


@pytest.fixture
def array(rng):
    return rng.random([100, 100])


@pytest.mark.thirdparty
@pytest.mark.benchmark(group="numpy slice vs fancy slice")
class TestSliceVsFancySlice:
    def test_slice(self, array, benchmark):
        # this creates a view
        benchmark(lambda: array[0:5])

    def test_fancy_slice(self, array, benchmark):
        # this creates a copy and is much slower
        idx = list(range(5))
        benchmark(lambda: array[idx])


@pytest.mark.thirdparty
@pytest.mark.benchmark(group="numpy concat vs slice")
class TestConcatVsSlice:
    """ what's faster? Slicing by list or concat single lists? """

    def test_concat(self, rng, benchmark):
        a1 = rng.random([1000000, 1])
        a2 = rng.random([1000000, 1])
        benchmark(lambda: np.concatenate([a1, a2]))

    def test_slice(self, rng, benchmark):
        # this is much faster than concat
        a = rng.random([1000000, 4])
        benchmark(lambda: a[[0, 2]])


@pytest.mark.thirdparty
@pytest.mark.benchmark(group="array access with different dtypes")
@pytest.mark.parametrize(
    "dtype", ["float64", "float32", "float16", "object", "int64", "int32"]
)
class TestDtypePerformance:
    def test_col_access(self, dtype, benchmark):
        a = np.zeros((100_000, 100), dtype=dtype)
        benchmark(lambda: a[:, 50])

    def test_element_access(self, dtype, benchmark):
        a = np.zeros((100_000, 100), dtype=dtype)
        benchmark(lambda: a[-1, 50])
