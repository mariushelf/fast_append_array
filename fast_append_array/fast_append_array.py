from numbers import Number
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import pandas as pd


class FastAppendBase:
    __slots__ = ["dtype", "col_indexes"]

    def __init__(
        self, cols: List[str], dtype: np.dtype = np.float64, col_indexes: Dict = None
    ):
        self.dtype = dtype
        if cols is None and col_indexes is None:
            raise ValueError("must provide exactly one of cols and col_indexes")
        elif cols is not None and col_indexes is not None:
            raise ValueError("must provide exactly one of cols and col_indexes")
        self.col_indexes: Dict[str, int] = col_indexes or {
            n: i for i, n in enumerate(cols)
        }

    def __repr__(self):
        return repr(self.a)

    def __array__(self):
        """ subclasses are encouraged to supply a fast implementation """
        return self.a

    def __len__(self):
        """ sublcasses are encouraged to supply a fast implementation """
        return len(self.a)

    @property
    def shape(self):
        return self.a.shape

    def __getitem__(self, items):
        """Slicing. Always returns a new instance of this class, with the
        exception of a single string index.

        Parameters
        ----------
        items:
            * A single numerical index refers to a specific row
            * a single string value refers to a column and returns the column
              as a numpy array of shape (n,)
            * a slice and a column name: return the given slice of the given column
              as np.array
            * an integer and a column name: return the value of a row in a column as scalar
            * a list of strings refers to column names
            * a slice and a list of column names refers to the given slice in the
              given columns

        Returns
        -------
        result : FastAppendArray or np.array
            depending on inputs, see above

        """
        if isinstance(items, str):
            # single column name
            idx = self.col_indexes[items]
            return self.a[:, idx]
        elif isinstance(items, Number):
            data = self.a[[items], :]
            return FastAppendArray(self.columns, data, dtype=self.dtype)
        elif isinstance(items, slice):
            data = self.a[items]
            return FastAppendArray(self.columns, data, dtype=self.dtype)
        elif isinstance(items, tuple):
            # several dimensions
            rows, cols = items
            if isinstance(cols, str):
                return self[cols][rows]
            if isinstance(rows, Number):
                rows = [rows]
            col_idxs = self.names2idx(cols)
            col_idxs = self.optimize_indexing(col_idxs)
            return FastAppendArray(cols, self.a[rows][:, col_idxs])
        elif isinstance(items, list):
            # list of indices for axis 0, or column list
            e0 = items[0]
            if isinstance(e0, str):
                # list of column names
                cols = items
                idx = self.names2idx(cols)
                idx = self.optimize_indexing(idx)
                data = self.a[:, idx]
                return FastAppendArray(cols, data, dtype=self.dtype)
            else:
                items = self.optimize_indexing(items)
                cols = self.columns
                data = self.a[items, :]
                return FastAppendArray(cols, data, dtype=self.dtype)

    @classmethod
    def optimize_indexing(cls, indices: List[int]) -> Union[slice, List]:
        """Optimize index by trying to convert it into a slice.

        When indexing a numpy array with a list it generates a copy, but when
        indexing with a slice it return a view, which is much faster.

        This function tries to convert a list of indices into a slice. This works
        only if the indices are a list of increasing, contiguous integers.

        Parameters
        ----------
        indices : List[int]
            list of indexes

        Returns
        -------
        optimized : List or slice
            an optimized slice or list giing the same result as `indices` when used
            to index a numpy array

        Examples
        --------
        >>> FastAppendArray.optimize_indexing([1, 2, 3])
        slice(1, 4)

        >>> FastAppendArray.optimize_indexing([1, 2, 4])  # can't be represented as slice
        [1, 2, 4]
        """
        if len(indices) == 0:
            return slice(0, 0)
        elif len(indices) <= 1:
            return slice(indices[0], indices[0] + 1)

        shift = indices[1:]
        diff = shift[0] - indices[0]

        contiguous = all([s - o == diff for o, s in zip(indices, shift)])

        if contiguous:
            stop = indices[-1]
            if stop == 0 and diff < 0:
                return indices
            else:
                return slice(indices[0], indices[-1] + np.sign(diff), diff)
        else:
            return indices

    def append(self, rows: np.array, cols: List[str] = None) -> None:
        raise NotImplementedError

    def append_dict(self, rowdict: Dict[str, Any]) -> None:
        """Append one row from a dictionary.

        Subclasses are encouraged to provide an optimized implementation. See
        `FastAppendArray.append_dict()` for an example.

        Parameters
        ----------
        rowdict : dict of str -> float
            dictionary. The keys are column names, the values the respective values
            in the new row. `rowdict` must contain an entry for each of the DataFrame's
            columns. Additional elements in `rowdict` will be ignored or raise an
            exception, depending on the implementation.
        """
        values = []
        cols = []
        for c, v in rowdict.items():
            values.append(v)
            cols.append(c)
        self.append(np.array(values), cols)

    @property
    def columns(self) -> List[str]:
        """ list of column names """
        raise NotImplementedError

    @property
    def a(self) -> np.array:
        """ Return data of this dataframe as numpy array """
        raise NotImplementedError

    def set_data(
        self,
        data: np.array,
        cols: List[str] = None,
        row: int = 0,
        size: str = "match",
    ) -> None:
        """Set data.

        The length of the FAA is adjusted depending on the `size` parameter.

        Parameters
        ----------
        data : ndarary of shape (n_rows, n_cols)
            data to put
        cols : list of length n_cols
            names of target columns
        row : int, optional
            index of first target row. The data will be written to rows
            `row`:`row`+`n_rows`-1. Can be greater than the current length if the
            `length` parameter is set to 'extend'.
        size : {'match', 'extend'}, optional
            if 'match', `row` + `len(data)` must be exactly the length of the current
            data. Exception: if the array is empty, treat the same as 'extend'.
            When `size` is 'extend', the length of the FAA is extended if the end of the
            newly written data is beyond the current length. Default 'match'
        """
        raise NotImplementedError

    def names2idx(self, names: List[str]) -> List[int]:
        """ Map list of column names to list of column indices """
        return [self.col_indexes[n] for n in names]

    @property
    def df(self) -> pd.DataFrame:
        """Return all data as pd.DataFrame

        Returns
        -------
        df :
            dataframe with all values
        """
        return pd.DataFrame(self.a, columns=self.columns)

    def values(self, cols: List[str] = None) -> np.array:
        """Return all data in the provided columns as numpy array

        Parameters
        ----------
        cols
            columns to return. If None, all columns will be returned

        Returns
        -------
        :
            numpy array of shape [n, n_columns]
        """
        if cols is None:
            return self.a
        else:
            return self.a[:, self.names2idx(cols)]

    def row_dict(self, idx, cols: List[str] = None) -> Dict[str, float]:
        """Return dictionary from column name to data at an index.

        Parameters
        ----------
        idx
            index of data to return. If ommitted, return current data
        cols
            columns to return. If empty, return all columns

        Returns
        -------
        dict
            dictionary from column name to value at `idx`
        """
        if cols is None:
            return {c: self.a[idx, cidx] for cidx, c in enumerate(self.columns)}
        else:
            return {c: self.a[idx, self.col_indexes[c]] for c in cols}


class FastAppendArray(FastAppendBase):
    """A wrapper around numpy arrays which allows for fast appends and access
    by column names.

    Works by pre-allocating memory for an np.array and expanding it in batches
    as data gets added.

    Performance: the performance for small arrays is much worse than for pure numpy
    arrays. Also the retrieval of a column subset by name or by index or slice is
    slower than for numpy arrays irrespectively of the array's size.

    Append operations are significantly faster once a certain size (about 10,000 rows)
    is reached. The speedup for the insertion of 100,000 rows is about 8 fold. For
    1,000,000 rows numpy needs minutes to insert them one by one, whereas this class
    needs just a few seconds. Pandas is already unbearably slow for just 10,000 inserts.

    Datatypes: unlike pandas, this class only supports a single data type because the
    underlying data structure is a numpy array. It defaults to float64, and at the
    moment there is no way to explicitly overwrite this default.

    Parameters
    ----------
    cols : list-like
        column names
    data : array-like, optional
        initial data. The first axis must have the same size os the length of `cols`
    initial_length : positive int
        amount of rows to pre-allocate
    increment : positive float
        once the data grows beyond the current allocation, increment by this fraction.
        If the current allocation is 1000 rows, and the increment is 0.2, an allocation
        step will increase the allocated size to 1200 rows
    """

    __slots__ = ["data", "length", "increment"]

    def __init__(
        self,
        cols: List[str],
        data: np.array = None,
        initial_length: int = 1000,
        increment: float = 0.2,
        dtype: np.dtype = np.float64,
        col_indexes: Dict[str, int] = None,
    ):
        if data is not None and dtype is not None and data.dtype != dtype:
            raise TypeError(
                f"dtype and data.dtype must match, but {dtype} != {data.dtype}"
            )
        if data is not None:
            dtype = dtype or data.dtype

        super().__init__(cols, dtype=dtype, col_indexes=col_indexes)

        if data is not None:
            if data.shape[1] != len(self.col_indexes):
                raise ValueError(
                    f"first dimension of data != number of columns ({data.shape[1]} != {len(self.col_indexes)})"
                )
            self.data = data
            self.length = data.shape[0]
        else:
            self.data = np.empty((initial_length, len(cols)), dtype=dtype)
            self.length = 0

        self.increment = increment

    @staticmethod
    def from_pandas(df, dtype: np.dtype = None, **kwargs) -> "FastAppendArray":
        """Construct a FastAppendArray from a pandas DataFrame.

        The index of the DataFrame is ignored.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame
        dtype : np.dtype
            dtype of the FastAppendArray. If it differs from the DataFrame's dtype,
            the function tries to cast it.
        kwargs : dict
            forwarded to `FastAppendArray`'s constructor
        """
        a = df.to_numpy()
        dtype = dtype or a.dtype
        if dtype != a.dtype:
            a = a.astype(dtype)
        return FastAppendArray(list(df.columns), a, dtype=dtype, **kwargs)

    @staticmethod
    def from_dicts(
        data: Iterable[Dict[str, Any]], dtype: np.dtype = np.float64, **kwargs
    ) -> "FastAppendArray":
        """ construct a `FastAppendArray` from a list of row dicts """
        cols = kwargs.pop("cols", None)
        df = None
        for d in data:
            if cols is None:
                cols = list(d.keys())
            if df is None:
                df = FastAppendArray(cols=cols, dtype=dtype, **kwargs)
            df.append_dict(d)
        return df

    def __repr__(self):
        try:
            return repr(self.data[: self.length])
        except AttributeError:
            return f"{self.__class__} (cols={self.columns})"

    def __array__(self):
        return self.data[: self.length]

    def __len__(self):
        return self.length

    @property
    def shape(self):
        return self.a.shape

    def append(self, rows: np.array, cols: List[str] = None) -> None:
        """Append rows to the data.

        Parameters
        ----------
        rows : array-like, shape [k, ncols] or [ncols]
            rows to append. If rows has one dimension less than the data, it is
            treated as a single row.
        cols : list of str, optional
            order of columns contained in the new data. If omitted, `self.columns`
            is used
        """
        s = rows.shape
        if len(s) == len(self.data.shape) - 1:
            # one dimension less than data; treat as row and add one dimension
            rows = np.expand_dims(rows, axis=0)
        elif len(s) == len(self.data.shape):
            pass
        else:
            raise ValueError(
                f"rows has {len(rows.shape)} dimensions, which is incompatible "
                f"to {len(self.data.shape)} dimensions in current data"
            )

        if cols and rows.shape[1] != len(cols):
            raise ValueError(
                f"expected {len(cols)} columns in data, but was {rows.shape[1]}"
            )
        elif rows.shape[1] != self.data.shape[1]:
            raise ValueError(
                f"expected {self.data.shape[1]} columns in data, but was {rows.shape[1]}"
            )

        if rows.shape[2:] != self.data.shape[2:]:
            raise ValueError(
                f"length of row differs from width of data "
                f"({rows.shape[2:]} != {self.data.shape[2:]})"
            )
        new_length = self.length + len(rows)
        if len(self.data) < new_length:
            self._allocate_new_rows(rows.shape[0])
        if cols is None:
            self.data[self.length : new_length] = rows
        else:
            idxs = self.names2idx(cols)
            self.data[self.length : new_length, idxs] = rows
        self.length = new_length

    @property
    def a(self) -> np.array:
        """ Return data of this dataframe as numpy array """
        return self.data[: self.length, :]

    def replace_data(self, data: np.array):
        """Replaces the data of this array.

        This can be used to replace the data without instantiating a new instance,
        and is hence faster.

        Parameters
        ----------
        data : np.array of shape (n, len(self.columns))
            data to be set. Can be of arbitrary length, but the width must match
            the current width of this array.
        """
        s = data.shape
        if s[1] != len(self.columns):
            raise ValueError(
                "The width of the new data must match the current widths,"
                f"but {s[1]} != {len(self.columns)}"
            )
        self.data = data
        self.length = s[0]

    def set_data(
        self, data: np.array, cols: List[str] = None, row: int = 0, size: str = "match"
    ) -> None:
        if self.length and size == "match" and row + len(data) != self.length:
            raise ValueError(
                f"length of new data must match length of current data ({row + len(data)} != {self.length})"
            )

        if cols is None:
            if data.shape[1] != len(self.columns):
                raise ValueError(
                    f"width of data must match, but {data.shape[1]} != {len(self.columns)}"
                )
            if len(self.data) < row + len(data):
                self._allocate_new_rows(max(0, row + len(data) - self.length))

            self.data[row : row + len(data)] = data

        else:
            if len(self.data) < row + len(data):
                self._allocate_new_rows(max(0, row + len(data) - self.length))

            self.data[row : row + len(data), self.names2idx(cols)] = data

        self.length = max(self.length, row + len(data))

    def append_dict(self, rowdict: Dict[str, Any]) -> None:
        row = [rowdict[n] for n in self.col_indexes]
        new_length = self.length + 1
        if len(self.data) < new_length:
            self._allocate_new_rows(1)
        self.data[self.length, :] = row
        self.length += 1

    @property
    def columns(self) -> List[str]:
        """ list of column names """
        return list(self.col_indexes.keys())

    def _allocate_new_rows(self, min_increment=None):
        """Allocate new empty rows.

        Parameters
        ----------
         min_increment : int, optional
            minimum number of rows to add.
            If None, the size of the data is increased using the increment argument
            of this class

        """
        new_rows = max(1, int(np.ceil(len(self.data) * self.increment)))
        if min_increment:
            new_rows = max(new_rows, min_increment - new_rows)
        append_data = np.empty([new_rows] + list(self.data.shape[1:]), dtype=self.dtype)
        self.data = np.concatenate([self.data, append_data])


class FastAppendArrayView(FastAppendBase):
    __slots__ = ["parent", "length", "col_mapping", "all_col_idxs"]

    def __init__(self, parent: FastAppendBase, cols: Dict[str, str], length=None):
        self.col_mapping = cols
        super().__init__(cols=list(self.col_mapping.keys()), dtype=parent.dtype)

        self.parent = parent
        self.all_col_idxs = self.optimize_indexing(
            parent.names2idx(list(cols.values()))
        )

        if length is not None:
            if length > len(parent):
                raise ValueError(("length cannot be longer than the parent's length"))
            self.length = length
        else:
            self.length = len(parent)

    def __len__(self):
        return self.length

    def append(self, rows: np.array, cols: List[str] = None) -> None:
        pcols = [self.col_mapping[c] for c in cols] if cols is not None else None
        self.parent.set_data(rows, cols=pcols, row=self.length, size="extend")
        self.length += len(rows)

    @property
    def a(self) -> np.array:
        return self.parent.a[: self.length, self.all_col_idxs]

    def set_data(
        self,
        data: np.array,
        cols: List[str] = None,
        row: int = 0,
        size: str = "match",
    ) -> None:
        if size == "match":
            if self.length and row + len(data) != self.length:
                raise ValueError("new data must end at length of current array")
            self.length = row + len(data)
        elif size == "extend":
            self.length = max(self.length, row + len(data))
        else:
            raise ValueError(f"size must be one of 'match' or 'extend', but was {size}")

        pcols = [
            self.col_mapping[c] for c in cols or self.columns
        ]  # if cols is not None else self.col_mapping
        self.parent.set_data(data, pcols, row=row, size=size)

    @property
    def columns(self) -> List[str]:
        """ list of column names """
        return list(self.col_mapping.keys())
