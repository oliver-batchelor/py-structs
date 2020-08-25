from typing import List
from collections import Counter
from collections.abc import Mapping
import numpy as np

from numbers import Number
import math

import operator
import itertools
from functools import partial

from .struct import struct, Struct, transpose_structs, map_type


class Table(Struct):
    def __init__(self, d:dict):

        assert len(d) > 0, "empty Table"
        t = next(iter(d.values()))
      
        for (k, v) in d.items():
            assert type(v) == np.ndarray, "expected ndarray, got " + type(t).__name__
            assert v.shape[0] == t.shape[0], "mismatched column sizes: " + str(shape(d))

        super(Table, self).__init__(d) 

    @staticmethod
    def from_structs(structs : List[Struct]) -> 'Table':
        struct_lists = transpose_structs(structs)
        return Table(struct_lists._map(np.stack).__dict__)
   
    @staticmethod 
    def build(**d):
        return Table(d)

    def __getitem__(self, index:int) -> np.ndarray:
        return self.__dict__[index]        

    def _index_select(self, index:np.ndarray) -> 'Table':
        if isinstance(index, np.ndarray):
            assert index.dtype == torch.int64 
            assert index.dim() == 1
        else:
            assert isinstance(index, int), "Table.index_select: unsupported index type" + type(index).__name__

        return self._map(lambda t: t[index])


    def _index(self, index:int) -> np.ndarray:
        return self._index_select(np.ndarray([index], dtype=int))

        
    def _narrow(self, start, n):
        return self._map(lambda t: t.narrow(0, start, n))


    def _take(self, n):
        return self._narrow(0, min(self._size, n))

    def _drop(self, n):
        n = min(self._size, n)
        return self._narrow(n, self._size - n)


    def _sequence(self):
        return (self._index_select(i) for i in range(0, self._size))

    def _sort_on(self, key, descending=False):
        assert key in self
        assert self[key].dim() == 1

        values, inds = self[key].sort(descending = descending)
        return Table({k: values if k == key else v[inds] for k, v in self.items()})

    @property
    def _head(self):
        return next(iter(self.__dict__.values()))

    @property
    def _size(self):
        return self._head.shape[0]

    @property
    def _device(self):
        return self._head.device

    def _to(self, device):
        return self._map(lambda t: t.to(device))

    def _cpu(self):
        return self._map(lambda t: t.cpu())



def table(**d):
    return Table(d)




class Histogram:
    def __init__(self, values = np.array([0]), range = (0, 1), num_bins = 10, trim = True):
        assert len(range) == 2

        self.range = range
        lower, upper = self.range

        bin_indexes = (values - lower) * num_bins / (upper - lower) 
        bin_indexes = bin_indexes.long()

        if trim:
            valid = (bin_indexes >= 0) & (bin_indexes < num_bins)

            values = values[valid]
            bin_indexes = bin_indexes[valid]
        
        bin_indexes.clamp_(0, num_bins - 1)

        self.sum         = values.sum().item()
        self.sum_squares = values.norm(2).item()
        self.counts = np.bincount(bin_indexes, minlength = num_bins)

    def __repr__(self):
        return self.counts.tolist().__repr__()

    def bins(self):
        lower, upper = self.range
        d = (upper - lower) / self.counts.shape[0]

        return np.array([lower + i * d for i in range(0, self.counts.shape[0] + 1)])

    def to_struct(self):
        return struct(sum=self.sum, sum_squares=self.sum_squares, counts=self.counts)

    def __add__(self, other):
        assert isinstance(other, Histogram)
        assert other.counts.shape[0] == self.counts.shape[0], "mismatched histogram sizes"
        
        total = Histogram(range = self.range, num_bins = self.counts.shape[0])
        total.sum = self.sum + other.sum
        total.sum_squares = self.sum_squares + other.sum_squares
        total.counts = self.counts + other.counts

        return total


    def __truediv__(self, x):
        assert isinstance(x, Number)

        total = Histogram(range = self.range, num_bins =  self.counts.shape[0])
        total.sum = self.sum / x
        total.sum_squares = self.sum_squares / x
        total.counts = self.counts / x

        return total

    @property
    def std(self):

        n = self.counts.sum().item()
        if n > 1:
            sum_squares = self.sum_squares - (self.sum * self.sum / n)
            var = max(0, sum_squares / (n - 1))

            return math.sqrt(var)
        else:
            return 0

    @property
    def mean(self):
        n = self.counts.sum().item()
        if n > 0:
            return self.sum / self.counts.sum().item()
        else:
            return 0

def map_arrays(data, f, *args, **kwargs):
    return map_type(data,  np.ndarray, partial(f, *args, **kwargs))  

def shape_info(x):
    return map_arrays(x, lambda x: tuple([*x.shape, type(x), x.dtype]))

def shape(x):
    return map_arrays(x, lambda x: tuple(x.shape))

def arrays_astype(t, **kwargs):
    return map_arrays(t, np.ndarray.astype, **kwargs)        


def stack_tables(tables, dim=0):
    t = transpose_structs(tables)
    return Table(dict(t._map(np.stack, axis=dim))) 

def cat_tables(tables, dim=0):
    t = transpose_structs(tables)
    return Table(dict(t._map(np.concatenate, axis=dim))) 
