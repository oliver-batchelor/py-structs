from typing import List
from collections import Counter
from collections.abc import Mapping, Sequence

import torch
import numpy as np

from numbers import Number
import math

import operator
import itertools

from .struct import struct, Struct, transpose_structs, map_type
from functools import partial

class Table(Struct):
    def __init__(self, d:dict):

        assert len(d) > 0, "empty Table"
        t = next(iter(d.values()))
      
        for (k, v) in d.items():
            assert type(v) == torch.Tensor, "expected tensor, got " + type(t).__name__
            assert v.size(0) == t.size(0), "mismatched column sizes: " + str(shape(d))

        super(Table, self).__init__(d) 

    @staticmethod
    def from_structs(structs : List[Struct]) -> 'Table':
        struct_lists = transpose_structs(structs)
        return Table(struct_lists._map(torch.stack).__dict__)
   
    @staticmethod 
    def build(**d):
        return Table(d)

    def __getitem__(self, index:int) -> torch.Tensor:
        return self.__dict__[index]        

    def _index_select(self, index:torch.Tensor) -> 'Table':
        if type(index) is torch.Tensor:
            assert index.dtype == torch.int64 
            assert index.dim() == 1
            
            return self._map(lambda t: t[index])

        elif type(index) is int:
            return Struct({k: v[index] for k, v in self.items()})
        assert False, "Table.index_select: unsupported index type" + type(index).__name__


    def _index(self, index:int) -> torch.Tensor:
        return self._index_select(torch.tensor([index], dtype=int))

        
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
        return self._head.size(0)

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
    def __init__(self, values = torch.FloatTensor(0), range = (0, 1), num_bins = 10, trim = True):
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
        self.counts = torch.bincount(bin_indexes, minlength = num_bins)

    def __repr__(self):
        return self.counts.tolist().__repr__()

    def bins(self):
        lower, upper = self.range
        d = (upper - lower) / self.counts.size(0)

        return torch.FloatTensor([lower + i * d for i in range(0, self.counts.size(0) + 1)])

    def to_struct(self):
        return struct(sum=self.sum, sum_squares=self.sum_squares, counts=self.counts)

    def __add__(self, other):
        assert isinstance(other, Histogram)
        assert other.counts.size(0) == self.counts.size(0), "mismatched histogram sizes"
        
        total = Histogram(range = self.range, num_bins = self.counts.size(0))
        total.sum = self.sum + other.sum
        total.sum_squares = self.sum_squares + other.sum_squares
        total.counts = self.counts + other.counts

        return total


    def __truediv__(self, x):
        assert isinstance(x, Number)

        total = Histogram(range = self.range, num_bins =  self.counts.size(0))
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


def map_tensors(data, f, *args, **kwargs):
    return map_type(data, torch.Tensor, partial(f, *args, **kwargs))  

def shape_info(x):
    return map_tensors(x, lambda x: tuple([*x.shape, type(x), x.dtype]))

def shape(x):
    return map_tensors(x, lambda x: tuple(x.shape))


def from_numpy(data):
    return map_type(data, np.ndarray, torch.Tensor.from_numpy)  

def to_numpy(data):
    return map_type(data, torch.Tensor, torch.Tensor.numpy)  



def tensors_to(t, **kwargs):
    return map_tensors(t, Tensor.to, **kwargs)        


def stack_tables(tables, dim=0):
    t = transpose_structs(tables)
    return Table(dict(t._map(torch.stack, dim=dim))) 

def cat_tables(tables, dim=0):
    t = transpose_structs(tables)
    return Table(dict(t._map(torch.cat, dim=dim))) 


def split_table(table, splits):
    split = {k: v.split(splits) for k, v in table.items()}

    def build_table(i):
        return Table({k : v[i] for k, v in split.items()})

    return [build_table(i) for i in range(len(splits))]
