from numbers import Integral
from os.path import commonprefix
from typing import List, Tuple

import numpy as np

from py_structs.container import transpose_structs
from py_structs.structs import Struct

from collections.abc import Mapping


class Table(Struct):
  """ Table type representing an array of Struct,
  represented as a Struct of arrays with the same prefix shape.
  """

  def __init__(self, d: Mapping):

    assert len(d) > 0, 'empty Table'
    for (k, v) in d.items():
      assert hasattr(v, 'shape'), f'{k}: expected ndarray, got {type(v).__name__}'

    shapes = [t.shape for t in d.values()]
    self._prefix = commonprefix(shapes)

    super().__init__(d)

  @staticmethod
  def from_structs(structs: List[Struct]) -> 'Table':
    struct_lists = transpose_structs(structs)
    return Table(struct_lists.map_(np.stack)._entries) # pylint: disable=protected-access

  @staticmethod
  def singleton(struct: Struct) -> 'Table':
    return Table(struct.map_(np.expand_dims, axis=0))

  @property
  def _shapes(self) -> Struct:
    return self.map_(lambda x: x.shape)

  @property
  def _shape(self) -> Tuple:
    return self._prefix

  @staticmethod
  def stack(structs: List[Struct]) -> 'Table':
    return Table.from_structs(structs)

  @staticmethod
  def build(d: dict):
    is_array = [isinstance(t, np.ndarray) for t in d.values()]
    if all(is_array):
      return Table(d)
    else:
      return Struct(d)

  @staticmethod
  def create(**d):
    return Table(d)

  def _index_select(self, index: np.ndarray, axis=0) -> 'Table':
    assert axis < len(self._prefix)

    if isinstance(index, np.ndarray):
      assert issubclass(index.dtype.type, Integral)
      assert index.dim() == 1
    else:
      assert issubclass(type(index), Integral),\
          'Table.index_select: unsupported index type' + type(index).__name__

    return self.map_(lambda t: np.take(t, index, axis=axis))

  def _narrow(self, start, n, axis=0):
    return self._index_select(np.arange(start, start + n), axis=axis)

  def _take(self, n, axis=0):
    return self._narrow(0, min(self._prefix[axis], n), axis=axis)

  def _drop(self, n, axis=0):
    n = min(self._prefix[axis], n)
    return self._narrow(n, self._prefix[axis] - n, axis=axis)

  def _sequence(self, axis=0):
    return (
        self._index_select(i, axis=axis) for i in range(0, self._prefix[axis]))

  def _sort_on(self, key, descending=False, axis=0):
    assert key in self
    assert self[key].dim() == 1

    values, inds = self[key].sort(descending=descending, axis=axis)
    return Table({
        k: values if k == key else np.take(v, inds, axis=axis)
        for k, v in self.items()
    })

  @property
  def _head(self):
    return next(iter(self.__dict__.values()))

  @property
  def _size(self):
    return self._head.shape[0]


def table(**d):
  return Table(d)


def stack_tables(tables, dim=0):
  t = transpose_structs(tables)
  return Table(dict(t.map_(np.stack, axis=dim)))


def cat_tables(tables, dim=0):
  t = transpose_structs(tables)
  return Table(dict(t.map_(np.concatenate, axis=dim)))
