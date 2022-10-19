""" Struct type and functions for manipulating them """

from __future__ import annotations

import operator
from collections.abc import Mapping
from numbers import Number
from immutables import Map

import numpy as np


class Indexed():

  def __init__(self, s: struct):
    assert isinstance(s, Mapping)
    self._d = s

  def __getitem__(self, index):
    return self._d.map_(lambda x: x[index])


class Struct(Mapping):
  """ A mapping type for immutable dictionaries which can be indexed by
  the '.' syntax (overloaded __getattr__), and associated utilities for
  mapping over keys and values, zipping, extending etc.
  """

  def __init__(self, entries):
    assert isinstance(entries, Mapping)

    if isinstance(entries, Struct):
      self._entries = entries._entries
    else:
      self._entries = Map(entries)



  @staticmethod
  def create(**d):
    return Struct(d)

  def __reduce__(self):
    return (self.__class__, (self._entries,))

  def __setitem__(self, index, value):
    self._entries[index] = value

  def __getitem__(self, index):
    return self._entries[index]

  @property
  def _index(self) -> np.ndarray:
    return Indexed(self)

  def __iter__(self):
    return self._entries.__iter__()

  def items(self):
    return self._entries.items()

  def keys(self):
    return self._entries.keys()

  def values(self):
    return self._entries.values()

  def __getattr__(self, k):
    if k[-1] == '_':
      return object.__getattribute__(self, k)
    elif k in self._entries:
      return self._entries[k]
    else:
      raise AttributeError(
          f"Struct does not contain '{k}', options are {list(self._entries.keys())}"
      )

  def __setattr__(self, k, v):
    if k[0] == '_':
      return object.__setattr__(self, k, v)
    else:
      self._entries[k] = v

  def __eq__(self, other):
    if isinstance(other, Struct):
      return self._entries == other._entries
    else:
      return False

  def to_dicts_(self):
    from . import container  # pylint: disable=import-outside-toplevel
    return container.to_dicts(self)

  def subset_(self, *keys):
    d = {k: self[k] for k in keys}
    return Struct(d)

  def without_(self, *keys):
    d = {k: v for k, v in self.items() if not (k in keys)}
    return Struct(d)

  def filter_none_(self):
    return Struct({k: v for k, v in self.items() if v is not None})

  def filter_with_key_(self, f, *args, **kwargs):
    return Struct({k: v for k, v in self.items() if f(k, *args, **kwargs)})

  def filter_map_with_key_(self, f, *args, **kwargs):
    return Struct({
        k: result for k, v in self.items()
        for result in [f(k, v, *args, **kwargs)]
        if result is not None
    })

  def filter_map_(self, f, *args, **kwargs):
    return Struct({
        k: result for k, v in self.items()
        for result in [f(v, *args, **kwargs)]
        if result is not None
    })

  def map_(self, f, *args, **kwargs):
    d = {k: f(v, *args, **kwargs) for k, v in self.items()}
    return Struct(d)

  def map_with_key_(self, f):
    m = {k: f(k, v) for k, v in self._entries.items()}
    return Struct(m)

  def __repr__(self):
    comma_sep = ', '.join([f'{k}={repr(v)}' for k, v in self.items()])
    return '{' + comma_sep + '}'

  def __str__(self):
    return self.__repr__()

  def __len__(self):
    return self._entries.__len__()

  def __floordiv__(self, other):
    if isinstance(other, Number):
      return self.map_(operator.floordiv, other)
    else:
      return self.zip_with_(operator.floordiv, other)

  def __truediv__(self, other):
    if isinstance(other, Number):
      return self.map_(operator.truediv, other)
    else:
      return self.zip_with_(operator.truediv, other)

  def __add__(self, other):
    if isinstance(other, Number):
      return self.map_(operator.add, other)
    else:
      return self.zip_with_(operator.add, other)

  def __mul__(self, other):
    if isinstance(other, Number):
      return self.map_(operator.mul, other)
    else:
      return self.zip_with_(operator.mul, other)

  def zip_with_(self, f, other):
    assert isinstance(other, Struct)

    r = {k: f(v, other[k]) for k, v in self.items()}
    return Struct(r)

  def merge_(self, other:Struct):
    """
    returns a struct which is a merge of this struct and another.
    """
    # pylint: disable=protected-access
    assert isinstance(other, Struct)
    d = self._entries.update(other._entries)

    return Struct(d)


  def extend_(self, **values):
    return Struct(self._entries.update(values))

  def update_(self, **values):
    for k in values:
      assert k in self,\
        f'update_: entry not found {k}, options are {list(self.keys())}'

    return self.extend_(**values)

  def __radd__(self, other):
    return self.__add__(other)

  def __rmul__(self, other):
    return self.__mul__(other)


def set_recursive(path, d, value):
  if len(path) == 0:
    return value

  head, *rest = path
  if isinstance(d, Struct):
    update = {head: set_recursive(rest, d[head], value)}
    up = d.update_(**update)
    return up

  assert False


def struct(**d):
  return Struct(d)
