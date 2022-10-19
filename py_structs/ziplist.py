""" Struct type and functions for manipulating them """

import operator
from numbers import Number
from typing import Sequence


class ZipList():
  """ A seqeunce type which adds by adding/dividing/multiplying 
  in an element-wise way """

  def __init__(self, elems:Sequence):
    self.elems = list(elems)

  def __getitem__(self, index):
    return self.elems[index]

  def __iter__(self):
    return self.elems.__iter__()

  def __eq__(self, other):
    if isinstance(other, ZipList):
      return self.elems == other.elems
    else:
      return False

  def __repr__(self):
    return self.elems.__repr__()

  def __str__(self):
    return self.elems.__str__()

  def __len__(self):
    return self.elems.__len__()

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

  def map_(self, f, *args, **kwargs):
    return ZipList([f(v, *args, **kwargs) for v in self.elems])

  def zip_with_(self, f, other):

    assert isinstance(other, ZipList)
    assert len(self) == len(other)

    r = [f(x, y) for x, y in zip(self.elems, other.elems)]
    return ZipList(r)

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

  def __radd__(self, other):
    return self.__add__(other)

  def __rmul__(self, other):
    return self.__mul__(other)

  def append(self, x):
    self.elems.append(x)
    return self
