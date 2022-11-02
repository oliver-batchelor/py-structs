import math
from numbers import Number

import numpy as np

from py_structs.structs import struct


class Histogram:
  """ A histogram class implemented as numpy arrays,
  can be added with other histograms which have the same bins """

  def __init__(self,
               values=np.array([0]),
               range=(0, 1),
               num_bins=10,
               trim=True):
    assert len(range) == 2

    self.range = range
    lower, upper = self.range

    bin_indexes = (values - lower) * num_bins / (upper - lower)
    bin_indexes = bin_indexes.astype(np.int32)

    if trim:
      valid = (bin_indexes >= 0) & (bin_indexes < num_bins)

      values = values[valid]
      bin_indexes = bin_indexes[valid]

    bin_indexes.clip(0, num_bins - 1)

    self.sum = values.sum().item()
    self.sum_squares = np.linalg.norm(values).item()
    self.counts = np.bincount(bin_indexes, minlength=num_bins)

  def __repr__(self):
    return self.counts.tolist().__repr__()

  @property
  def normalised(self):
    return self.counts / self.counts.sum()
     

  def bins(self):
    lower, upper = self.range
    d = (upper - lower) / self.counts.shape[0]

    return np.array([lower + i * d for i in range(0, self.counts.shape[0] + 1)])

  def to_json(self):
    return dict(range = self.range, counts=self.counts.tolist())


  def to_struct(self):
    return struct(sum=self.sum,
                  sum_squares=self.sum_squares,
                  counts=self.counts)


  def __add__(self, other):
    assert isinstance(other, Histogram)
    assert other.counts.shape == self.counts.shape\
       and (other.range == self.range), 'mismatched histogram sizes'

    total = Histogram(range=self.range, num_bins=self.counts.shape[0])
    total.sum = self.sum + other.sum
    total.sum_squares = self.sum_squares + other.sum_squares
    total.counts = self.counts + other.counts

    return total

  def __truediv__(self, x):
    assert isinstance(x, Number)

    total = Histogram(range=self.range, num_bins=self.counts.shape[0])
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
