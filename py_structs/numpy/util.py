from functools import partial

import numpy as np
from py_structs.container import (map_tree, map_type, reduce_type, traverse_type)


def map_arrays(data, f, *args, **kwargs):
  return map_type(data, np.ndarray, partial(f, *args, **kwargs))


def map_array_like(data, f, *args, **kwargs):

  def g(x):
    if hasattr(x, 'shape'):
      return f(x)
    else:
      return None

  return map_tree(data, g, *args, **kwargs)


def traverse_arrays(data, f, *args, **kwargs):
  return traverse_type(data, np.ndarray, partial(f, *args, **kwargs))


def reduce_arrays(data, f, op, initializer=None):
  return reduce_type(data, np.ndarray, f, op, initializer=initializer)


def shape_info(x):
  return map_array_like(x, lambda x: tuple([*x.shape, x.dtype]))


def shape(x):
  return map_array_like(x, lambda x: tuple(x.shape))


def arrays_astype(t, **kwargs):
  return map_arrays(t, np.ndarray.astype, **kwargs)
