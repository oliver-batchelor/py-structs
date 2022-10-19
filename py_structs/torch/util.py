""" Torch related utilities """

import numpy as np
import torch

from py_structs.container import map_type
from py_structs.numpy.util import map_array_like


def map_tensors(data, f, *args, **kwargs):
  return map_type(data, torch.Tensor, f, *args, **kwargs)


def shape_info(x):

  def get_info(x):
    flags = [x.device]
    if isinstance(x, torch.Tensor):
      if x.is_contiguous(memory_format=torch.channels_last):
        flags += ['channels_last']
      if x.is_contiguous(memory_format=torch.contiguous_format):
        flags += ['contiguous']
      if x.is_contiguous(memory_format=torch.channels_last_3d):
        flags += ['channels_last_3d']

    return tuple([x.__class__.__name__, tuple(x.shape), x.dtype, *flags])

  return map_array_like(x, get_info)


def shape(x):
  return map_array_like(x, lambda x: tuple(x.shape))


def from_numpy(data):
  return map_type(data, np.ndarray, torch.Tensor.from_numpy)


def to_numpy(data):
  return map_type(data, torch.Tensor, torch.Tensor.numpy)


def tensors_to(t, **kwargs):
  return map_tensors(t, torch.Tensor.to, **kwargs)
