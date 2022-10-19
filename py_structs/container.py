""" Struct type and functions for manipulating them """

import dataclasses
import itertools
import operator
import pprint
from functools import partial, reduce
from .structs import Struct
from typing import Any, Callable, Dict, Mapping, Sequence, Iterable


def to_dicts(s):
  if isinstance(s, str):
    return s
  if isinstance(s, Mapping):
    return {k: to_dicts(v) for k, v in s.items()}
  if isinstance(s, Sequence):
    return s.__class__([to_dicts(v) for v in s])
  else:
    return s


def to_structs(d, ignore=()):

  for t in ignore:
    if isinstance(d, t):
      return d

  rec = partial(to_structs, ignore=ignore)

  if isinstance(d, str):
    return d
  if dataclasses.is_dataclass(d):
    return rec(dataclasses.asdict(d))
  if isinstance(d, dict):
    return Struct({k: rec(v) for k, v in d.items()})
  if isinstance(d, Sequence):
    return d.__class__([rec(v) for v in d])
  else:
    return d


def _flatten(x: Mapping, sep='_', prefix='') -> dict:

  def add_prefix(k):
    if prefix == '':
      return str(k)
    else:
      return prefix + sep + str(k)

  def flatten_iter(it):
    return {
        k: v for i, inner in it
        for k, v in _flatten(inner, sep, add_prefix(i)).items()
    }

  if isinstance(x, Sequence):
    return flatten_iter(enumerate(x))
  elif isinstance(x, Mapping):
    return flatten_iter(x.items())
  else:
    return {prefix: x}


def flatten(x, sep='_', dict_type=None):
  """ Flatten a dict of dicts or lists,
  into a single dict with keys concatenated """

  dict_type = dict_type or x.__class__
  return dict_type(_flatten(x, sep))


def replace(d: Mapping, key: Any, value: Any):
  dict_type = d.__class__
  return dict_type({**d, key: value})


def over_struct(key, f):

  def modify(d:Struct):
    value = f(d[key])
    return d.extend_(**{key: value})

  return modify


def over(key, f):

  def modify(d):
    value = f(d[key])
    return replace(d, key, value)

  return modify


def transpose_partial(dicts:Sequence[Mapping]) -> Dict:
  accum = {}
  for d in dicts:
    for k, v in d.items():
      if k in accum:
        accum[k].append(v)
      else:
        accum[k] = [v]
  return accum


def transpose_partial_structs(structs:Sequence[Struct]) -> Struct:
  return Struct(transpose_partial(d.mapping() for d in structs))


def transpose_structs(structs):
  elem = structs[0]
  d = {key: [d[key] for d in structs] for key in elem.keys()}
  return Struct(d)


def transpose_lists(lists, list_type=None):
  list_type = list_type or type(lists[0])
  return [list_type(x) for x in zip(*lists)]


def split_list(xs, sizes):
  assert len(xs) == sum(sizes)
  splits = []
  for size in sizes:

    splits.append(xs[:size])
    xs = xs[size:]
  return splits


def split_dict(d):
  return list(d.keys()), list(d.values())


def invert_keys(d):
  inverted_d = {v: k for k, v in d.items()}
  return d.__class__(inverted_d)


def subset(d, keys):
  subset_ = {k: d[k] for k in keys}
  return d.__class__(subset_)


def drop_while(f, xs):
  while (len(xs) > 0 and f(xs[0])):
    _, *xs = xs

  return xs


def filter_none(xs):
  return [x for x in xs if x is not None]


def filter_map_(f, xs):
  return filter_none(map(f, xs))


def find_by(f, xs):
  for x in xs:
    if f(x):
      return x
  return None


def pluck(k, xs, default=None):
  return [d.get(k, default) for d in xs]


def pluck_struct(k, xs, default=None):
  return xs.map_(lambda x: x.get(k, default))


def const(x):

  def f(*_):
    return x

  return f


def concat_lists(xs):
  return list(itertools.chain.from_iterable(xs))


def map_dict(f: Callable, d: Dict, *args, **kwargs):
  return {k: f(v, *args, **kwargs) for k, v in d.items()}


def map_list(f, xs, *args, **kwargs):
  return [f(x, *args, **kwargs) for x in xs]


def map_none(f, x, *args, **kwargs):
  return f(x, *args, **kwargs) if x is not None else None


def sort_dict(d, key=None, reverse=False, dict_type=None):
  dict_type = dict_type or d.__class__

  d = {k: d[k] for k in sorted(d.keys(), key=key, reverse=reverse)}
  return dict_type(d)


def apply_none(f, *args, **kwargs):
  return f(*args, **kwargs) if f is not None else None


def pprint_struct(s, indent=2, width=160):
  pp = pprint.PrettyPrinter(indent=indent, width=width)
  pp.pprint(to_dicts(s))


def pformat_struct(s, indent=2, width=160):
  pp = pprint.PrettyPrinter(indent=indent, width=width)
  return pp.pformat(to_dicts(s))


def sum_list(xs):
  assert len(xs) > 0
  return reduce(operator.add, xs)


def append_dict(d, k, v):
  xs = d.get(k) or []
  xs.append(v)

  d[k] = xs
  return d


def transpose_dicts(d, dict_type=None):
  dict_type = dict_type or d.__class__
  r = {}

  for k, v in d.items():
    for j, u in v.items():
      inner = r.get(j) or {}
      inner[k] = u
      r[j] = inner

  r = {k:dict_type(d) for k, d in r.items()}
  return dict_type(r)


def transpose_dict_lists(d, dict_type=None):
  dict_type = dict_type or d.__class__

  n = max(len(v) for v in d.values())
  r = [{} for _ in range(n)]

  for k, v in d.items():
    for j, u in enumerate(v):
      r[j][k] = u
  
  return [dict_type(d) for d in r]


def transpose_list_dicts(xs, dict_type=None):
  if len(xs) == 0:
    raise ValueError('empty list')

  dict_type = dict_type or xs[0].__class__

  r = {}
  for d in xs:
    for k, v in d.items():
      inner = r.get(k) or []
      inner.append(v)
      r[k] = inner
  return dict_type(r)


def choose(*options):
  for x in options:
    if x is not None:
      return x
  assert False, 'choose: all options were None'


def when(x, value):
  return None if x is None else value


def add_dict(d, k):
  d[k] = d[k] + 1 if k in d else 1
  return d


def count_dict(xs):
  counts = {}
  for k in xs:
    add_dict(counts, k)

  return counts


def sum_dicts(dicts:Iterable[Mapping], dict_type=None):
  assert isinstance(dicts, Iterable)
  assert len(dicts) > 0

  dict_type = dict_type or dict

  merged = {}
  for d in dicts:
    for k, v in d.items():
      if k in merged:
        merged[k] += v
      else:
        merged[k] = v

  return dict_type(merged)


def partition_by(xs, f):
  """ Multi-way partition by key """
  partitions = {}

  for x in xs:
    k, v = f(x)
    append_dict(partitions, k, v)

  return partitions


def partition_list(f, xs):
  """ Partition into two lists depending on boolean condition """
  a, b = [], []
  for x in xs:
    (a if f(x) else b).append(x)
  return a, b


def merge_dicts(dicts, dict_type=None):
  assert isinstance(dicts, Iterable)
  assert len(dicts) > 0

  dict_type = dict_type or dicts[0].__class__

  merged = {}
  for d in dicts:
    merged.update(d)

  return dict_type(merged)


def map_tree(data, f, *args, **kwargs):

  def rec(x):

    r = f(x, *args, **kwargs)
    if r is not None:
      return r
    elif hasattr(x, 'map_'):
      return x.map_(rec)
    elif isinstance(x, Mapping):
      return x.__class__({k: rec(v) for k, v in x.items()})
    elif not isinstance(x, str) and isinstance(x, Sequence):
      return x.__class__(map(rec, x))
    else:
      return x

  return rec(data)


def map_type(data, data_type, f, *args, **kwargs):

  def g(x):
    if isinstance(x, data_type):
      return f(x, *args, **kwargs)

  return map_tree(data, g)


def traverse_type(data, data_type, f, *args, **kwargs):

  def rec(x):

    if isinstance(x, data_type):
      f(x, *args, **kwargs)
    elif isinstance(x, str):
      pass
    elif isinstance(x, Sequence):
      for v in x:
        rec(v)
    elif isinstance(x, Mapping):
      for v in x.values():
        rec(v)

  rec(data)


def reduce_type(data, data_type, f, op, initializer=None):

  def rec(x):
    if isinstance(x, data_type):
      return f(x)
    elif isinstance(x, Sequence):
      x = [rec(v) for v in x]
      return reduce(op, x, initializer)
    elif isinstance(x, Mapping):
      x = [rec(v) for v in x.values()]
      return reduce(op, x, initializer)
    else:
      return x

  return rec(data)


def map_array_like(data, f, *args, **kwargs):

  def g(x):
    if hasattr(x, 'shape'):
      return f(x)
    else:
      return None

  return map_tree(data, g, *args, **kwargs)

