from typing import List
from collections import Counter
from collections.abc import Mapping, Sequence
import numpy as np

from numbers import Number
import math

import operator
import itertools

import pprint

from functools import reduce


def to_dicts(s):
    if isinstance(s, Struct):
        return {k:to_dicts(v) for k, v in s.__dict__.items()}
    if isinstance(s, dict):
        return {k:to_dicts(v) for k, v in s.items()}
    if isinstance(s, list):
        return [to_dicts(v) for v in s]        
    if isinstance(s, tuple):
        return tuple(to_dicts(v) for v in s)
    else:
        return s

def to_structs(d):
    if isinstance(d, dict):
        return Struct( {k : to_structs(v) for k, v in d.items()} )
    if isinstance(d, list):
        return [to_structs(v) for v in d]
    if isinstance(d, tuple):
        return tuple(to_structs(v) for v in d)
    else:
        return d
 


class Struct(Mapping):
    def __init__(self, entries):
        assert type(entries) == dict
        self.__dict__.update(entries)

    @staticmethod 
    def build(**d):
        return Struct(d)

    def __reduce__(self):
        return (self.__class__, (self.__dict__,))

    def __setitem__(self, index, value):
        self.__dict__[index] = value

    def __getitem__(self, index):
        return self.__dict__[index]

    def __iter__(self):
        return self.__dict__.__iter__()

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __eq__(self, other):
        if isinstance(other, Struct):
            return self.__dict__ == other.__dict__
        else:
            return False

    def _to_dicts(self):
        return to_dicts(self)

    def _subset(self, *keys):
        d = {k:self[k] for k in keys}
        return self.__class__(d)

    def _without(self, *keys):
        d = {k:v for k, v in self.items() if not (k in keys)}
        return self.__class__(d)

    def _filter_none(self):
        return self.__class__({k: v for k, v in self.items() if v is not None})

    def _filterMapWithKey(self, f, *args, **kwargs):
        return self.__class__({k: result for k, v in self.items() 
            for result in [f(k, v)] 
                if result is not None
            })

    def _filterMap(self, f, *args, **kwargs):
        return self.__class__({k: result for k, v in self.items() 
            for result in [f(v, *args, **kwargs)] 
                if result is not None
            })

    def _map(self, f, *args, **kwargs):
        return self.__class__({k: f(v, *args, **kwargs) for k, v in self.items()})

    def _mapWithKey(self, f):
        m = {k: f(k, v) for k, v in self.__dict__.items()}
        return self.__class__(m)

    def __repr__(self):
        commaSep = ", ".join(["{}={}".format(str(k), repr(v)) for k, v in self.items()])
        return "{" + commaSep + "}"

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self.__dict__.__len__()

    def __floordiv__(self, other):
        if isinstance(other, Number):
            return self._map(operator.floordiv, other)
        else:
            return self._zipWith(operator.floordiv, other)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return self._map(operator.truediv, other)
        else:
            return self._zipWith(operator.truediv, other)

    def __add__(self, other):
        if isinstance(other, Number):
            return self._map(operator.add, other)
        else:
            return self._zipWith(operator.add, other)

    def __mul__(self, other):
        if isinstance(other, Number):
            return self._map(operator.mul, other)
        else:
            return self._zipWith(operator.mul, other)


    def _zipWith(self, f, other):
        assert isinstance(other, Struct)
        assert self.keys() == other.keys(), str(self.keys()) + " vs. " + str(other.keys())

        r = {k:f(self[k], other[k]) for k in self.keys()}
        return self.__class__(r)


    def _merge(self, other):
        """
        returns a struct which is a merge of this struct and another.
        """

        assert isinstance(other, Struct)
        d = self.__dict__.copy()
        d.update(other.__dict__)

        return self.__class__(d)

    def _extend(self, **values):
        d = self.__dict__.copy()
        d.update(values)

        return self.__class__(d)


    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)





class ZipList():
    def __init__(self, elems):
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
            return self._map(operator.floordiv, other)
        else:
            return self._zipWith(operator.floordiv, other)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return self._map(operator.truediv, other)
        else:
            return self._zipWith(operator.truediv, other)


    def _map(self, f, *args, **kwargs):
        return ZipList([f(v, *args, **kwargs) for v in self.elems])

    def _zipWith(self, f, other):

        assert isinstance(other, ZipList)
        assert len(self) == len(other)

        r = [f(x, y) for x, y in zip(self.elems, other.elems)]
        return ZipList(r)


    def __add__(self, other):
        if isinstance(other, Number):
            return self._map(operator.add, other)
        else:
            return self._zipWith(operator.add, other)

    def __mul__(self, other):
        if isinstance(other, Number):
            return self._map(operator.mul, other)
        else:
            return self._zipWith(operator.mul, other)        


    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def append(x):
        self.elems.append(x)
        return self

def struct(**d):
    return Struct(d)

def flatten(x, prefix = ''):
    def add_prefix(k):
        if prefix == '':
            return str(k)
        else:
            return prefix + "." + str(k)

    def flatten_iter(iter):
        return [flatten(inner, add_prefix(i)) for i, inner in iter]

    if type(x) == list:
        return flatten_iter(enumerate(x))
    elif type(x) == tuple:
        return flatten_iter(enumerate(x))
    elif isinstance(x, Mapping):
        return flatten_iter(x.items())
    else:
        return [(prefix, x)]


def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return dict(zip(reversed(args), reversed(defaults)))


def replace(d, key, value):
    return {**d, key:value}

def over_struct(key, f):
    def modify(d):
        value = f(d[key])
        return Struct(replace(d, key, value))
    return modify

def over(key, f):
    def modify(d):
        value = f(d[key])
        return replace(d, key, value)
    return modify

def transpose_partial(dicts):
    accum = {}
    for d in dicts:
        for k, v in d.items():
            if k in accum:
                accum[k].append(v)
            else:
                accum[k] = [v]
    return accum

def transpose_partial_structs(structs):
    return Struct(transpose_partial(d.__dict__ for d in structs))


def transpose_structs(structs):
    elem = structs[0]
    d =  {key: [d[key] for d in structs] for key in elem.keys()}
    return Struct(d) 



def transpose_lists(lists):
    return list(zip(*lists))


def drop_while(f, xs):
    while(len(xs) > 0 and f(xs[0])):
        _, *xs = xs    

    return xs


def filter_none(xs):
    return [x for x in xs if x is not None]

def filter_map(f, xs):
    return filter_none(map(f, xs))

def pluck(k, xs, default=None):
    return [d.get(k, default) for d in xs]

def pluck_struct(k, xs, default=None):
    return xs._map(lambda x: x.get(k, default))


def const(x):
    def f(*y):
        return x
    return f

def concat_lists(xs):
    return list(itertools.chain.from_iterable(xs))


def map_dict(f, d):
    return {k :  f(v) for k, v in d.items()}

def pprint_struct(s, indent=2, width=160):
    pp = pprint.PrettyPrinter(indent=indent, width=width)
    pp.pprint(s._to_dicts())

def pformat_struct(s, indent=2, width=160):
    pp = pprint.PrettyPrinter(indent=indent, width=width)
    return pp.pformat(s._to_dicts())

def sum_list(xs):
    assert len(xs) > 0
    return reduce(operator.add, xs)


def append_dict(d, k, v):
    xs = d.get(k) or []
    xs.append(v)

    d[k] = xs
    return d


def transpose_dicts(d):
    r = {}
    for k, v in d.items():
        for j, u in v.items():
            inner = r.get(j) or {}
            inner[k] = u
            r[j] = inner
    return r    

def transpose_dict_lists(d):
    n = max([len(v) for v in d.values()])
    r = [{}] * n
    
    for k, v in d.items():
        for j, u in enumerate(v):
            r[j][k] = u
    return r    


def transpose_list_dicts(xs):
    r = {}
    for d in xs:
        for k, v in d.items():
            inner = r.get(k) or []
            inner.append(v)
            r[k] = inner
    return r    


def add_dict(d, k):
    d[k] = d[k] + 1 if k in d else 1
    return d


def count_dict(xs):
    counts = {}
    for k in xs:
        add_dict(counts, k)

    return counts

def sum_dicts(ds):
    r = {}

    for d in ds:
        for k, v in d.items():
            r[k] = r.get(k, 0) + v

    return r


def partition_by(xs, f):
    partitions = {}

    for x in xs:
        k, v = f(x)
        append_dict(partitions, k, v)    

    return partitions


def map_type(data, data_type, f, *args, **kwargs):
    def rec(x):
        
        if isinstance(x, data_type):
            return f(x, *args, **kwargs)
        elif isinstance(x, str):
            return x
        elif isinstance(x, Sequence):
            return x.__class__(map(rec, x))
        elif isinstance(x, Mapping):
            return x.__class__({k : rec(v) for k, v in x.items()})
        else:
            return x
            
    return rec(data)

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
