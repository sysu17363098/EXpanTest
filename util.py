import numpy as np
from typing import *
import re
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.init as init


def split_list_by_ratio(l: list, portions: tuple):
    results = []
    accumulated = 0
    for portion in portions:
        size = int(portion * len(l))
        results.append(l[accumulated: accumulated + size])
        accumulated += size
    return results

def get_key_getter(key_getter_indicator):
  if type(key_getter_indicator) == str or type(key_getter_indicator) == int:
    row_key = key_getter_indicator
    key_getter = lambda x: x[row_key]
  else:
    key_getter = key_getter_indicator
  return key_getter

def groupby(iter, key_getter_indicator, grouping_handler=None):
  key_getter = get_key_getter(key_getter_indicator)

  r = {}
  for row in iter:
    key = key_getter(row)
    if key in r:
      r[key].append(row)
    else:
      r[key] = [row]

  if grouping_handler:
    for k, v in r.items():
      r[k] = grouping_handler(v)

  return r

def selector(key_getter_indicator):
  key_getter = get_key_getter(key_getter_indicator)

  def select(l):
    return [key_getter(element) for element in l]

  return select

def comma_escape(s: str):
  return re.sub(r'([,\\])',  r'\\\1', s)

def comma_unescape(s: str):
  return re.sub(r'\\([,\\])',  r'\1', s)

def tuple2str(t: Tuple[str, ...]):
  return ",".join([comma_escape(x) for x in t])

def str2tuple(s: str):
  split = re.split(r'(?<=[^\\]),', s)
  return tuple((comma_unescape(x) for x in split))

def unique_by(iter, key_getter_indicator):
  d = {}
  key_getter = get_key_getter(key_getter_indicator)
  for row in iter:
    key = key_getter(row)
    d[key] = row
  return list(d.values())

def count_if(iter, key_getter_indicator):
  key_getter = get_key_getter(key_getter_indicator)
  count = 0
  for row in iter:
    if key_getter(row) == True:
      count += 1
  return count

def list_diff(l1: List, l2: List):
  return list(set(l1) - set(l2))

def print_lines(iter):
  for i, row in enumerate(iter):
    print(i + 1, row)

def grid(gen: callable, **ranges):
  keys = ranges.keys()
  values = ranges.values()
  return [
    gen(**dict(zip(keys, vs)))
  for vs in itertools.product(*values)]

eps = np.finfo(np.float32).eps.item()

class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

    def __str__(self):
        return str(time.clock() - self.start)

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


if __name__ == '__main__':
    l = nn.LSTM(3, 3)
    print(list(l.parameters()))
    l.apply(weight_init)

    print(list(l.parameters()))

    print(init.orthogonal(torch.zeros(3, 3)))

    grid(lambda x, y: print(x, y), x=range(0, 5), y=range(0, 3))