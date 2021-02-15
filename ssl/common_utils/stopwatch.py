# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from collections import defaultdict
from datetime import datetime
import numpy as np


def millis_interval(start, end):
    """start and end are datetime instances"""
    diff = end - start
    millis = diff.days * 24 * 60 * 60 * 1000
    millis += diff.seconds * 1000
    millis += diff.microseconds / 1000
    return millis


class Stopwatch:
    def __init__(self):
        self.last_time = datetime.now()
        self.times = defaultdict(list)
        self.keys = []

    def reset(self):
        self.last_time = datetime.now()
        self.times = defaultdict(list)
        self.keys = []

    def time(self, key):
        if key not in self.times:
            self.keys.append(key)
        self.times[key].append(millis_interval(self.last_time, datetime.now()))
        self.last_time = datetime.now()

    def summary(self):
        num_elems = -1
        total = 0
        max_key_len = 0
        for k, v in self.times.items():
            if num_elems == -1:
                num_elems = len(v)

            assert len(v) == num_elems
            total += np.sum(v)
            max_key_len = max(max_key_len, len(k))

        s = "@@@Time"
        for k in self.keys:
            v = self.times[k]
            s += "\t%s: %d MS, %.2f%%\n" % (k.ljust(max_key_len), np.mean(v), 100.0 * np.sum(v) / total)
        s += "@@@total time per iter: %.2f ms\n" % (float(total) / num_elems)
        self.reset()

        return s
