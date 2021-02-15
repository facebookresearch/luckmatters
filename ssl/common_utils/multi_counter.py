import os
import math
from collections import defaultdict, Counter
from datetime import datetime
from tensorboardX import SummaryWriter


class ValueStats:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def feed(self, v):
        self.summation += v
        self.sumSqr += v * v
        
        if v > self.max_value:
            self.max_value = v
            self.max_idx = self.counter
        if v < self.min_value:
            self.min_value = v
            self.min_idx = self.counter

        self.counter += 1

    def mean(self):
        if self.counter == 0:
            print("Counter %s is 0" % self.name)
            assert False
        return self.summation / self.counter

    def summary(self, info=None):
        info = "" if info is None else info
        name = "" if self.name is None else self.name
        if self.counter > 0:
            mean = self.summation / self.counter
            meanSqr = self.sumSqr / self.counter
            std = math.sqrt(meanSqr - mean * mean + 1e-6)

            # try:
            return "%s%s[%4d]: avg: %8.4f (± %8.4f), min: %8.4f[%4d], max: %8.4f[%4d]" % (
                info,
                name,
                self.counter,
                mean,
                std / math.sqrt(self.counter), # std for the mean
                self.min_value,
                self.min_idx,
                self.max_value,
                self.max_idx,
            )
            # except BaseException:
            #     return "%s%s[Err]:" % (info, name)
        else:
            return "%s%s[0]" % (info, name)

    def reset(self):
        self.counter = 0
        self.summation = 0.0
        self.sumSqr = 0.0
        self.max_value = -1e38
        self.min_value = 1e38
        self.max_idx = None
        self.min_idx = None


class MultiCounter:
    def __init__(self, root, verbose=False):
        # TODO: rethink counters
        self.last_time = None
        self.verbose = verbose
        self.counts = Counter()
        self.stats = defaultdict(lambda: ValueStats())
        self.total_count = 0
        self.max_key_len = 0
        if root is not None:
            self.tb_writer = SummaryWriter(os.path.join(root, "stat.tb"))
        else:
            self.tb_writer = None

    def __getitem__(self, key):
        if len(key) > self.max_key_len:
            self.max_key_len = len(key)

        if key in self.counts:
            return self.counts[key]

        return self.stats[key]

    def inc(self, key):
        if self.verbose:
            print("[MultiCounter]: %s" % key)
        self.counts[key] += 1
        self.total_count += 1
        if self.last_time is None:
            self.last_time = datetime.now()

    def reset(self):
        for k in self.stats.keys():
            self.stats[k].reset()

        self.counts = Counter()
        self.total_count = 0
        self.last_time = datetime.now()

    def time_elapsed(self):
        return (datetime.now() - self.last_time).total_seconds()

    def summary(self, global_counter):
        assert self.last_time is not None
        time_elapsed = (datetime.now() - self.last_time).total_seconds()
        s = "[%d] Time spent = %.2f s\n" % (global_counter, time_elapsed)

        for key, count in self.counts.items():
            s += "%s: %d/%d\n" % (key, count, self.total_count)

        for k in sorted(self.stats.keys()):
            v = self.stats[k]
            info = str(global_counter) + ":" + k
            s += v.summary(info=info.ljust(self.max_key_len + 4)) + "\n"

            if self.tb_writer is not None:
                self.tb_writer.add_scalar(k, v.mean(), global_counter)

        return s
