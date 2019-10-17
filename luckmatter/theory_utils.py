# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random

def haar_measure(n):
    '''Generate an n-by-n Random matrix distributed with Haar measure'''
    z = np.random.randn(n,n)
    q,r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d/np.absolute(d)
    q = np.dot(np.dot(q,np.diag(ph)), q)
    return q

def init_separate_w(output_d, input_d, choices):
    existing_encoding = set()
    existing_encoding.add(tuple([0] * input_d))

    w = np.zeros((output_d, input_d))

    for i in range(output_d):
        while True:
            encoding = tuple( random.sample(choices, 1)[0] for j in range(input_d) )
            if encoding not in existing_encoding:
                break
        for j in range(input_d):
            w[i, j] = encoding[j]
        existing_encoding.add(encoding)

    return w
