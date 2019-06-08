# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from datetime import datetime
import pickle

def signature():
    return str(datetime.now()).replace(" ", "_").replace(":", "-").replace(".", "_")

def sig():
    return datetime.now().strftime("%m%d%y_%H%M%S_%f")

def print_info():
    print("cuda: " + os.environ.get("CUDA_VISIBLE_DEVICES", ""))

def add_parser_argument(parser):
    parser.add_argument("--save_dir", type=str, default="./")

def set_args(argv, args):
    cmdline = " ".join(argv)
    signature = sig()
    setattr(args, 'signature', signature)
    setattr(args, "cmdline", cmdline)

def save_data(prefix, args, data):
    filename = f"{prefix}-{args.signature}.pickle"
    save_dir = os.path.join(args.save_dir, filename) 
    print(f"Save to {save_dir}")
    pickle.dump(dict(data=data, args=args, save_dir=save_dir), open(save_dir, "wb"))


