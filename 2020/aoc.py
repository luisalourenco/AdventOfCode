# Based on template from https://github.com/scout719/adventOfCode/
# -*- coding: utf-8 -*-
# pylint: disable=import-error
# pylint: disable=wrong-import-position
import functools
import math
import os
import sys
import time
import copy

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
print(FILE_DIR)
sys.path.insert(0, FILE_DIR + "/")
sys.path.insert(0, FILE_DIR + "/../")
sys.path.insert(0, FILE_DIR + "/../../")
from utils import read_input, main, clear  # NOQA: E402

def day1_1(data):
    return data

def day1_2(data):
    return data

if __name__ == "__main__":
    main(sys.argv, globals(), 2020)

