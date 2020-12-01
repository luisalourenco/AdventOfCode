# Based on template from https://github.com/scout719/adventOfCode/
# -*- coding: utf-8 -*-
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
from common.utils import read_input, main, clear  # NOQA: E402

def day1_1(data):
    sum = 2020
    data = sorted(data, key=int)

    for i in range(0, len(data)):
        elem1 = int(data[i])
        for j in range(i, len(data)):
            elem2 = int(data[j])
            if (elem1 + elem2 == sum):
                 return elem1 * elem2

def day1_2(data):
    sum = 2020
    data = sorted(data, key=int)

    for i in range(0, len(data)):
        elem1 = int(data[i])
        for j in range(i, len(data)):
            elem2 = int(data[j])
            for k in range(j, len(data)):
                elem3 = int(data[k])
                if (elem1 + elem2 + elem3 == sum):
                    result = elem1 * elem2 * elem3
    return result

if __name__ == "__main__":
    main(sys.argv, globals(), 2020)

