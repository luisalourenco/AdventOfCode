# Based on template from https://github.com/scout719/adventOfCode/
# -*- coding: utf-8 -*
# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=wildcard-import
# pylint: disable=wrong-import-position
# pylint: disable=consider-using-enumerate-

import functools
import math
import os
import sys
import time
import copy
import re
import itertools 
import numpy as np
from functools import lru_cache
import operator
from itertools import takewhile
from turtle import Turtle, Screen
from math import sqrt

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
print(FILE_DIR)
sys.path.insert(0, FILE_DIR + "/")
sys.path.insert(0, FILE_DIR + "/../")
sys.path.insert(0, FILE_DIR + "/../../")

from common.utils import read_input, main, clear, AssertExpectedResult, ints  # NOQA: E402
from common.mapUtils import printMap, buildMapGrid, buildGraphFromMap
from common.graphUtils import printGraph, find_all_paths, find_path, find_shortest_path, find_shortest_pathOptimal, bfs, dfs, Graph, hashable_lru
from common.aocVM import HandheldMachine
from lark import Lark, Transformer, v_args
from pyformlang.cfg import Production, Variable, Terminal, CFG, Epsilon


# pylint: enable=import-error
# pylint: enable=wrong-import-position

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

WHITE_SQUARE = "█"
WHITE_CIRCLE = "•"
BLUE_CIRCLE = f"{bcolors.OKBLUE}{bcolors.BOLD}•{bcolors.ENDC}"
RED_SMALL_SQUARE = f"{bcolors.FAIL}{bcolors.BOLD}■{bcolors.ENDC}"

################# Advent of Code - 2021 Edition #################


#Day 1, part 1: 
#Day 1, part 2: 
def day1_1(data):
    #data = read_input(2021, "11")    
    #data = sorted(data, key=int)
    result = 0
   
    
    #AssertExpectedResult(1006875, result, 1)
    return result

#def day1_2(data):    

if __name__ == "__main__":
    main(sys.argv, globals(), 2020)

