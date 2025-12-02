# Based on template from https://github.com/scout719/adventOfCode/
# -*- coding: utf-8 -*
# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=wildcard-import
# pylint: disable=wrong-import-position
# pylint: disable=consider-using-enumerate-

from io import DEFAULT_BUFFER_SIZE
from threading import current_thread
from timeit import default_timer as timer
from collections import deque
from functools import reduce
import functools
import math
import os
import sys
import time
import copy
import re
import itertools
from typing import ChainMap, DefaultDict
import numpy as np
from functools import lru_cache, cache
import operator
from itertools import takewhile
from itertools import permutations 
import itertools, collections
from turtle import Turtle, Screen, heading, left
from math import exp, pi, remainder, sqrt
from collections import namedtuple
from collections import Counter
from collections import defaultdict
#from numpy.lib.arraypad import pad
#from termcolor import colored
#import termcolor
import random
from parse import parse
from parse import search
from parse import findall
from aocd import get_data
from aocd import submit
from shapely import *
from itertools import combinations
import networkx as nx
import graphviz
from queue import PriorityQueue
import regex as re

# UPDATE THIS VARIABLE
AOC_EDITION_YEAR = 2025

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
print(FILE_DIR)
sys.path.insert(0, FILE_DIR + "/")
sys.path.insert(0, FILE_DIR + "/../")
sys.path.insert(0, FILE_DIR + "/../../")

DEBUG_MODE = False

from functools import cmp_to_key
from common.mathUtils import *
from common.utils import *# read_input, main, clear, AssertExpectedResult, ints, printGridsASCII  # NOQA: E402
from common.mapUtils import printMap, buildMapGrid, buildGraphFromMap, buildGraphFromMap_v2, find_starting_point, build_empty_grid, buildGraphFromMap_v3
from common.graphUtils import dijsktra, printGraph, find_all_paths, find_path, find_shortest_path, find_shortest_pathOptimal, bfs, dfs, Graph, hashable_lru, BFS_SP, hash_list, hashable_cache, Graph3, dijkstra_shortest_path, find_cliques, bron_kerbosch
from common.aocVM import *
from lark import Lark, Transformer, v_args, UnexpectedCharacters, UnexpectedToken
from pyformlang.cfg import Production, Variable, Terminal, CFG, Epsilon
from itertools import islice

# pylint: enable=import-error
# pylint: enable=wrong-import-position


################# Advent of Code - 2025 Edition #################

#region ##### Day 1 #####

#Day 1, part 1: 55029 (0.036 secs) 
#Day 1, part 2: 55686 (0.008 secs) 
def day1_1(data):    
    data = read_input(2025, "01") 
    result = 0     
    dial = 50     
    for line in data:
        dir = line[0]
        turns = int(line[1:])
        if dir == 'L':
            dial = (dial - turns) % 100
        elif dir == 'R': 
            dial = (dial + turns) % 100

        if dial == 0:
            result += 1

    AssertExpectedResult(962, result)
    return result

def day1_2(data):    
    data = read_input(2025, "01") 
    result = 0     
    dial = 50     
    for line in data:
        dir = line[0]
        turns = int(line[1:])  

        if dir == 'L':
           sign = 1           
        elif dir == 'R':
            sign = -1

        for _ in range(turns):
            dial += (1 * sign)
            if dial < 0:
                dial += 100
            else:
                dial %= 100
            if dial == 0:
                result += 1
            
    AssertExpectedResult(5782, result)
    return result


#endregion


def day2_1(data):    
    data = read_input(2025, "02") 
    result = 0     
    line = data[0].split(",")
    for l in line:
        [low, high] = ints(l.split("-"))
        for n in range(low, high+1):
            num = str(n)
            if len(num) % 2 == 0 and num[:len(num) // 2] == num[len(num) // 2:]:
                result += n

    AssertExpectedResult(52316131093, result)
    return result

def day2_2(data):    
    data = read_input(2025, "02_teste") 
    result = 0     
    line = data[0].split(",")
    for l in line:
        [low, high] = ints(l.split("-"))
        print("checking [",low,high,"]")
        for n in range(low, high+1):
            num = str(n)
            if len(num) % 2 == 0 and num[:len(num) // 2] == num[len(num) // 2:]:
                result += n
            
            end_p = len(num) // 2
            
            for i in range(0,len(num)):                
                
                check_num = (num[i:])
                p = num.find(check_num)
                print("checking",check_num,"in",num,"found:",p)
                
                if check_num != '' and p != -1:
                    result += int(num)


    AssertExpectedResult(52316131093, result)
    return result


if __name__ == "__main__":
    # override timeout
    main(sys.argv, globals(), AOC_EDITION_YEAR, 28800)

