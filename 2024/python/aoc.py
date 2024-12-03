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
import itertools, collections
from turtle import Turtle, Screen, heading, left
from math import exp, pi, remainder, sqrt
from collections import namedtuple
from collections import Counter
from collections import defaultdict
from numpy.lib.arraypad import pad
from termcolor import colored
import termcolor
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

# UPDATE THIS VARIABLE
AOC_EDITION_YEAR = 2024

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
print(FILE_DIR)
sys.path.insert(0, FILE_DIR + "/")
sys.path.insert(0, FILE_DIR + "/../")
sys.path.insert(0, FILE_DIR + "/../../")

DEBUG_MODE = False

from common.mathUtils import *
from common.utils import *# read_input, main, clear, AssertExpectedResult, ints, printGridsASCII  # NOQA: E402
from common.mapUtils import printMap, buildMapGrid, buildGraphFromMap, buildGraphFromMap_v2, find_starting_point, build_empty_grid
from common.graphUtils import dijsktra, printGraph, find_all_paths, find_path, find_shortest_path, find_shortest_pathOptimal, bfs, dfs, Graph, hashable_lru, BFS_SP, hash_list, hashable_cache, Graph3
from common.aocVM import *
from lark import Lark, Transformer, v_args, UnexpectedCharacters, UnexpectedToken
from pyformlang.cfg import Production, Variable, Terminal, CFG, Epsilon
from itertools import islice

# pylint: enable=import-error
# pylint: enable=wrong-import-position


################# Advent of Code - 2024 Edition #################

#region ##### Day 1 #####

#Day 1, part 1: 55029 (0.036 secs) 
#Day 1, part 2: 55686 (0.008 secs) 
def day1_1(data):    
    data = read_input(2024, "01")    
    result = 0
    
    list1 = []
    list2 = []
    for line in data:
        vals = parse("{} {}", line)
        list1.append(int(vals[0]))
        list2.append(int(vals[1]))
    
    list1.sort()
    list2.sort()

    for n,m in zip(list1, list2):
        result += abs(n-m)
        
    AssertExpectedResult(0, result)
    return result


def day1_2(data):
    data = read_input(2024, "01")    
    result = 0
    
    list1 = []
    list2 = []
    for line in data:
        vals = parse("{} {}", line)
        list1.append(int(vals[0]))
        list2.append(int(vals[1]))
    
    similarity = Counter(list2)

    for s in list1:
        result += s*similarity[s]

    AssertExpectedResult(0, result)
    return result

#endregion

#region ##### Day 2 #####

def is_safe_record(levels):    
    linear = '-'
    for l1, l2 in itertools.pairwise(levels):
        diff = l2-l1
        if 1 <= abs(diff) <= 3:
            if diff > 0:
                if linear == '-':
                    linear = 'inc'
                else:
                    if linear != 'inc':
                        return False
            elif diff < 0:
                if linear == '-':
                    linear = 'dec'
                else:
                    if linear != 'dec':
                        return False
        else:
            return False
    return True




def day2_1(data):    
    data = read_input(2024, "02")    
    result = 0
    
    for line in data:
        levels = ints(line.split(' '))
        if is_safe_record(levels):
            result += 1
            
    AssertExpectedResult(371, result)
    return result


def day2_2(data):
    data = read_input(2024, "02")    
    result = 0

    for line in data:
        levels = ints(line.split(' '))
        
        if is_safe_record(levels):
            result += 1
        else:
            for i in range(len(levels)):
                n_levels = levels[0:i] + levels[i+1:len(levels)]
                if is_safe_record(n_levels):
                    result += 1
                    break    

    AssertExpectedResult(426, result)
    return result


#endregion

#region ##### Day 3 #####
                
def day3_1(data):    
    data = read_input(2024, "03")    
    result = 0
    
    for line in data:
        for r in findall("mul({:d},{:d})", line):
            result += r.fixed[0] * r.fixed[1]
       
    AssertExpectedResult(196826776, result)
    return result

# 106780429
# 122793038 too high
# 166434763 too high
def day3_2(data):
    data = read_input(2024, "03")    
    result = 0

    for line in data:
        
        start = [m.start() for m in re.finditer("don't()", line)]
        end = [m.end() for m in re.finditer("don't()", line)]
        donts = list(zip(start,end))
        init = donts[0][0]        
        
        start = [m.start() for m in re.finditer("do()", line) if m.start() not in [i for i,j in donts] and m.start() > init]
        end = [m.end() for m in re.finditer("do()", line) if m.start() not in [i for i,j in donts] and m.start() > init]
        dos = list(zip(start,end)) + [(0,0)]        
        #print("donts:", donts)
        #print("dos:", dos)                
        do_i, do_j = dos.pop()
        
        sums = []
        sums.append((0, init))
        
        while dos:
            i,j = dos.pop(0)
            
            donts = [(ii,jj) for ii, jj in donts if ii > i]
            #print("i:",i)
            #print("t:",donts)
            if donts:
                ii,jj = donts.pop(0)
                sums.append((i,ii)) 
                dos = [(di,dj) for di, dj in dos if di > ii] 
            else:
                sums.append((i, len(line)-1)) 
                
        #print(sums)  
        for i,j in sums:
            for r in findall("mul({:d},{:d})", line[i:j]):
                result += r.fixed[0] * r.fixed[1]


    AssertExpectedResult(106780429, result)
    return result


#endregion

#region ##### Day 4 #####
#Day 4, part 1: 26443 (0.096 secs)
#Day 4, part 2: 6284877 (1.065 secs)


#endregion

#region ##### Day 5 #####

#Day 5, part 1: 240320250 (0.110 secs)
#Day 5, part 2: 28580589 (0.003 secs)


#endregion

#region ##### Day 6 #####

#Day 6, part 1: 393120 (0.034 secs)
#Day 6, part 2: 36872656 (3.633 secs)

#endregion


#region ##### Day 7 #####

#Day 7, part 1: 251136060 (0.191 secs)

#endregion


#region ##### Day 8 #####

#Day 8, part 1: 16043 (0.042 secs)
#Day 8, part 2: 15726453850399 (0.584 secs)

#endregion

#region ##### Day 9 #####

#Day 9, part 1: 1681758908 (0.037 secs)
#Day 9, part 2: 803 (0.005 secs)

#endregion

#region ##### Day 10 #####
   
#Day 10, part 1: 6979 (0.104 secs)
#Day 10, part 2: 443 (55.916 secs)

#endregion


#region ##### Day 11 #####

#Day 11, part 1: 9724940 (0.135 secs)
#Day 11, part 2: 569052586852 (0.030 secs)

#endregion

#region ##### Day 12 #####

#endregion


#region ##### Day 13 #####

#Day 13, part 1: 34918 (0.119 secs)
#Day 13, part 2: 33054 (6.659 secs)

#endregion

#region ##### Day 14 #####

#endregion


#region ##### Day 15 #####


#endregion

#region ##### Day 16 #####

#endregion

#region ##### Day 17 #####

#Day 17, part 1: 686 (1.775 secs)

#endregion

#region ##### Day 18 #####


#endregion


#region ##### Day 19 #####


#endregion


#region ##### Day 20 #####


#endregion

#region ##### Day 21 #####


#endregion

#region ##### Day 22 #####

#endregion

#region ##### Day 23 #####

#endregion

#region ##### Day 24 #####

#endregion

#region ##### Day 25 #####


#endregion

if __name__ == "__main__":
    # override timeout
    main(sys.argv, globals(), AOC_EDITION_YEAR, 28800)

