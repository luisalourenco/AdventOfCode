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
import regex as re

# UPDATE THIS VARIABLE
AOC_EDITION_YEAR = 2024

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
print(FILE_DIR)
sys.path.insert(0, FILE_DIR + "/")
sys.path.insert(0, FILE_DIR + "/../")
sys.path.insert(0, FILE_DIR + "/../../")

DEBUG_MODE = False

from functools import cmp_to_key
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

def findXMAS(xmas, line):
    return line.count(xmas)

# taken from https://stackoverflow.com/questions/6313308/get-all-the-diagonals-in-a-matrix-list-of-lists-in-python
def groups(data, func):
    grouping = defaultdict(list)
    for y in range(len(data)):
        for x in range(len(data[y])):
            grouping[func(x, y)].append(data[y][x])
    
    return list(map(grouping.get, sorted(grouping)))

#1217 too low
def day4_1(data):    
    data = read_input(2024, "04")
    result = 0
    
    #verticals = ['']* len(data[0])

    #for line in data:
    #    result += findXMAS("XMAS", line)
    #    result += findXMAS("SAMX", line)
        
    #    for i in range(len(line)):
    #       verticals[i] = verticals[i] + line[i]
    
    #for line in verticals:
    #    result += findXMAS("XMAS", line)
    #    result += findXMAS("SAMX", line)
    
    cols = groups(data, lambda x, y: x)
    rows = groups(data, lambda x, y: y)
    fdiag = groups(data, lambda x, y: x + y)
    bdiag = groups(data, lambda x, y: x - y) 

    all_data = []
    for l in fdiag:
        if len(l) >= 4:
            all_data.append(''.join(l))
    
    for l in bdiag:
        if len(l) >= 4:
            all_data.append(''.join(l))
    
    for l in rows:
        if len(l) >= 4:
            all_data.append(''.join(l))
    
    for l in cols:
        if len(l) >= 4:
            all_data.append(''.join(l))
    
    for line in all_data:
        result += findXMAS("XMAS", line)
        result += findXMAS("SAMX", line)
    
    AssertExpectedResult(2483, result)
    return result

def day4_2(data):    
    data = read_input(2024, "04")
    result = 0
    
    for y in range(1, len(data)-1):
        for x in range(1, len(data[y])-1):
            if data[y][x] == 'A':
                fdiag = data[y-1][x-1] + 'A' + data[y+1][x+1]
                bdiag = data[y+1][x-1] + 'A' + data[y-1][x+1]
                if (fdiag == 'MAS' or fdiag == 'SAM') and (bdiag == 'MAS' or bdiag == 'SAM'):
                    result += 1
    
    AssertExpectedResult(2483, result)
    return result

#endregion

#region ##### Day 5 #####

def is_in_correct_order(pages_to_print, page_orders):
    printed_pages = set()
    
    for page in pages_to_print:
        pages_needed = page_orders[page]
        #print("pages needed to be printed before:", pages_needed)
        for p in pages_needed:
            # precedent page is in the list to be printed but has not been printed yet, order is not ok
            #print("is",p,"in the update list?", p in pages_to_print)
            #print("has",p,"not been printed yet?", not p in printed_pages)
            if p in pages_to_print and not p in printed_pages:
                return False
        printed_pages.add(page)
    return True

def day5_1(data):    
    data = read_input(2024, "05")
    result = 0
    printed_pages = set()
    
    parse_updates = False
    page_orders = defaultdict(set)
    
    for line in data: 
        if line == '':
            parse_updates = True
        else:
            if parse_updates:
                pages_to_print = ints(line.split(","))
                
                if is_in_correct_order(pages_to_print, page_orders):
                    print("correct order:", pages_to_print)
                    result += pages_to_print[int(len(pages_to_print)/2)]
            else:
                order_rule = parse("{}|{}", line)
                page_orders[int(order_rule[1])].add(int(order_rule[0]))

    AssertExpectedResult(6051, result)
    return result


def compare(x, y):
    if y in page_orders[x]:
        return -1
    elif y not in page_orders[x]:
        return 1
    else:
        return 0


def day5_2(data):    
    data = read_input(2024, "05")
    result = 0
    printed_pages = set()
    
    parse_updates = False
    global page_orders 
    page_orders = defaultdict(set)
    
    for line in data: 
        if line == '':
            parse_updates = True
        else:
            if parse_updates:
                pages_to_print = ints(line.split(","))                
                
                sorted_pages = sorted(pages_to_print, key=cmp_to_key(compare), reverse=True)
                
                if not is_in_correct_order(pages_to_print, page_orders):
                    #print("sorted order:", sorted_pages)
                    
                    if is_in_correct_order(sorted_pages, page_orders):
                        #print("correct order:", sorted_pages)
                        result += sorted_pages[int(len(sorted_pages)/2)]
                        
            else:
                order_rule = parse("{}|{}", line)
                page_orders[int(order_rule[1])].add(int(order_rule[0]))

    AssertExpectedResult(5093, result)
    return result


#endregion

#region ##### Day 6 #####


def find_obstacle(guard_position, direction, obstacles_x, obstacles_y):
    x,y = guard_position
    
    print("guard @", guard_position)
    #^
    if direction == (0, 1):
        obstacle = obstacles_x[x]
        obstacle = [o for o in obstacle if o < y]
        
        print("obstacles found in direction UP ^", obstacle)
        if obstacle:
            obstacle = (x, obstacle[-1])
        print("obstacle @", obstacle)
    # >
    elif direction == (1, 0):
        obstacle = obstacles_y[y]
        obstacle = [o for o in obstacle if o > x]
        
        print("obstacles found in direction RIGHT ->", obstacle)
        if obstacle:
            obstacle = (obstacle[0], y)
        print("obstacle @", obstacle)
    # <
    elif direction == (-1, 0):
        obstacle = obstacles_y[y]
        obstacle = [o for o in obstacle if o < x]
        print("obstacles found in direction LEFT <-", obstacle)
        if obstacle:
            obstacle = (obstacle[-1], y)
        print("obstacle @", obstacle)
    # v
    elif direction == (0, -1):
        obstacle = obstacles_x[x]
        obstacle = [o for o in obstacle if o > y]
        print("obstacles found in direction DOWN v", obstacle)
        if obstacle:
            obstacle = (x, obstacle[0])
        print("obstacle @", obstacle)
    
    print()
    return obstacle

def rotate_direction_clockwise(direction):
    #^
    if direction == (0, 1):
        return (1,0)
    # >
    elif direction == (1, 0):
        return (0,-1)
    # <
    elif direction == (-1, 0):
        return (0,1)
    # v
    elif direction == (0, -1):
        return (-1,0)

def get_new_guard_poosition(obstacle, direction):
    x,y = obstacle
    #^
    if direction == (0, 1):
        return (x, y+1)
    # >
    elif direction == (1, 0):
        return (x-1,y)
    # <
    elif direction == (-1, 0):
        return (x+1,y)
    # v
    elif direction == (0, -1):
        return (x,y-1)

# too low 1103
# too low 1956
# 2640
def day6_1(data):    
    data = read_input(2024, "06")    
    result = 0
    maxX = len(data[0])
    maxY = len(data)
    direction = None
    guard_position = None
    obstacles_x = defaultdict(list)
    obstacles_y = defaultdict(list)
    y = maxY - 1
    x = 0
    visited = set()
    
    for y in range(maxY):
        for x in range(maxX):
            l = data[y][x]
            
            if l == '#':
                obstacles_y[y].append(x)
                obstacles_x[x].append(y)
                obstacles_y[y].sort()
                obstacles_x[x].sort()
            if l == '^':
                direction = (0, 1)
                guard_position = (x,y)
                visited.add(guard_position)
            elif l == '>':
                direction = (1, 0)
                guard_position = (x,y)
                visited.add(guard_position)
            elif l == '<':
                direction = (-1, 0)
                guard_position = (x,y)
                visited.add(guard_position)
            elif l == 'v':
                direction = (0, -1)
                guard_position = (x,y)
                visited.add(guard_position)
    
    #print(obstacles_x)
    #print(obstacles_y)
        
    obstacle = find_obstacle(guard_position, direction, obstacles_x, obstacles_y)

    #while obstacle != []:
    for i in range(40000):
        #print("#:",obstacle)
        x,y = guard_position
        
        if obstacle == []:
            #up
            if direction == (0, 1):
                y_pos = [y for y in range(y)]
                x_pos = [x for _ in range(y)]
            # >
            elif direction == (1, 0):
                x_pos = [x for x in range(x+1, maxX)]
                y_pos = [y for _ in range(x+1, maxX)]
            # <
            elif direction == (-1, 0):
                x_pos = [x for x in range(x)]
                y_pos = [y for _ in range(x)]
            # v
            elif direction == (0, -1):
                y_pos = [y for y in range(y+1, maxY)]
                x_pos = [x for _ in range(y+1, maxY)]
            
            for xx in x_pos:
                for yy in y_pos:
                    print("visiting", (xx,yy))
                    visited.add((xx,yy))
            break
        
        
        guard_position = get_new_guard_poosition(obstacle, direction)
        nx,ny = guard_position
        
        if x < nx:
            x_pos = [x for x in range(x+1, nx+1)]
        else:
            x_pos = [x for x in range(nx+1, x+1)]
        
        if y < ny:
            y_pos = [y for y in range(y+1, ny+1)]
        else:
            y_pos = [y for y in range(ny+1, y+1)]
            
        if nx == x:
            x_pos = [x]
        if ny == y:
            y_pos = [y]
            
        #print("x_pos",x_pos, x, nx)
        for x in x_pos:
            for y in y_pos:
                print("visiting", (x,y))
                visited.add((x,y))
        
        direction = rotate_direction_clockwise(direction)
        
        print("visiting", (nx,ny))
        visited.add((nx,ny))
        print("guard now @",guard_position)
        print("dir:",direction)
        obstacle = find_obstacle(guard_position, direction, obstacles_x, obstacles_y)
    
    result = len(visited)
    #print(visited)
    AssertExpectedResult(196826776, result)
    return result

def day6_2(data):    
    data = read_input(2024, "06_teste")    
    result = 0
    
    for line in data:
        True
       
    AssertExpectedResult(196826776, result)
    return result

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

