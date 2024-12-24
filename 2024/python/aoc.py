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
from common.mapUtils import printMap, buildMapGrid, buildGraphFromMap, buildGraphFromMap_v2, find_starting_point, build_empty_grid, buildGraphFromMap_v3
from common.graphUtils import dijsktra, printGraph, find_all_paths, find_path, find_shortest_path, find_shortest_pathOptimal, bfs, dfs, Graph, hashable_lru, BFS_SP, hash_list, hashable_cache, Graph3, dijkstra_shortest_path, find_cliques, bron_kerbosch
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
    
    #print("guard @", guard_position)
    #^
    if direction == (0, 1):
        obstacle = obstacles_x[x]
        obstacle = [o for o in obstacle if o < y]
        
        #print("obstacles found in direction UP ^", obstacle)
        if obstacle:
            obstacle = (x, obstacle[-1])
        #print("obstacle @", obstacle)
    # >
    elif direction == (1, 0):
        obstacle = obstacles_y[y]
        obstacle = [o for o in obstacle if o > x]
        
        #print("obstacles found in direction RIGHT ->", obstacle)
        if obstacle:
            obstacle = (obstacle[0], y)
        #print("obstacle @", obstacle)
    # <
    elif direction == (-1, 0):
        obstacle = obstacles_y[y]
        obstacle = [o for o in obstacle if o < x]
        #print("obstacles found in direction LEFT <-", obstacle)
        if obstacle:
            obstacle = (obstacle[-1], y)
        #print("obstacle @", obstacle)
    # v
    elif direction == (0, -1):
        obstacle = obstacles_x[x]
        obstacle = [o for o in obstacle if o > y]
        #print("obstacles found in direction DOWN v", obstacle)
        if obstacle:
            obstacle = (x, obstacle[0])
        #print("obstacle @", obstacle)
    
    #print()
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
    data = read_input(2024, "06_teste")    
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
        
    obstacle = find_obstacle(guard_position, direction, obstacles_x, obstacles_y)

    path = []
    #while obstacle != []:
    for i in range(40000):
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
                    #print("visiting", (xx,yy))
                    path.append((xx,yy))
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
            
        
        for x in x_pos:
            for y in y_pos:
                #print("visiting", (x,y))
                path.append((x,y))
                visited.add((x,y))
        
        direction = rotate_direction_clockwise(direction)
        
        #print("visiting", (nx,ny))
        visited.add((nx,ny))
        #print("guard now @",guard_position)
        #print("dir:",direction)
        obstacle = find_obstacle(guard_position, direction, obstacles_x, obstacles_y)
    
    #print(path)
    result = len(visited)
    AssertExpectedResult(5129, result)
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

def day7_1(data):    
    data = read_input(2024, "07_teste")    
    result = 0
    
    operations = ["+","*"]
    for line in data:
        equation = line.split(":")
        test_value = int(equation[0])
        values = equation[1].strip().split(" ")
        ops = [p for p in itertools.product(operations, repeat=len(values))]
        #print(ops)
        leave_loop = False

        while ops and not leave_loop:
            operators = ops.pop()
            operation = ''
            i = 0
            r = 0
            op = ''

            #print("testing operators", operators)
            for v in values:
                
                if i == 0:
                    operation = v
                    #print("t0:",operation)
                else:

                    op = operators[i-1]
                    operation = operation + op + v

                    #print("t:",operation)
                    r = eval(operation)
                    operation = str(r)
                    #print("partial op:", operation)
                    
                i+=1
            
            if r == test_value:
                #print("TRUE:",operation)
                result += test_value
                leave_loop = True
                break

    AssertExpectedResult(5512534574980, result)
    return result

def day7_2(data):    
    data = read_input(2024, "07_teste")    
    result = 0
    
    operations = ["+","*",'||']
    for line in data:
        equation = line.split(":")
        test_value = int(equation[0])
        values = equation[1].strip().split(" ")
        ops = [p for p in itertools.product(operations, repeat=len(values))]
        #print(ops)
        leave_loop = False

        while ops and not leave_loop:
            operators = ops.pop()
            operation = ''
            i = 0
            r = 0
            op = ''

            #print("testing operators", operators)
            values.reverse()
            for v in values:
                
                if i == 0:
                    operation = v
                    #print("t0:",operation)
                else:

                    op = operators[i-1]
                    if op == '||':
                        #print("concat", operation, v) 
                        operation = operation + v   
                        #print("concat res", operation) 
                    else:
                        operation = operation + op + v

                        #print("t:",operation)
                    r = eval(operation)
                    operation = str(r)
                    #print("partial op:", operation)
                    
                i+=1
            
            if r == test_value:
                #print("TRUE:",operation)
                result += test_value
                leave_loop = True
                break
       
    AssertExpectedResult(196826776, result)
    return result

#endregion


#region ##### Day 8 #####

#404 too high
def day8_1(data):    
    data = read_input(2024, "08")    
    result = 0
    maxX = len(data[0])
    maxY = len(data)
    antenas = defaultdict(list)

    for y in range(maxY):
        for x in range(maxX):
            obj = data[y][x]
            if obj != '.':
                antenas[obj].append((x,y))
    #print(antenas)
    
    antinodes = set()

    for antena in antenas:
        combinations = [p for p in itertools.product(antenas[antena], repeat=2) if p[0] != p[1]]

        visited = set()
        for a1, a2 in combinations:
            if (a1,a2) not in visited:
                combinations.remove((a2,a1))
            visited.add((a1,a2))       

        #print()
        #print("antena", antena,"is in positions",combinations)
        for a1, a2 in combinations:

            tx,ty = a1 
            txx,tyy = a2 
            
            if txx < tx:
                x = txx
                xx = tx
                y = tyy
                yy = ty
            else:
                x = tx 
                xx = txx
                y = ty
                yy = tyy
            
            dx = abs(x-xx)
            dy = abs(y-yy)

            if x == xx:
                na1 = (x-dx, y-dy)
            elif y == yy:
                na1 = (x-dx, y-dy)
            elif y < yy:
                na1 = (x-dx, y-dy)
            else:
                na1 = (x-dx, y+dy)

            nx, ny = na1
            if 0 <= nx < maxX and 0 <= ny < maxY:
                antinodes.add(na1)

            if x == xx:
                na2 = (xx+dx, yy+dy)
            if y == yy:
                na2 = (xx-dx, yy-dy)
            elif y < yy:
                na2 = (xx+dx, yy+dy)
            else:
                na2 = (xx+dx, yy-dy)
            nnx, nny = na2

            if 0 <= nnx < maxX and 0 <= nny < maxY:
                antinodes.add(na2)

            #print((x,y), (xx,yy),"has dx,dy (",dx,dy,") - generates antinodes ->",na1,"and",na2)


    #print("antinodes:",list(antinodes))
    result = len(antinodes)    
       
    AssertExpectedResult(381, result)
    return result

#2498 too high
def day8_2(data):    
    data = read_input(2024, "08_teste")    
    result = 0
    maxX = len(data[0])
    maxY = len(data)
    antenas = []

    for y in range(maxY):
        for x in range(maxX):
            obj = data[y][x]
            if obj != '.':
                antenas.append((x,y))
    #print(antenas)
    
    antinodes = set()

    combinations = [p for p in itertools.product(antenas, repeat=2) if p[0] != p[1]]
    
    visited = set()
    for a1, a2 in combinations:
         if (a1,a2) not in visited:
            combinations.remove((a2,a1))
            visited.add((a1,a2))       

    #print()
    #print("antena", antena,"is in positions",combinations)
    for a1, a2 in combinations:
            antinodes.add(a1)
            antinodes.add(a2)
            tx,ty = a1 
            txx,tyy = a2 
            
            if txx < tx:
                x = txx
                xx = tx
                y = tyy
                yy = ty
            else:
                x = tx 
                xx = txx
                y = ty
                yy = tyy
            
            dx = abs(x-xx)
            dy = abs(y-yy)

            resonating = True
            while resonating:

                old_antinodes = len(antinodes)
                if x == xx:
                    na1 = (x-dx, y-dy)
                elif y == yy:
                    na1 = (x-dx, y-dy)
                elif y < yy:
                    na1 = (x-dx, y-dy)
                else:
                    na1 = (x-dx, y+dy)

                nx, ny = na1
                if 0 <= nx < maxX and 0 <= ny < maxY:
                    antinodes.add(na1)
                if x == xx:
                    na2 = (xx+dx, yy+dy)
                if y == yy:
                    na2 = (xx-dx, yy-dy)
                elif y < yy:
                    na2 = (xx+dx, yy+dy)
                else:
                    na2 = (xx+dx, yy-dy)
                nnx, nny = na2

                if 0 <= nnx < maxX and 0 <= nny < maxY:
                    antinodes.add(na2)

                x = nx 
                y = ny 
                xx = nnx
                yy = nny
                if old_antinodes == len(antinodes):
                    resonating = False

            #print((x,y), (xx,yy),"has dx,dy (",dx,dy,") - generates antinodes ->",na1,"and",na2)


    #print("antinodes:",list(antinodes))
    result = len(antinodes)    
       
    AssertExpectedResult(381, result)
    return result

#endregion

#region ##### Day 9 #####

def find_last_fileblock(filesystem, last_fileblock):
    #last_fileblock = len(filesystem)-1
    while last_fileblock != 0:
        if filesystem[last_fileblock] == '.':
            last_fileblock-=1
        else:
            return last_fileblock
    return last_fileblock

def defrag_filesystem(filesystem, last_fileblock, free_space):
    
    #print("before:", filesystem)
    for i in range(len(filesystem)):        
        if ''.join(filesystem)[:-free_space].count('.') == 0:
            break
        
        if filesystem[i] == '.':
            filesystem[i] = filesystem[last_fileblock]
            #print("switching", filesystem[last_fileblock], "to pos", i)
            filesystem[last_fileblock] = '.'
            #print("freeing space at", last_fileblock)
            last_fileblock = find_last_fileblock(filesystem, last_fileblock)
            #print("last_file_block found at ",last_fileblock)
            #print("interaction:", filesystem)
            
    #print()
    #print("after:", filesystem)
    return filesystem

def day9_1(data):    
    data = read_input(2024, "09_teste")    
    result = 0
    size = 0
    
    input_data = str(data[0])
    filesystem = []
    curr_id = 0
    
    file_blocks = defaultdict()
    free_space = []
    is_free_space = False
    
    last_fileblock = 0
    idx = 0
    for i in range(len(input_data)):
        val = int(input_data[i])
        size += int(val)
        
        if is_free_space:
            for j in range(val):
                filesystem.append('.')
                idx+=1
        else:
            file_blocks[curr_id] = val
            for j in range(val):
                filesystem.append(str(curr_id)) 
                idx +=1
            last_fileblock = idx-1               
            curr_id +=1
        is_free_space = not is_free_space
    
    #filesystem.reverse()
    free_space = sum([1 for b in filesystem if b == '.'])
    
    filesystem = defrag_filesystem(filesystem, last_fileblock, free_space)
    for i in range(len(filesystem)):
        if filesystem[i] == '.':
            break
        result += i*int(filesystem[i])
    
    AssertExpectedResult(6241633730082, result)
    return result

def defrag_filesystemv2(filesystem, last_fileblock, free_space):
    
    free_space.reverse()
    curr_file = filesystem[last_fileblock]
    curr_file_size = 0
    k = last_fileblock
    #print("before:", filesystem)
    for i in range(len(filesystem)):        
        if not free_space:
            break
        
        
        if filesystem[k] == '.' or filesystem[k] != curr_file:
            init, size = free_space.pop()
            print("moving file", curr_file,"of size", curr_file_size,"to position", init)
            
            if curr_file_size <= size:
                for j in range(curr_file_size):
                    filesystem[init+j] = curr_file
                    print("moved file", curr_file,"to position", init+j)
                    print(filesystem)
                    filesystem[last_fileblock] = '.'            
                    last_fileblock -=1
            else:
                print("cannot move file",curr_file,"of size", curr_file_size,"to position", init)
                
            last_fileblock = find_last_fileblock(filesystem, last_fileblock)
            curr_file = filesystem[last_fileblock]
            k = last_fileblock
            curr_file_size = 0
        else:
            curr_file_size += 1
            k -=1
            
            
    #print()
    #print("after:", filesystem)
    return filesystem

def day9_2(data):    
    data = read_input(2024, "09_teste")    
    result = 0
    size = 0
    
    input_data = str(data[0])
    filesystem = []
    curr_id = 0
    
    is_free_space = False
    free_space_list = []
    last_fileblock = 0
    idx = 0
    for i in range(len(input_data)):
        val = int(input_data[i])
        size += int(val)
        
        if is_free_space:
            if val > 0:
                free_space_list.append((idx, val))
            for j in range(val):
                filesystem.append('.')
                idx+=1
        else:
            for j in range(val):
                filesystem.append(str(curr_id)) 
                idx +=1
            last_fileblock = idx-1               
            curr_id +=1
        is_free_space = not is_free_space
    
    free_space = sum([1 for b in filesystem if b == '.'])
    
    print(free_space_list)
    
    filesystem = defrag_filesystemv2(filesystem, last_fileblock, free_space_list)
    
    for i in range(len(filesystem)):
        if filesystem[i] == '.':
            break
        result += i*int(filesystem[i])
    
    AssertExpectedResult(6241633730082, result)
    return result

#endregion

#region ##### Day 10 #####

def is_connected(map, p, n):
    x,y = p
    xx,yy = n
    if int(map[yy][xx]) - int(map[y][x]) == 1:
        return True
    else:
        return False

def day10_1(data):    
    data = read_input(2024, "10")    
    result = 0
    trailheads = []
    ends = []
    
    graph = buildGraphFromMap_v2(data, '.', is_connected)
    #for k in graph:
    #    print("k:", k, "has neighbours", graph[k])
        
    rows = len(data)
    columns = len(data[0])

    for y in range(rows):          
        for x in range(columns):
            if data[y][x] == '0':
                trailheads.append((x,y))
            elif data[y][x] == '9':
                ends.append((x,y))
    
    for trailhead in trailheads:
        score = 0
        for end  in ends:
            p = find_path(graph, trailhead, end)
            if p:
                score+=1
            #2,0 -> 5,4
            #print("path from",trailhead,"to",end,":",p)
        #print("score:", score)
        result+=score
    
    AssertExpectedResult(587, result)
    return result 

def find_all_paths(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if start not in graph:
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths 

def day10_2(data):    
    data = read_input(2024, "10")    
    result = 0
    trailheads = []
    ends = []
    
    graph = buildGraphFromMap_v2(data, '.', is_connected)
    #for k in graph:
    #    print("k:", k, "has neighbours", graph[k])
        
    rows = len(data)
    columns = len(data[0])

    for y in range(rows):          
        for x in range(columns):
            if data[y][x] == '0':
                trailheads.append((x,y))
            elif data[y][x] == '9':
                ends.append((x,y))
    
    for trailhead in trailheads:
        score = 0
        for end  in ends:
            p = find_all_paths(graph, trailhead, end)
            if p:
                score+=len(p)
        result+=score
    
    AssertExpectedResult(587, result)
    return result 

#endregion


#region ##### Day 11 #####

def blink(stones):
    new_stones = []
    
    for stone in stones:
        stone_size = len(stone)
        if int(stone) == 0:
            new_stones.append('1')
        elif stone_size%2 == 0:
            mid = int(stone_size / 2)
            new_stones.append(str(int(stone[:mid])))
            new_stones.append(str(int(stone[mid:])))
        else:
            new_stones.append(str(int(stone)*2024))
    
    
    return new_stones

def day11_1(data):    
    data = read_input(2024, "11")    
    result = 0    
    
    stones = data[0].split(' ')
    print(stones)
    
    blinks = 25
    for _ in range(blinks):
        stones = blink(stones)
        #print(stones)
    result = len(stones)
    AssertExpectedResult(220999, result)
    return result 


def day11_2(data):    
    data = read_input(2024, "11")    
    result = 0    
    
    stones = data[0].split(' ')
    #print(stones)
    
    blinks = 25
    for i in range(blinks):
        stones = blink(stones)
        print("iteration",i,":",stones)
    result = len(stones)
    AssertExpectedResult(587, result)
    return result 

#endregion

#region ##### Day 12 #####

def get_perimeter(p, plots, sizeX, sizeY):
    
    x, y = p
    neighbours = 0
    perimeter = 4        

    east = (x+1, y)
    west = (x-1, y)
    north = (x, y-1)
    south = (x, y+1)
        
    if 0 <= east[0] < sizeX and 0 <= east[1] < sizeY:
        if east in plots:
            neighbours +=1
    
    if 0 <= west[0] < sizeX and 0 <= west[1] < sizeY:                        
        if west in plots:
            neighbours +=1
    
    if 0 <= north[0] < sizeX and 0 <= north[1] < sizeY: 
        if north in plots:
            neighbours +=1
    
    if 0 <= south[0] < sizeX and 0 <= south[1] < sizeY: 
        if south in plots:
            neighbours +=1       
    
    return perimeter - neighbours   

def day12_1(data):    
    data = read_input(2024, "12_teste")    
    result = 0    
    
    garden_plots = defaultdict()
    
    rows = len(data)
    columns = len(data[0])

    for y in range(rows):          
        for x in range(columns):
            plant = data[y][x]
            if plant not in garden_plots:
                garden_plots[plant] = (1, 4, [(x,y)])
            else:
                area, perimeter, plots = garden_plots[plant]
                plots.append((x,y))
                garden_plots[plant] = (area+1, perimeter+4, plots)

    final_garden = defaultdict()
    for garden in garden_plots:
        area, _, plots = garden_plots[garden]
        perimeter = 0
        for p in plots:
            res = get_perimeter(p, plots, columns, rows)
            
            if res == 4:
                #print("-",p)
                r = random.randint(1,100)
                final_garden[garden+str(r)] = (1, 4, plots)
                perimeter = 0
            else:   
                perimeter += res     
                final_garden[garden] = (area, perimeter, plots)
    
    for a,p,_ in final_garden.values():
        result += a*p
    
    
    for garden in final_garden:
        a, p, pp = final_garden[garden]
        print(garden,":", a, p, pp)
    
    
    AssertExpectedResult(587, result)
    return result 

def day12_2(data):    
    data = read_input(2024, "12_teste")    
    result = 0    
    
    AssertExpectedResult(587, result)
    return result 

#endregion


#region ##### Day 13 #####

from z3 import *

def day13_1(data):    
    data = read_input(2024, "13")    
    result = 0    
    
    machines = []
    id = 0
    for line in data:
        res = parse("Button {}: X+{}, Y+{}", line)
        #print(res)
        if not res:
            res = parse("Prize: X={}, Y={}", line)
            if res:
                #print(res[0], res[1])
                machines[id].append((res[0], res[1]))
                id+=1
        else:
            b = res[0]
            ax = res[1]
            ay = res[2]
            #print(b, ax, ay)
            if b == 'A':
                machines.append([(ax,ay)])
            else:
                machines[id].append((ax,ay))
                        
    for ba, bb, p in machines:
        solver = Solver()

        a = Int('A')
        b = Int('B')
        
        solver.add(p[0] == (a*ba[0] + b*bb[0]), p[1] == (a*ba[1] + b*bb[1]))
        
        try:    
            solver.check()
            model = solver.model()
            #print(model)
            button_a = model[a].as_long()
            button_b = model[b].as_long()
            #print("-",button_a)
            result += button_a * 3 + button_b
    
        except:
            True
            #print("No model found!")
    
    
    AssertExpectedResult(587, result)
    return result 

def day13_2(data):    
    data = read_input(2024, "13")    
    result = 0    
    
    machines = []
    id = 0
    for line in data:
        res = parse("Button {}: X+{}, Y+{}", line)
        #print(res)
        if not res:
            res = parse("Prize: X={}, Y={}", line)
            if res:
                machines[id].append(('1'+res[0].zfill(13), '1'+res[1].zfill(13)))
                #print('1'+res[0].zfill(13))
                id+=1
        else:
            b = res[0]
            ax = res[1]
            ay = res[2]
            #print(b, ax, ay)
            if b == 'A':
                machines.append([(ax,ay)])
            else:
                machines[id].append((ax,ay))
                        
    for ba, bb, p in machines:
        solver = Solver()

        a = Int('A')
        b = Int('B')
        
        solver.add(p[0] == (a*ba[0] + b*bb[0]), p[1] == (a*ba[1] + b*bb[1]))
        
        try:    
            solver.check()
            model = solver.model()
            #print(model)
            button_a = model[a].as_long()
            button_b = model[b].as_long()
            #print("-",button_a)
            result += button_a * 3 + button_b
    
        except:
            True
            #print("No model found!")
    
    
    AssertExpectedResult(587, result)
    return result  
#endregion

#region ##### Day 14 #####

def simulate_robots(robots, time, width, tall):
    new_robots = []

    min_safety = sys.maxsize
    for i in range(time):
        new_robots = []
        for pos, vel in robots:
            x,y = pos
            vx, vy = vel 
            new_x = (x+vx)%width
            new_y = (y+vy)%tall
            new_pos = (new_x, new_y)
            new_robots.append( (new_pos, vel) )

        robots = new_robots
        val = compute_safety_factor(robots, width, tall)
        if val < min_safety:
            min_safety = val
            #print(min_safety, "for time", i)
            #print_robots(robots, width, tall)


    return new_robots


def compute_safety_factor(robots, width, tall):
    q1 = 0
    q2 = 0
    q3 = 0
    q4 = 0

    for p, _ in robots:
        x,y = p
        if x < int(width/2) and y < int(tall/2):
            q1 +=1
        if x > int(width/2) and y > int(tall/2):
            q2 +=1
        if x < int(width/2) and y > int(tall/2):
            q3 +=1
        if x > int(width/2) and y < int(tall/2):
            q4 +=1

    #print(q1,q2,q3,q4)
    return q1 * q2 * q3 * q4

# 94154500 too low
# 157090752 too low
def day14_1(data): 
    data = read_input(2024, "14")    
    width = 101
    tall = 103
    time = 100

    robots = []
    for line in data:
        vals = parse("p={},{} v={},{}", line)
        x = int(vals[0])
        y = int(vals[1])
        vx = int(vals[2])
        vy = int(vals[3])
        robots.append(((x,y), (vx,vy)))

    robots = simulate_robots(robots, time, width, tall)
    result = compute_safety_factor(robots, width, tall)


    AssertExpectedResult(587, result)
    return result 

def print_robots(robots, width, tall):
    map = [ [ ('.') for i in range(width) ] for j in range(tall) ]    
    for p, _ in robots:
        x,y = p
        map[y][x] = '#'

    for y in range(tall):
        for x in range(width):
            print(map[y][x], end="")
        print()


#304 too low
# 6667 too low
def day14_2(data): 
    data = read_input(2024, "14")    
    width = 101
    tall = 103

    time = 6668

    robots = []
    print("part2")
    for line in data:
        vals = parse("p={},{} v={},{}", line)
        x = int(vals[0])
        y = int(vals[1])
        vx = int(vals[2])
        vy = int(vals[3])
        robots.append(((x,y), (vx,vy)))

    robots = simulate_robots(robots, time, width, tall)
    
    print_robots(robots, width, tall)

    result = compute_safety_factor(robots, width, tall)


    AssertExpectedResult(587, result)
    return result 

#endregion


#region ##### Day 15 #####

def find_robot_position(map):
    rows = len(map)
    columns = len(map[0])
    print(rows, columns)
    for y in range(rows) :
        for x in range(columns):
            print(map[y][x])
            if map[y][x] == '@':
                return (x,y)

def get_delta(move):
    if move == '^':
        dx = 0
        dy = -1
    elif move == 'v':
        dx = 0
        dy = 1
    elif move == '<':
        dx = -1
        dy = 0
    elif move == '>':
        dx = 1
        dy = 0
    return dx, dy

def execute_moves(map, robot, moves, part2 = False):
    
    #printMap(map, "initial map")
    for move in moves:
        rx, ry = robot
        dx, dy = get_delta(move)

        # free square, robot moves
        if map[ry + dy][rx + dx] == '.':
            robot =  (rx+dx, ry+dy)
            map[ry + dy][rx + dx] = '@'
            map[ry][rx] = '.'
        # found box, try to push it
        elif map[ry + dy][rx + dx] == 'O':
            
            if abs(dy) > 0:
                limit = len(map)-dy
            elif abs(dx) > 0:
                limit = len(map[0])-dx
            # if no wall is found, it can push and move
            for i in range(2, limit):
                if map[ry + i*dy][rx + i*dx] == '#':
                    break

                if map[ry + i*dy][rx + i*dx] == '.':
                    robot =  (rx+dx, ry+dy)
                    map[ry + dy][rx + dx] = '@'
                    map[ry][rx] = '.'
                    map[ry + i*dy][rx + i*dx] = 'O'

                    break
    return map

def get_result(map):
    rows = len(map)
    columns = len(map[0])
    result = 0
    for y in range(rows):          
        for x in range(columns):
            if map[y][x] == 'O':
                result += ((100*y)+x)
    return result
    

def day15_1(data):    
    data = read_input(2024, "15")    
    result = 0    

    moves = data.pop()
    data.pop()
    map = buildMapGrid(data, initValue='.', withPadding = False)
    
    robot = find_robot_position(map)

    map = execute_moves(map, robot, moves)
    
    result = get_result(map)   
    
    AssertExpectedResult(587, result)
    return result 

def buildPart2Map(data):
    data = copy.deepcopy(data)

    rows = len(data)
    columns = len(data[0])

    map = [ [ ('.') for i in range(columns*2) ] for j in range(rows) ]    
    
    i = 0
    for y in range(rows):
        for x in range(columns):
            tile =  data[y][x]
            if tile == 'O':
                map[y][i] = '['
                map[y][i+1] = ']'
            elif tile == '@':
                map[y][i] = '@'
                map[y][i+1] = '.'
            else:
                map[y][i] = tile
                map[y][i+1] = tile
            i+=2
        i = 0
    return map

def execute_movesv2(map, robot, moves):
    
    #printMap(map, "initial map")
    for move in moves:
        rx, ry = robot
        dx, dy = get_delta(move)

        # free square, robot moves
        if map[ry + dy][rx + dx] == '.':
            robot =  (rx+dx, ry+dy)
            map[ry + dy][rx + dx] = '@'
            map[ry][rx] = '.'
        # found box, try to push it
        elif map[ry + dy][rx + dx] in ['[',']']:
            
            if abs(dy) > 0:
                limit = len(map)-dy
            elif abs(dx) > 0:
                limit = len(map[0])-dx
            # if no wall is found, it can push and move
            for i in range(2, limit):
                if map[ry + i*dy][rx + i*dx] == '#':
                    break

                if map[ry + i*dy][rx + i*dx] == '.':
                    robot =  (rx+dx, ry+dy)
                    map[ry + dy][rx + dx] = '@'
                    map[ry][rx] = '.'
                   
                    switch = False
                    for j in range(2,i+1):
                        print("j",j, switch)
                        if switch:
                            map[ry + i*dy][rx + i*dx]= '['    
                        else:
                            map[ry + i*dy][rx + i*dx] = ']'
                        switch = not switch
                        #map[ry + (i+1)*dy][rx + (i+1)*dx] = ']'
                    break
        
        msgToPrint = "robot moved: "+ move
        printMap(map, msg=msgToPrint)
    return map

def day15_2(data):    
    data = read_input(2024, "15_teste")    
    result = 0   


    moves = data.pop()
    data.pop()
    map = buildPart2Map(data)
    robot = find_robot_position(map)
    printMap(map)
    map = execute_movesv2(map, robot, moves)
    
    #result = get_result(map)   

    
    AssertExpectedResult(587, result)
    return result 

#endregion

#region ##### Day 16 #####

def is_connectedv2(map, p, n):
    x,y = p
    xx,yy = n
    if map[yy][xx] in ('.','S','E'):
        return True
    else:
        return False


def compute_score(start, path):
    score = 0
    i,j = start
    y_axis = False
    x_axis = False
    forward = 0
    rotate = 0

    for x,y in path:
        #print(x,y)
        if (x == i and y != j):
            if not y_axis:
                y_axis = True
                x_axis = False
                score += 1001
                rotate +=1
            else:
                score+= 1
                forward+=1
        elif (x != i and y == j):
            if not x_axis:
                x_axis = True
                y_axis = False
                score += 1001
                rotate +=1
            else:
                score+= 1
                forward+=1
        i = x
        j = y
    return score, forward, rotate



# too high 109404
def day16_1(data):    
    data = read_input(2024, "16_teste")    
    result = 0    
    
    graph = buildGraphFromMap_v3(data, '#', is_connectedv2)
    #printGraph(graph)
    start = find_starting_point(data, "S")
    end = find_starting_point(data, "E")
    print(start,end)
    
    #paths = find_all_paths(graph, start, end)
    #p = find_shortest_path(graph, start, end)
    shortest_distance, path = dijkstra_shortest_path(graph, start, end)
    
    print(shortest_distance)
    print(path)
    print(len(path))
    
    
    result, f, r = compute_score(start, path)
    print(f,r)

    #result = compute_score(start, paths)
    #print(paths)
    '''
    min_path = sys.maxsize
    for p in paths:
        result, f, r = compute_score(start, p)
        if result < min_path:
            min_path = result
    result = min_path
    '''
    #printGraph(graph)
    
    AssertExpectedResult(587, result)
    return result 

def day16_2(data):    
    data = read_input(2024, "16_teste")    
    result = 0    
    
    AssertExpectedResult(587, result)
    return result 

#endregion

#region ##### Day 17 #####

def combo_operand(operand, registers):
    if 0 <= operand <= 3:
        return operand
    elif operand == 4:
        return registers[0]
    elif operand == 5:
        return registers[1]
    elif operand == 6:
        return registers[2]

def execute_program(registers, program):
    instruction_pointer = 0
    output = ''
    loops = 0
    while instruction_pointer < len(program):
        opcode = program[instruction_pointer]
        operand = program[instruction_pointer+1]
        #print("op:", opcode,"operand:", operand)
        #print("pc:",instruction_pointer)
                
        # adv  A = int (A / 2**combo(operand))
        if opcode == 0:
            registers[0] = int (registers[0] / (2**combo_operand(operand, registers)))
            instruction_pointer += 2            
        
        # bxl  B = B bitwise XOR lit operand (a ^ b)
        elif opcode == 1:
            registers[1] = registers[1] ^ operand
            instruction_pointer += 2
            
        # bst  B = combo operand modulo 8
        elif opcode == 2:
            registers[1] = combo_operand(operand, registers) % 8
            instruction_pointer += 2
        
        # jnz  if A == 0 : nop else instruction_pointer = A
        elif opcode == 3:

            if registers[0] != 0:
                instruction_pointer = operand
            else:
                instruction_pointer += 2
        
        # bxc  B = B bitwise XOR C
        elif opcode == 4:
            registers[1] = registers[1] ^ registers[2]
            instruction_pointer += 2
            
        # out output combo operand % 8
        elif opcode == 5:
            output += str(combo_operand(operand, registers) % 8) + ','
            instruction_pointer += 2
            loops +=1
            
        # bdv B = int (A / 2**combo(operand))
        elif opcode == 6:
            registers[1] = int (registers[0] / (2**combo_operand(operand, registers)))
            instruction_pointer += 2  
            
        # cdv C = int (A / 2**combo(operand))
        elif opcode == 7:
            registers[2] = int (registers[0] / (2**combo_operand(operand, registers)))
            instruction_pointer += 2  
        #if loops == 16:
            #break
    
    #print("looped", loops)       
    return output
        
    
    
def day17_1(data):    
    data = read_input(2024, "17")    
    result = 0    
    
    registers = []
    registers.append( int(parse("Register A: {}", data[0])[0]) )
    registers.append( int(parse("Register B: {}", data[1])[0]) )
    registers.append( int(parse("Register C: {}", data[2])[0]) )
    program = ints(parse("Program: {}", data[4])[0].split(","))

    result = execute_program(registers, program)
    
    
    AssertExpectedResult(587, result)
    return result 

def day17_2(data):    
    data = read_input(2024, "17")    
    result = 0    
    
    registers = []
    registers.append( int(parse("Register A: {}", data[0])[0]) )
    registers.append( int(parse("Register B: {}", data[1])[0]) )
    registers.append( int(parse("Register C: {}", data[2])[0]) )
    program = ints(parse("Program: {}", data[4])[0].split(","))

    
    exp = ','.join(map(str,program)) +','
    print("expects:", exp)
    
    for _ in range(10000):
            A = random.randint(1000000000000,99999999999999)
            registers[0] = A
            #registers[0] = 123456789000000
            result = execute_program(registers, program)
            #print(len(result), len(exp))
            if result == exp:
                print("FOUND:", A)
            
    
    ''' 
    2,4,
    1,5,
    7,5,
    0,3,
    1,6,
    4,3,
    5,5,
    3,0
    
    bst B = A%8
    bxl B = B ^ 5
    cdv C = A / 2**B
    adv A = A / 8
    bxl B = B ^ 6
    bxc B = B ^ C
    out  output += B%8
    jnz if A != 0 then program_counter = 0 (init) -> while A != 0 repeat all
    
    '''
    
    AssertExpectedResult(587, result)
    return result 

#endregion

#region ##### Day 18 #####


def day18_1(data):    
    data = read_input(2024, "18")    
    result = 0
    bytes_to_fall = []
    
    for line in data:
        v = line.split(',')
        bytes_to_fall.append((int(v[0]), int(v[1])))
    
    fall = 1024
    rows = 71
    columns = 71

    memory_grid = [ [ ('.') for i in range(columns) ] for j in range(rows) ]    
    for i in range(fall):
        x,y = bytes_to_fall[i]
        memory_grid[y][x] = "#"
    
    graph = buildGraphFromMap_v3(memory_grid, '#', is_connectedv2)
    
    start = (0,0)
    end = (70,70)

    shortest_distance, path = dijkstra_shortest_path(graph, start, end)
    result = shortest_distance
      
    AssertExpectedResult(356, result)
    return result

def day18_2(data):    
    data = read_input(2024, "18")    
    result = 0
    bytes_to_fall = []
    
    for line in data:
        v = line.split(',')
        bytes_to_fall.append((int(v[0]), int(v[1])))
    
    fall = 3450
    rows = 71
    columns = 71

    memory_grid = [ [ ('.') for i in range(columns) ] for j in range(rows) ]    
    for i in range(fall):
        x,y = bytes_to_fall[i]
        memory_grid[y][x] = "#"
        if i > 1024:
            graph = buildGraphFromMap_v3(memory_grid, '#', is_connectedv2)
        
            start = (0,0)
            end = (70,70)
        
            shortest_distance, path = dijkstra_shortest_path(graph, start, end)
            if not path:
                result = shortest_distance
                result = x,y
                break
        
    AssertExpectedResult("22,33", result)
    return result

#endregion


#region ##### Day 19 #####

def match_string(string, target, index):
    lens = len(string)
    lent = len(target)
    
    if(index + lens > lent):
        return False
    
    for i in range(index,index+lens):
        if(string[i-index] != target[i]):
            return False
    
    if(target[index+lens] == " " or target[index+lens] == "."):
        return True
    else:
        return False

    
def solve(arr,target,index,dp):
    
    n = len(arr)
    ln = len(target)
    
    if(index == ln):
        return True
    elif(index > ln):
        return False
    
    if(dp[index] != None):
        return dp[index]
    
    result = False
    for em in arr:
        if(match_string(em,target,index)):
            result = result or solve(arr,target,index+len(em)+1,dp)
    dp[index] = result
    return dp[index]
    

def day19_1(data):    
    data = read_input(2024, "19_teste")    
    result = 0    
    
    towels = [towel.strip() for towel in data[0].split(",")]
    towels_size = defaultdict(list)
    max_size = 0
    for towel in towels:
        towels_size[len(towel)].append(towel)
        if len(towel) > max_size:
            max_size = len(towel)
    
    print(towels)
        
    for pattern in data[2:]:
        dp = [None for i in range(len(pattern))]
        answer = solve(towels,pattern,0,dp)
        print(pattern, answer)
    
    AssertExpectedResult(587, result)
    return result 

def day19_2(data):    
    data = read_input(2024, "19_teste")    
    result = 0    
    
    AssertExpectedResult(587, result)
    return result 

#endregion


#region ##### Day 20 #####

def find_walls(map):
    rows = len(map)
    columns = len(map[0])
    walls = []
    for y in range(rows):          
        for x in range(columns):
            if map[y][x] == '#' and y != 0 and y != rows-1 and x != 0 and x != columns-1:
                walls.append((x,y))
    return walls

def is_cheating_move(og_map, c1, c2, start, end, visited):
    map = copy.deepcopy(og_map)
    x,y = c1
    xx,yy = c2
    map[y][x] = '.'
    map[yy][xx] = '.'    
    
    graph = buildGraphFromMap_v3(map, '#', is_connectedv2)
    
    #printGraph(graph)  
    shortest_distance, path = dijkstra_shortest_path(graph, start, end)
    #path = find_shortest_path(graph, start, end)
    
    p = str(path)
    print(str(c2) in visited)
    if path and str(c2) in p:
     #   shortest_distance = len(p)
        
        print(visited)
        #print(str(c2) in p)
        visited.add(p)
        return True, shortest_distance, visited
    else:
        return False, -1, visited



def day20_1(data):    
    data = read_input(2024, "20_teste")    
    result = 0
    
    map = buildMapGrid(data, initValue='.', withPadding = False)    
    walls = find_walls(map)
    
    sizeX = len(map[0])
    sizeY = len(map)
    cheatings_moves = defaultdict(set)
    
    
    graph = buildGraphFromMap_v3(map, '#', is_connectedv2)
    start = find_starting_point(map, "S")
    end = find_starting_point(map, "E")      
    shortest_distance, path = dijkstra_shortest_path(graph, start, end)
    print(shortest_distance)
    
    visited = set()
    for x,y in path:     
    #for x,y in walls:
        east = (x+1, y)
        west = (x-1, y)
        north = (x, y-1)
        south = (x, y+1)
        
        if 0 < east[0] < sizeX-1 and 0 < east[1] < sizeY and map[east[1]][east[0]] == '#':
            res, dist, visited = is_cheating_move(map, (x,y), (east[0], east[1]), start, end, visited)
            if res and dist < shortest_distance:
                cheatings_moves[dist].add( ((x,y), (east[0]+1, east[1]) ) )
        
        if 0 < west[0] < sizeX and 0 <= west[1] < sizeY and map[west[1]][west[0]] == '#':                        
            res, dist, visited = is_cheating_move(map, (x,y), (west[0], west[1]), start, end, visited)
            if res and dist < shortest_distance:
                cheatings_moves[dist].add( ((x,y), (west[0]-1, west[1]) ) )
        
        if 0 < north[0] < sizeX and 0 <= north[1] < sizeY and map[north[1]][north[0]] == '#': 
            res, dist, visited = is_cheating_move(map, (x,y), (north[0], north[1]), start, end, visited)
            if res and dist < shortest_distance:
                cheatings_moves[dist].add( ( (x,y),(north[0], north[1]-1) ) )
        
        if 0 < south[0] < sizeX and 0 <= south[1] < sizeY and map[south[1]][south[0]] == '#': 
            res, dist, visited = is_cheating_move(map, (x,y), (south[0], south[1]), start, end, visited)
            if res and dist < shortest_distance:
                cheatings_moves[dist].add( ( (x,y),(south[0], south[1]+1) ) )
        
    
    for k, v in sorted(cheatings_moves.items(), reverse = True):    
    #for k in cheatings_moves.keys().sort():
       print("There are",len(v),"cheats that save", shortest_distance - k,"picoseconds.")
    
    AssertExpectedResult(587, result)
    return result 

def day20_2(data):    
    data = read_input(2024, "20_teste")    
    result = 0    
    
    AssertExpectedResult(587, result)
    return result 

#endregion

#region ##### Day 21 #####

def day21_1(data):    
    data = read_input(2024, "21_teste")    
    result = 0    
    
    AssertExpectedResult(587, result)
    return result 

def day21_2(data):    
    data = read_input(2024, "21_teste")    
    result = 0    
    
    AssertExpectedResult(587, result)
    return result 



#endregion

#region ##### Day 22 #####

def mix_secret_number(secret, n):
    return secret ^ n

def prune_secret_number(secret):
    return secret % 16777216

def next_secret_number(secret):
    n1 = secret * 64
    n1 = mix_secret_number(secret, n1)
    n1 = prune_secret_number(n1)
    secret = n1

    n2 = int(secret/32)
    n2 = mix_secret_number(secret, n2)
    n2 = prune_secret_number(n2)
    secret = n2

    n3 = secret * 2048
    n3 = mix_secret_number(secret, n3)
    n3 = prune_secret_number(n3)
    secret = n3
    return secret


def day22_1(data):    
    data = read_input(2024, "22")    
    result = 0
    times = 2000
    secrets = []

    for line in data:
        secrets.append(int(line))
    
    for secret in secrets:
        for _ in range(times):
            secret = next_secret_number(secret)
        #print(secret)
        result += secret

       
    AssertExpectedResult(18261820068, result)
    return result


# 1687 low
# 1639 low
def day22_2(data):    
    data = read_input(2024, "22")    
    result = 0
    times = 2000
    secrets = []

    for line in data:
        secrets.append(int(line))
    
    res = defaultdict(list)
    buyer = 0
    for secret in secrets:
        buyer +=1
        diff = int(str(secret)[-1])
        key = []
        for _ in range(times):
            secret = next_secret_number(secret)
            price = int(str(secret)[-1])
            key.append(price-diff)            
            
            if len(key) == 4:
                val = ",".join(str(x) for x in key)
                l = [(x,y) for x,y in res[val] if x != buyer]
                buyer_price = [(x,y) for x,y in res[val] if x == buyer]
               
                if buyer_price:
                    _, p = buyer_price[0]
                    l.append((buyer,p))
                else:
                    l.append((buyer,price))
                res[val] = l

                key = key[1:]

            diff = price
        #print(secret)

    
    for results in res.values():
        res_bananas = 0
        for _, bananas in results:
            res_bananas += bananas
        
        if res_bananas > result:
            result = res_bananas

    AssertExpectedResult(2044, result)
    return result

#endregion

#region ##### Day 23 #####

def day23_1(data):    
    data = read_input(2024, "23_teste")    
    result = 0
    
    historian_pcs = set()
    total_pcs = set()
    network_map = defaultdict(list)
    network_map2 = defaultdict(list)
    for line in data:
        pc = parse("{:w}-{:w}", line)
        network_map[pc[0]].append(pc[1])
        network_map[pc[1]].append(pc[0])
        total_pcs.add(pc[0])
        total_pcs.add(pc[1])
        
        if pc[0][0] == 't':
            historian_pcs.add(pc[0])
        if pc[1][0] == 't':
            historian_pcs.add(pc[1])
    
    #print(total_pcs)
    combinations = [p for p in itertools.product(total_pcs, repeat=3) if p[0] != p[1] and p[0] != p[2] and p[1] != p[2] and (p[0][0] == 't' or p[1][0] == 't' or p[2][0] == 't')]
    aux = set()
    for x,y,z in combinations:
        l = [x,y,z]
        l.sort()
        #print(l)
        s = ','.join(l)
        aux.add(s)
    
    for l in aux:
        l = l.split(',')
        if l[1] in network_map[l[0]] and l[2] in network_map[l[0]] and l[1] in network_map[l[2]]:
            result += 1
    
    AssertExpectedResult(587, result)
    return result 

def day23_2(data):    
    data = read_input(2024, "23")    
    result = 0
    
    network_map = defaultdict(set)
    for line in data:
        pc = parse("{:w}-{:w}", line)
        network_map[pc[0]].add(pc[1])
        network_map[pc[1]].add(pc[0])
    
    bron_kerbosch(set(), set(network_map.keys()), set(), network_map)
    # the result was taken from the prints done in bron_kerbosch
    result = ['fo', 'au', 'nt', 'im', 'qz', 'so', 'cm', 'be', 'rr', 'hh', 'am', 'ha', 'os']
    result.sort()
    
    
    AssertExpectedResult(587, result)
    return result 

#endregion

#region ##### Day 24 #####

def process_gates(wires, gates):
    
    while gates:
        gate = gates.pop(0)
        operator, operand1, operand2, output = gate
        try:
            if operator == 'AND':
                wires[output] = wires[operand1] and wires[operand2]
            elif operator == 'OR':
                wires[output] = wires[operand1] or wires[operand2]
            elif operator == 'XOR':
                if (wires[operand1] and not wires[operand2]) or (not wires[operand1] and wires[operand2]):
                    wires[output] = 1
                else:
                    wires[output] = 0
        except KeyError:
            gates.append(gate)
    
    return wires

def day24_1(data):    
    data = read_input(2024, "24")    
    result = 0
    wires = {}
    gates = []
    parse_gates = False
    
    for line in data:
        if parse_gates:
            vals = parse("{:w} {:w} {:w} -> {:w}", line)
            # (AND, op1, op2, out)
            gates.append((vals[1], vals[0], vals[2], vals[3] ))
        else:
            vals = parse("{:w}: {:d}", line)
            if vals:
                wires[vals[0]] = vals[1]
            else:
                parse_gates = True  
    
  
    wires = process_gates(wires, gates)
    
    wires = {k: v for k, v in sorted(list(wires.items()), reverse = True)}
    #for k in wires.keys():
    #    print(k,":", wires[k])
    
    z_wires = [wires[z] for z in wires.keys() if z[0] == 'z']
    binary = ''.join(map(str,z_wires))
    #print(binary)
    result = int(binary, 2)
  
    AssertExpectedResult(587, result)
    return result 

def day24_2(data):    
    data = read_input(2024, "24_teste")    
    result = 0    
    
    AssertExpectedResult(587, result)
    return result 

#endregion

#region ##### Day 25 #####

def day25_1(data):    
    data = read_input(2024, "25_teste")    
    result = 0    
    
    AssertExpectedResult(587, result)
    return result 

#endregion

if __name__ == "__main__":
    # override timeout
    main(sys.argv, globals(), AOC_EDITION_YEAR, 28800)

