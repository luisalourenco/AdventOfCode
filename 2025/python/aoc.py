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
    data = read_input(2025, "02") 
    result = 0     
    line = data[0].split(",")
    for l in line:
        [low, high] = ints(l.split("-"))

        #print("checking [",low,high,"]")

        for n in range(low, high+1):
            num = str(n)
            #print("check", num)

            nn = ''
            for i in num:
                nn += i
                if len(nn) == len(num):
                    break
                #print("testing",nn)
                
                invalid_num = nn *  len(num)
                invalid_num = invalid_num[:len(num)]
                
                #print("generated:",invalid_num, "is invalid?", invalid_num == num and (int(invalid_num)/int(nn)) % 1 == 0)
                
                if invalid_num == num and (int(invalid_num)/int(nn)) % 1 == 0:
                    result += n
                    break

    AssertExpectedResult(69564213293, result)
    return result

def day3_1(data):    
    data = read_input(2025, "03") 
    result = 0     
    for line in data:
        perm = combinations(line,2)
        max_jolt = 0
        for i in perm:
            joltage = (int(''.join(i)))
            if joltage > max_jolt:
                max_jolt = joltage
        #print(max_jolt)
        result += max_jolt
        
            
    AssertExpectedResult(17034, result)
    return result


def day3_2(data):    
    data = read_input(2025, "03_teste") 
    result = 0     
    for line in data:
        print("checking", line)
        res = ''
        enabled_positions = set()
        
        for i in range(len(line)):
            aux = len(res)
            print("checking digit", line[i])
            
            for j in range(i, len(line)):
                #print(i,j)
                #print("checking", line[i], "with", line[j])
                if int(int(line[i] < line[j])) and j not in enabled_positions:
                    res += line[j]  
                    enabled_positions.add(j)
                    print("digit", line[j], "is greater")                 
                    break
            
            # digit to check is bigger than all the rest
            if aux == len(res) and i not in enabled_positions:
                enabled_positions.add(i)
                print("all digits are smaller")                 
                res += line[i]
            elif j not in enabled_positions:
                res += line[j]
                enabled_positions.add(j)
                print("default")                 
            if len(res) == 12:
                break
        print("res:",res)

        max_jolt = 0
        
        '''
        for i in perm:
            joltage = (int(''.join(i)))
            if joltage > max_jolt:
                max_jolt = joltage
        '''
        #print(max_jolt)
        result += max_jolt
        
            
    AssertExpectedResult(5782, result)
    return result

def remove_rolls_of_paper(map):
    map = copy.deepcopy(map)
    result = 0
    rows = len(map)
    columns = len(map[0])
    for y in range(rows):          
        for x in range(columns):            
            #print("testing:",x,y)
            if map[y][x] == '@':
                rolls = 0
                if map[y-1][x] == '@':
                    rolls+=1
                if map[y+1][x] == '@':
                    rolls+=1
                if map[y-1][x-1] == '@':
                    rolls+=1
                if map[y-1][x+1] == '@':
                    rolls+=1
                if map[y+1][x-1] == '@':
                    rolls+=1
                if map[y+1][x+1] == '@':
                    rolls+=1
                if map[y][x-1] == '@':
                    rolls+=1
                if map[y][x+1] == '@':
                    rolls+=1
                
                if rolls < 4:
                    result+=1
                    map[y][x] = 'x'
                    #print("roll removed at",x,y)
    
    return result, map

def day4_1(data):    
    data = read_input(2025, "04") 
    result = 0     
    map = buildMapGrid(data)

    result, _ = remove_rolls_of_paper(map)
            
    AssertExpectedResult(1489, result)
    return result


def day4_2(data):    
    data = read_input(2025, "04") 
    total_rolls = 0     
    map = buildMapGrid(data)

    while True:
        result, map = remove_rolls_of_paper(map)
        total_rolls += result
        if result == 0:
            break
            
    AssertExpectedResult(1489, total_rolls)
    return total_rolls


def day5_1(data):    
    data = read_input(2025, "05") 
    result = 0     
    available_ingredients = []
    fresh = []

    for line in data:
        range_ig = parse("{:d}-{:d}", line)
        

        if range_ig == None:
            if line != '':
                available_ingredients.append(int(line))
        else:
           fresh.append((range_ig[0], range_ig[1]))
    
    for ig in available_ingredients:
        for low, high in fresh:
            if low <= ig <= high:
                result+= 1
                break

    AssertExpectedResult(789, result)
    return result

def day5_2(data):    
    data = read_input(2025, "05_d") 
    result = 0     
    fresh = []

    for line in data:
        range_ig = parse("{:d}-{:d}", line)      

        if range_ig == None:
            break
        else:
           fresh.append((range_ig[0], range_ig[1]))

    #fresh.sort(key=lambda tup: tup[0]) 
    '''
    l < lo and lo <= h --> lo = l
    lo < l < ho and lo < h < ho --> remove [l,h]
    l < ho and ho < h --> ho = h    
    '''
    
    for i in range(4):
        print(fresh)
        new_fresh = []
        fresh_o = copy.deepcopy(fresh)
        to_remove = []

        #while len(fresh) > 0:
        for i in range(len(fresh)):
            print(i,":",fresh)
            low, high = fresh.pop(0)
            
            for low_o, high_o in fresh_o:
                if low == low_o and high == high_o:
                    continue
             
                #print("checking:", low,high,"with", low_o,high_o)

                if low <= low_o and low_o <= high:
                    if (low, high_o) not in new_fresh:
                        #print("merged left to:",(low, high_o))
                        new_fresh.append((low, high_o)) 
                    if (low, high) not in new_fresh:               
                        to_remove.append((low, high))
                    if (low_o, high_o) not in new_fresh:
                        to_remove.append((low_o, high_o))
                    break
                elif low <= high_o and high_o <= high:
                    if (low_o, high) not in new_fresh:
                        #print("merged right to:",(low_o, high))
                        new_fresh.append((low_o, high))
                    if (low, high) not in new_fresh:
                        to_remove.append((low, high))
                    if (low_o, high_o) not in new_fresh:
                        to_remove.append((low_o, high_o))
                    break
                elif low >= low_o and high <= high_o:
                    #print("contained in:",(low_o, high_o))
                    to_remove.append((low, high))
                    
                    break

            new_fresh.append((low, high))
            for i in to_remove:
                if i in new_fresh:
                    new_fresh.remove(i)
            
            #print("res:",new_fresh)
            #print("to_remove:",to_remove)
            #print()
        fresh = list(set(new_fresh))
    
            
    print(fresh)
    for l,h in fresh:
        result+= (h-l+1)

    # 221057730900912 low
    # 219892874544636 low
    # 308518728212455 wrong (low)

    AssertExpectedResult(1489, result)
    return result

def day6_1(data):    
    data = read_input(2025, "06") 
    result = 0     

    results = defaultdict()

    for line in data:
        aux = line.strip().split(' ')
        i = 0
        for n in aux:
            
            if i not in results:
                results[i] = []

            if n != '':
                if n != '*' and n != '+':
                    #print(n)
                    results[i].append(int(n))
                    #results.append(int(n))
                else:
                    results[i].append(n)
                i +=1
    
    for key in results:
        r = results[key]
        op = r[len(r)-1]
        if op == '+':
            for n in r:
                if n != '+':
                    result += n
        else:
            aux = 1
            for n in r:
                if n != '*':
                    aux *= n
            result += aux
            
    #print(results)
    AssertExpectedResult(4951502530386, result)
    return result

def day6_2(data):    
    data = read_input(2025, "06") 
    result = 0     
    
    sums = []    
    rows = len(data)   
    cols = len(data[0])
    #print(rows,cols)
    
    for line in data:
        sums.append(list(line))

    #print(sums)
    #tlist = list(zip(*sums))

    tlist = list(map(list, itertools.zip_longest(*sums, fillvalue=None)))
    
    op = ''
    num = ''
    res = 0

    
    #print(tlist)
    #print()
    for ll in tlist:
        empty = 0

        #print("ll",ll)
        for l in ll:
            #print("l",l)

            if l == '*':
                #print("l == *")
                op = '*'
                res = 1
                #print("num final *:",num)
                res *= int(num)
                num = ''
                empty = 0
            
            elif l == '+':
                #print("l == +")
                op = '+'
                res = 0
                #print("num final +:",num)
                res += int(num)
                num = ''
                empty = 0
            
            elif empty == rows-1:
                #print("num final op",num,op)
                if num != '':
                    if op == '*':
                        res *= int(num)
                    elif op == '+':
                        res += int(num)
                else:
                    result += res
                    #print("problem result:",res)
                    res = 0
                    op = ''                              
                num = ''
                empty = 0
                break
            elif l != ' ':
                num += l
                empty +=1
            elif l == ' ':                
                empty +=1
                #print("l == '  '",empty)
    if op != '':
        result += res
        #print("problem result2:",res)
        res = 0
        op = '' 
      
            
    AssertExpectedResult(8486156119946, result)
    return result

def day7_1(data):    
    data = read_input(2025, "07_t") 
    result = 0
    map = buildMapGrid(data, withPadding=False)
    beams = []
    rows = len(map)
    columns = len(map[0])
    splits = set()
    printMap(map)

    for x in range(columns):          
        if map[0][x] == 'S':
            beams.append((x,1))
            break
    
    while len(beams) > 0:
        x,y = beams.pop()
        #print("testing:",x,y,map[y][x])
        if map[y][x] == '^' and (x,y) not in splits:
            splits.add((x,y))
            #print("split at",x,y,map[y][x])
            beams.append((x-1,y))
            beams.append((x+1,y))
        elif y < rows-1:
            beams.append((x,y+1))
            
    result = len(splits)
    #1725 high
    #212 low
    AssertExpectedResult(1489, result)
    return result

def day9_1(data):    
    data = read_input(2025, "09") 
    result = 0
    red_tiles = []

    for line in data:
        x,y = line.split(",")
        red_tiles.append((int(x),int(y)))

    for x,y in red_tiles:
        for x2,y2 in red_tiles:
            area = (abs(x-x2)+1) * (abs(y-y2)+1)
            if area > result:
                result = area
            
    AssertExpectedResult(1489, result)
    return result

def day11_1(data):    
    data = read_input(2025, "11") 
    result = 0
    graph = defaultdict()
    for line in data:
        n = line.split(":")
        graph[n[0]] = []
        for node in n[1].strip().split(" "):
            graph[n[0]].append(node)

    #print(graph)
    paths  = find_all_paths(graph, 'you','out')
    result = len(paths)

            
    AssertExpectedResult(699, result)
    return result

def count_all_paths(graph, src, dst):
    """
    Return the number of distinct simple paths from ``src`` to ``dst`` in a
    graph expressed as an adjacency‑list dictionary.

    Parameters
    ----------
    graph : dict
        {node: [neighbour1, neighbour2, ...]}
    src   : hashable
        Start vertex.
    dst   : hashable
        Target vertex.

    Returns
    -------
    int
        Number of different simple paths from ``src`` to ``dst``.
    """
    # Use an explicit stack to avoid recursion limits on large graphs.
    # Each stack element is a tuple (current_node, visited_set).
    stack = [(src, {src})]          # start with src marked as visited
    path_count = 0

    while stack:
        node, visited = stack.pop()

        if len(visited) > 10000:
            continue
        

        # Destination reached → one more valid path
        if node == dst:
            path_count += 1
            continue

        # Explore neighbours that have not been visited yet
        for nbr in graph.get(node, []):
            if nbr not in visited:               # keep the path simple
                # Create a new visited set for the next branch.
                # Using ``visited | {nbr}`` creates a fresh set without
                # mutating the current one (important for correctness).
                stack.append((nbr, visited | {nbr}))

    return path_count



def day11_2(data):    
    data = read_input(2025, "11") 
    result = 0
    graph = defaultdict()
    for line in data:
        n = line.split(":")
        graph[n[0]] = []
        for node in n[1].strip().split(" "):
            graph[n[0]].append(node)

    #print(graph)
    #paths  = find_all_paths(graph, 'svr','out')
    paths1  = find_all_paths(graph, 'svr','fft')
    #paths2  = count_all_paths(graph, 'fft','dac')
   # paths3  = find_all_paths(graph, 'dac','out')
    
    print("svr -> fft",len(paths1))
    #print("fft -> dac",len(paths2))
    #print("dac -> out",len(paths3))
    #result = len(paths)

    print(paths1)
    #print(paths3)

            
    AssertExpectedResult(699, result)
    return result


def day12_1(data):    
    data = read_input(2025, "12") 
    result = 0
    shapes = []
    count = 0
    for line in data:
        if line != '':
            l = line.split(":")
            if len(l) == 2:
                if l[1] != '':
                    #print("l",l)
                    ll = ints(l[0].split("x"))
                    area = ll[0]*ll[1]
                    presents = ints(l[1].strip().split(" "))
                    total = 0
                    for i in range(len(presents)):
                        total += shapes[i]*presents[i]
                    #print(l,total, area)
                    if total <= area:
                        result+=1

            else:
                count += l[0].count("#")
        else:
            shapes.append(count)
            count = 0


    #print(shapes)

    AssertExpectedResult(699, result)
    return result
       


if __name__ == "__main__":
    # override timeout
    main(sys.argv, globals(), AOC_EDITION_YEAR, 28800)

