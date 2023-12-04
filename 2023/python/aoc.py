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
from functools import lru_cache
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
from aocd import get_data
from aocd import submit

# UPDATE THIS VARIABLE
AOC_EDITION_YEAR = 2023

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
print(FILE_DIR)
sys.path.insert(0, FILE_DIR + "/")
sys.path.insert(0, FILE_DIR + "/../")
sys.path.insert(0, FILE_DIR + "/../../")

DEBUG_MODE = False

from common.mathUtils import *
from common.utils import *# read_input, main, clear, AssertExpectedResult, ints, printGridsASCII  # NOQA: E402
from common.mapUtils import printMap, buildMapGrid, buildGraphFromMap
from common.graphUtils import dijsktra, printGraph, find_all_paths, find_path, find_shortest_path, find_shortest_pathOptimal, bfs, dfs, Graph, hashable_lru
from common.aocVM import *
from lark import Lark, Transformer, v_args, UnexpectedCharacters, UnexpectedToken
from pyformlang.cfg import Production, Variable, Terminal, CFG, Epsilon
from itertools import islice

# pylint: enable=import-error
# pylint: enable=wrong-import-position


################# Advent of Code - 2023 Edition #################

#region ##### Day 1 #####

#Day 1, part 1: 55029 (0.036 secs) 
#Day 1, part 2: 55686 (0.008 secs) 
def day1_1(data):    
    data = read_input(2023, "01")    
    first_number = ''
    last_number = ''
    result = 0

    for line in data:
        if line != '':
            for c in line:
                if c.isdigit():
                    if first_number == '':
                        first_number = c
                        last_number = c
                    else:
                        last_number = c
            result += int(first_number + last_number)
            print(first_number + last_number)
            first_number = ''
            last_number = ''

            

    AssertExpectedResult(55029, result)
    return result


def day1_2(data):
    #data = read_input(2023, "01_teste")    

    numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }

    first_number = ''
    last_number = ''
    result = 0
    
    for line in data:
        if line != '':
            pos = 0
            for c in line:
                
                if c.isdigit():
                    if first_number == '':
                        first_number = c
                        last_number = c
                    else:
                        last_number = c
                else:
                    p1 = line[pos:pos+3]
                    p2 = line[pos:pos+4]
                    p3 = line[pos:pos+5]
                    n1 = numbers.get(p1)
                    n2 = numbers.get(p2)
                    n3 = numbers.get(p3)
         
                    if not n1:
                        n = n2
                        if not n2:
                            n = n3
                        if not n3:
                            n = n2 
                    else:
                        n = n1
                    if n:
                        
                        if first_number == '':
                            first_number = str(n)
                            last_number = str(n)
                        else:
                            last_number = str(n)

                pos += 1 


            result += int(first_number + last_number)
            #print(first_number + last_number)
            first_number = ''
            last_number = ''

            

    AssertExpectedResult(55686, result)
    return result

#endregion

#region ##### Day 2 #####

def check_possible_games(games, cubes_limit):
    possible_games = []
    for game in games.keys():
        sets = games[game]
        possible = True
        for s in sets:
            if not (s['red'] <= cubes_limit['red'] and s['green'] <= cubes_limit['green'] and s['blue'] <= cubes_limit['blue']):
                possible = False
        if possible:
            possible_games.append(int(game))
    return possible_games

def parse_games(data):
    games = {}
    for line in data:           
        game = [parse("Game {}: {}", line)][0]
        cubes = game[1].split(";")        
        games[game[0]] = []
        for cube_set in cubes:
            red = search('{:d} red', cube_set)
            green = search('{:d} green', cube_set)
            blue = search('{:d} blue', cube_set)
            c = {
                'red':  0 if (red is None) else red[0],
                'green': 0 if (green is None) else green[0],
                'blue': 0 if (blue is None) else blue[0]
            } 
            games[game[0]].append(c)
    return games

#Day 2, part 1: 1734 (0.116 secs)
#Day 2, part 2: 70387 (0.021 secs)
def day2_1(data):
    #data = read_input(2023, "02_teste")    
    result = 0  
    cubes_limits = {
        'red': 12,
        'green': 13,
        'blue': 14
    }  
    
    games = parse_games(data)          
    possible_games = check_possible_games(games, cubes_limits)
    result = sum(possible_games)
    AssertExpectedResult(1734, result)
    return result

def get_fewest_cubes_per_game(games):
    fewest_cubes_per_game = {}

    for game in games.keys():
        fewest_cubes = {
            'red': 0,
            'green': 0,
            'blue': 0 
        }
        for s in games[game]:
            if s['red'] > fewest_cubes['red']:
                fewest_cubes['red'] = s['red']
            if s['green'] > fewest_cubes['green']:
                fewest_cubes['green'] = s['green']
            if s['blue'] > fewest_cubes['blue']:
                fewest_cubes['blue'] = s['blue']
        fewest_cubes_per_game[game] = fewest_cubes.values()
    return fewest_cubes_per_game

def day2_2(data):
    #data = read_input(2023, "02_teste")    
    result = 0    
    
    games = parse_games(data) 
    fewest_cubes_per_game = get_fewest_cubes_per_game(games)
    for game in fewest_cubes_per_game.keys():
        result += reduce(lambda x, y: x*y, fewest_cubes_per_game[game])
             
    AssertExpectedResult(0, result)
    return result

#endregion

#region ##### Day 3 #####


def check_if_is_part_number(engine_map, x, y, part_number):
    rows = len(engine_map)
    columns = len(engine_map[0])
    size = len(part_number)
    above = []
    below = []
                
    #print("checking if", part_number,"is part number with values (x,y):",x,y)
    
    xx = x-size-1
    if xx < 0:
        xx = 0
        left = ''
    else:
        left = engine_map[y][xx]
            
    xxx = x
    if xxx == columns:
        xxx = columns - 1
        right = ''
    else:
        right = engine_map[y][xxx]    
    # row above
    if y != 0:
        above = engine_map[y-1][xx : x+1]
    
    # row below
    if y+1 != rows:                  
        below = engine_map[y+1][xx : x+1]        
    
    is_part_number = False
    symbols_above = sum(1 for c in above if not c.isdigit() and c != '.')
    symbols_below = sum(1 for c in below if not c.isdigit() and c != '.')
    
    
    #print(above)
    #print(left, end="")
    #print(part_number, end="")
    #print(right)
    #print(below)
    #print("symbols_above", symbols_above)
    #print("symbols_below", symbols_below)

    if symbols_above > 0:        
        is_part_number = True
    if symbols_below > 0: 
        is_part_number = True
    if left != '' and left != '.' and not left.isdigit():
        is_part_number = True
    if right != '' and right != '.' and not right.isdigit():
        is_part_number = True
    
    return is_part_number

# Test case values
#Part 1: 925
#Part 2: 6756
#
def get_part_numbers(engine_map):
    rows = len(engine_map)
    columns = len(engine_map[0])      
    part_numbers = []
    ignored_numbers = []
    part_number = ''
    
    for y in range(rows):   
        if part_number != '':
            is_part_number = check_if_is_part_number(engine_map, columns, y-1, part_number)
            #print(part_number,'is part number?', is_part_number)
            #print()
            if not is_part_number:
                ignored_numbers.append(int(part_number))
            else:
                part_numbers.append(int(part_number))
            part_number = ''
            
        for x in range(columns):
            elem = engine_map[y][x]
            if elem.isdigit():
                part_number += elem
            elif part_number != '':                
                is_part_number = check_if_is_part_number(engine_map, x, y, part_number)                    
                
                #print(part_number,'is part number?', is_part_number)
                #print()
                if not is_part_number:
                    ignored_numbers.append(int(part_number))
                else:
                    part_numbers.append(int(part_number))
                    
                part_number = '' 
                               
    if part_number != '':
        is_part_number = check_if_is_part_number(engine_map, x, y, part_number)
        #print(part_number,'is part number?', is_part_number)
        #print()
        if not is_part_number:
            ignored_numbers.append(int(part_number))
        else:
            part_numbers.append(int(part_number))
            
    return part_numbers
                

#Day 3, part 1: 8401 (0.075 secs)
#Day 3, part 2: 2641 (0.188 secs)
# 505770 too low
def day3_1(data):
    #data = read_input(2023, "03_teste")    
    result = 0    
    
    engine_map = buildMapGrid(data, initValue='.')       
    
    y = 0
    for line in data:
        for x in range(len(line)):
            engine_map[y][x] = line[x]
        y += 1
    
    #printMap(engine_map)
    part_numbers = get_part_numbers(engine_map)
    #print(part_numbers)
    result = sum(part_numbers)
    
    AssertExpectedResult(532445, result)
    return result


# Day 3, part 2: 2641 (0.188 secs)
def day3_2(data):
    data = read_input(2023, "03_teste")     
    result = 0    
    engine_map = buildMapGrid(data, initValue='.')       
    
    y = 0
    for line in data:
        for x in range(len(line)):
            engine_map[y][x] = line[x]
        y += 1
    
    #printMap(engine_map)
    part_numbers = get_part_numbers(engine_map)  
    print(part_numbers)
    
    AssertExpectedResult(0, result)
    return result

#endregion

#region ##### Day 4 #####
#Day 4, part 1: 26443 (0.096 secs)
def day4_1(data):
    #data = read_input(2023, "04_teste")    
    result = 0    
    
    for line in data:
        points = 0
        ndata = parse("Card {}: {}", line)[1]
        numbers = ndata.split("|")
        winning_numbers = ints(numbers[0].split())
        played_numbers = ints(numbers[1].split() )
        
        winning_numbers.sort()
        for n in played_numbers:
            if n in winning_numbers:
                points = 1 if points == 0 else points*2
        result += points
    
    AssertExpectedResult(26443, result)
    return result

#endregion



if __name__ == "__main__":
    # override timeout
    main(sys.argv, globals(), AOC_EDITION_YEAR, 900)

