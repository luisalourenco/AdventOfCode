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
        #[x for x in fewest_cubes_per_game[game]]
        result += reduce(lambda x, y: x*y,fewest_cubes_per_game[game])
             
    AssertExpectedResult(0, result)
    return result

#endregion

#region ##### Day 3 #####

#Day 3, part 1: 8401 (0.075 secs)
#Day 3, part 2: 2641 (0.188 secs)
def day3_1(data):
    #data = read_input(2023, "01_teste")    
    result = 0    
    #for line in data:           
    AssertExpectedResult(0, result)
    return result


# Day 3, part 2: 2641 (0.188 secs)
def day3_2(data):
    ##data = read_input(2023, "01_teste")    
    result = 0    
    #for line in data:           
    AssertExpectedResult(0, result)
    return result

#endregion

if __name__ == "__main__":
    # override timeout
    main(sys.argv, globals(), AOC_EDITION_YEAR, 900)

