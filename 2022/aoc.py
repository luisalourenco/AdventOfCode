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

# UPDATE THIS VARIABLE
AOC_EDITION_YEAR = 2022

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



################# Advent of Code - 2022 Edition #################

#region ##### Day 1 #####

#Day 1, part 1: 73211 (0.056 secs)
#Day 1, part 2: 213958 (0.006 secs)
def day1_1(data):
    #data = read_input(2022, "01_teste")    
    result = 0
    max_calories = 0
    for line in data:
        if line != '':
            n = int(line)
            result += n   
        else:
            if result > max_calories:
                max_calories = result
            result = 0           
    
    AssertExpectedResult(73211, max_calories)
    return max_calories


def day1_2(data):
    #data = read_input(2022, "01t")    
    result = 0
    max_calories = 0
    elves = []
    for line in data:
        if line != '':
            n = int(line)
            result += n   
        else:
            elves.append(result)
            if result > max_calories:
                max_calories = result
            result = 0

    elves.sort(reverse=True)
    # this is not the same as above :wat:
    #sorted(elves, reverse=True) 
    result = sum(elves[:3])
    AssertExpectedResult(213958, result)
    return result

#endregion

#region ##### Day 2 #####

def compute_round_score(opponent_play, my_play, part2 = False):
    round_score = 0
    # A,X - Rock
    # B,Y - Paper
    # C,Z - Scissors
    play = ''
    if part2:
        if my_play == 'X':
            round_score = 0
            if opponent_play == 'A':
                play = 'C'
            if opponent_play == 'B':
                play = 'A'
            if opponent_play == 'C':
                play = 'B'
        if my_play == 'Y':
            round_score = 3
            if opponent_play == 'A':
                play = 'A'
            if opponent_play == 'B':
                play = 'B'
            if opponent_play == 'C':
                play = 'C'  
        if my_play == 'Z':
            round_score = 6
            if opponent_play == 'A':
                play = 'B'
            if opponent_play == 'B':
                play = 'C'
            if opponent_play == 'C':
                play = 'A'        
    else:
        if my_play == 'X':
            if opponent_play == 'A':
                round_score = 3
            if opponent_play == 'B':
                round_score = 0
            if opponent_play == 'C':
                round_score = 6
            
        if my_play == 'Y':
            if opponent_play == 'A':
                round_score = 6
            if opponent_play == 'B':
                round_score = 3
            if opponent_play == 'C':
                round_score = 0
            
        if my_play == 'Z':
            if opponent_play == 'A':
                round_score = 0
            if opponent_play == 'B':
                round_score = 6
            if opponent_play == 'C':
                round_score = 3 
                
    return round_score, play

#Day 2, part 1: 11150 (0.064 secs)
#Day 2, part 2: 8295 (0.013 secs)
def day2_1(data):
    #data = read_input(2022, "02t")    
    # A,X - Rock
    # B,Y - Paper
    # C,Z - Scissors
    scores = {'A': 1, 'X': 1, 'B': 2, 'Y': 2, 'C': 3, 'Z': 3}
    round_score = 0
    score = 0
    for line in data:
        play = line.split(' ')
        my_score, _ = compute_round_score(play[0], play[1])
        score += (my_score + scores[play[1]])
            
    
    AssertExpectedResult(11150, score)
    return score

def day2_2(data):
    #data = read_input(2022, "02t")    
    result = 0
    # A,X - Rock
    # B,Y - Paper
    # C,Z - Scissors
    scores = {'A': 1, 'X': 1, 'B': 2, 'Y': 2, 'C': 3, 'Z': 3}
    round_score = 0
    score = 0
    for line in data:
        play = line.split(' ')
        my_score, my_play = compute_round_score(play[0], play[1], part2 = True)        
        score += (my_score + scores[my_play])            
    
    AssertExpectedResult(8295, score)
    return score

#endregion

#region ##### Day 3 #####

#Day 3, part 1: 8401 (0.075 secs)
#Day 3, part 2: 2641 (0.188 secs)
def day3_1(data):
    #data = read_input(2022, "03t")    
    
    rucksacks = []
    result = 0
    for line in data:
        if line:
            rucksacks.append(line)            
    
    for rucksack in rucksacks:
        middle = int(len(rucksack)/2)
        first_compartment = rucksack[:middle]
        second_compartment = rucksack[middle:]
        common_item = ''.join(set(first_compartment).intersection(second_compartment))
        result += get_priority(common_item)                  
    
    AssertExpectedResult(8401, result)
    return result


def get_priority(item):
    if item.islower():
        return (ord(item)-96)
    else:
        return (ord(item)-65+27)

#8067887
# Day 3, part 2: 2641 (0.188 secs)
def day3_2(data):
    #data = read_input(2022, "03t")        
    
    rucksacks = []
    result = 0
    visited = set()
    for line in data:
        if line:
            rucksacks.append(line)            
    
    for three_rucksacks in itertools.combinations(rucksacks, 3):
        if three_rucksacks[0] in visited or three_rucksacks[1] in visited or three_rucksacks[2] in visited:
            continue
        
        common_item1 = ''.join(set(three_rucksacks[0]).intersection(three_rucksacks[1]))
        common_item2 = ''.join(set(three_rucksacks[2]).intersection(common_item1))
        
        if len(common_item2) == 1:
            result += get_priority(common_item2)            
            visited.add(three_rucksacks[0])
            visited.add(three_rucksacks[1])
            visited.add(three_rucksacks[2])            
      
    AssertExpectedResult(2641, result)
    return result

#endregion


#region ##### Day 4 #####

def day3_1(data):
    #data = read_input(2022, "04t")    
    
    rucksacks = []
    result = 0
    for line in data:
        if line:
            rucksacks.append(line)           
    

    AssertExpectedResult(0, result)
    return result

#endregion



if __name__ == "__main__":
    # override timeout
    main(sys.argv, globals(), AOC_EDITION_YEAR, 900)

