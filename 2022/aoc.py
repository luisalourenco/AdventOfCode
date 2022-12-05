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
from parse import parse

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

#Day 4, part 1: 513 (0.066 secs)
#Day 4, part 2: 878 (0.017 secs)
def day4_1(data):
    #data = read_input(2022, "04t")    
    
    result = 0
    for line in data:
        if line:
           sections_pairs = line.split(",")
           first_section = sections_pairs[0].split('-')
           section1 = set(range(int(first_section[0]), int(first_section[1])+1))
           
           second_section = sections_pairs[1].split('-')
           section2 = set(range(int(second_section[0]), int(second_section[1])+1))
          
           if section1.issubset(section2) or section2.issubset(section1):
               result+=1
    

    AssertExpectedResult(513, result)
    return result

def day4_2(data):
    #data = read_input(2022, "04t")    
    
    result = 0
    for line in data:
        if line:
           sections_pairs = line.split(",")
           first_section = sections_pairs[0].split('-')
           section1 = set(range(int(first_section[0]), int(first_section[1])+1))
           
           second_section = sections_pairs[1].split('-')
           section2 = set(range(int(second_section[0]), int(second_section[1])+1))
          
           if len(section1.intersection(section2)) > 0:
               result+=1    

    AssertExpectedResult(878, result)
    return result

#endregion


#region ##### Day 5 #####

#    [C]             [L]         [T]
#    [V] [R] [M]     [T]         [B]
#    [F] [G] [H] [Q] [Q]         [H]
#    [W] [L] [P] [V] [M] [V]     [F]
#    [P] [C] [W] [S] [Z] [B] [S] [P]
#[G] [R] [M] [B] [F] [J] [S] [Z] [D]
#[J] [L] [P] [F] [C] [H] [F] [J] [C]
#[Z] [Q] [F] [L] [G] [W] [H] [F] [M]
# 1   2   3   4   5   6   7   8   9 

def print_top_crates(stacks):
    result = ''
    for i in range(1, len(stacks.keys())+1):
        s = stacks[i]
        if len(s) > 0:
            result += s.pop()
    return result

#Day 5, part 1: WSFTMRHPP (0.185 secs)
#Day 5, part 2: GSLCMFBRP (0.042 secs)
''' 
    Unable to parse the stacks at 5am ಥ_ಥ so I gave up and hardcoded the input 
    (it was reasonable to do, the size is small)
'''
def day5_1_original(data):
    #data = read_input(2022, "05t")    
    stack1 = ['Z','J','G']
    stack2 = ['Q','L','R','P','W','F','V','C']
    stack3 = ['F','P','M','C','L','G','R']
    stack4 = ['L','F','B','W','P','H','M']
    stack5 = ['G','C','F','S','V','Q']
    stack6 = ['W','H','J','Z','M','Q','T','L']
    stack7 = ['H','F','S','B','V']
    stack8 = ['F','J','Z','S']
    stack9 = ['M','C','D','P','F','H','B','T']
    
    stacks = {1: stack1, 2: stack2, 3: stack3, 4: stack4, 5: stack5, 6: stack6, 7: stack7, 8: stack8, 9: stack9}
    result = ''
    for line in data:
        if line:
           crates = [parse("move {} from {} to {}", line)]
          
           if crates[0] != None:
                for c in crates:
                    num = int(c[0])
                    from_crate = int(c[1])
                    to_crate = int(c[2])
                    from_stack = stacks[from_crate]
                    for i in range(num):
                        crate = from_stack.pop()
                        stacks[to_crate].append(crate)
    
    result = print_top_crates(stacks)            
    print(result)

    AssertExpectedResult('WSFTMRHPP', result)
    return result

def day5_2(data):
    #data = read_input(2022, "05t")    
    stack1 = ['Z','J','G']
    stack2 = ['Q','L','R','P','W','F','V','C']
    stack3 = ['F','P','M','C','L','G','R']
    stack4 = ['L','F','B','W','P','H','M']
    stack5 = ['G','C','F','S','V','Q']
    stack6 = ['W','H','J','Z','M','Q','T','L']
    stack7 = ['H','F','S','B','V']
    stack8 = ['F','J','Z','S']
    stack9 = ['M','C','D','P','F','H','B','T']
    
    stacks = {1: stack1, 2: stack2, 3: stack3, 4: stack4, 5: stack5, 6: stack6, 7: stack7, 8: stack8, 9: stack9}
    result = ''
    
    #stacks = {1: ['Z','N'], 2: ['M','C','D'], 3: ['P']}
    
    for line in data:
        if line:
           crates = [parse("move {} from {} to {}", line)]

           if crates[0] != None:
                for c in crates:
                    num = int(c[0])
                    from_crate = int(c[1])
                    to_crate = int(c[2])
                    from_stack = stacks[from_crate]
                    moving_crates = []
                    for i in range(num):
                        crate = from_stack.pop()
                        moving_crates.append(crate)
                    moving_crates.reverse()
                    stacks[to_crate] = stacks[to_crate] + moving_crates
                    
    result = print_top_crates(stacks)
    print(result)
    AssertExpectedResult('GSLCMFBRP', result)
    return result

def parse_stacks(line, stacks):
    for stack, crate in enumerate(line[1::4]):
        if crate.isdigit():
            break
        if crate == ' ':
            continue
        
        stack+=1
        if stack not in stacks:
            stacks[stack] = []
        stacks[stack].insert(0,crate)
                
    return stacks
               

def day5_1(data):
    #data = read_input(2022, "05t")    
    result = ''   
    
    stacks = {}
    
    for line in data:
        if line:
           crates = [parse("move {} from {} to {}", line)]
           
           if crates[0] == None:
               stacks = parse_stacks(line, stacks)
           if crates[0] != None:

                for c in crates:
                    num = int(c[0])
                    from_crate = int(c[1])
                    to_crate = int(c[2])
                    from_stack = stacks[from_crate]
                    moving_crates = []
                    for i in range(num):
                        crate = from_stack.pop()
                        stacks[to_crate].append(crate)
                    
    result = print_top_crates(stacks)
    print(result)
    AssertExpectedResult('WSFTMRHPP', result)
    return result

#endregion


#region ##### Day 6 #####

def day6_1(data):
    #data = read_input(2022, "06t")    
    
    result = 0
    for line in data:
        if line:
           sections_pairs = line.split(",")
           
    AssertExpectedResult(0, result)
    return result
#endregion

if __name__ == "__main__":
    # override timeout
    main(sys.argv, globals(), AOC_EDITION_YEAR, 900)

