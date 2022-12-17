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

#Day 6, part 1: 1909 (0.141 secs)
#Day 6, part 2: 3380 (0.030 secs)
def day6_1(data):
    #data = read_input(2022, "06t")    
    
    result = 0
    datastream = data[0]
    
    for i in range(len(datastream)):
        packet = datastream[i:i+4]
        c = Counter(packet)
        
        if len(c) == 4:
            result = i+4
            print(packet, i+4)
            break    
           
    AssertExpectedResult(1909, result)
    return result

def day6_2(data):
    #data = read_input(2022, "06t")    
    
    result = 0
    datastream = data[0]
    
    for i in range(len(datastream)):
        packet = datastream[i:i+14]
        c = Counter(packet)
        
        if len(c) == 14:
            result = i+14
            print(packet, i+14)
            break    
           
    AssertExpectedResult(3380, result)
    return result
#endregion


#region ##### Day 7 #####

def print_filesystem_data(filesystem, dir_sizes= None):
    print('filesystem:')
    for d in filesystem.keys():
        print(d, filesystem[d])
    print('****')
    print()


# this should work but it's bugged :(
def compute_dir_sizes(filesystem):
        
    dir_sizes = {} 
    for directory, contents in filesystem.items():
        total_size = 0
        for filename, size in contents:        
            total_size += size
            
        dir_sizes[directory] = total_size
        
    #print(dir_sizes)
    directories = sorted(list(filesystem.keys()), key=len, reverse=True)
    #print("directories ordered:",directories)
    #print()
    for directory in directories:
        if directory == '/':
            continue
        
        #print("dir",directory)
        #print(directory.split('/')[-2])
        
        dir_to_update = directory.split('/')[-1]
        parent_dir = directory.split('/')[-2]
        
        #print("dir_to_update",dir_to_update)
        if parent_dir == '': #update root
            parent_dir = '/'
        else:
            i = directory.rfind('/')
            parent_dir = directory[:i]
        #print("parent_dir:", parent_dir)
        
        value = [v for d,v in filesystem[parent_dir] if d == dir_to_update][0]
        #print("value", value)
        filesystem[parent_dir].remove((dir_to_update, value))
        filesystem[parent_dir].append((dir_to_update, value + dir_sizes[directory]))
        #dir_sizes[parent_dir]+= value
        
        dir_sizes = {} 
        for directory, contents in filesystem.items():
            total_size = 0
            for filename, size in contents:        
                total_size += size
            
            dir_sizes[directory] = total_size        
        
        #print("updated", parent_dir,'with', dir_sizes[directory]+value, 'for entry', dir_to_update)
        #print()
    

    #print(dir_sizes)
    return dir_sizes

#horrible first approach that must be buggy AF, worked for part1
def compute_dir_sizes2(filesystem):
    dir_sizes = {} 
    for directory, contents in filesystem.items():
        total_size = 0
        for filename, size in contents:        
            total_size += size
            
        dir_sizes[directory] = total_size
        
    repeat_computation = True
    
    while (repeat_computation):
        repeat_computation = False
        for directory, contents in filesystem.items():
            total_size = dir_sizes[directory]
            path = [directory]
            for filename, size in contents:        
                if size == 0:   # means there is a subdir whose size we need to compute             
                    #print("size for?", get_path(path, filename))
                    total_size += dir_sizes[get_path(path, filename)]
                    
                    missing_dir = sum([1 for f,s in filesystem[get_path(path, filename)] if s == 0])
                    if missing_dir == 0: #means that it is safe to replace the parent dir size
                                         #in filesystem with the computed value above
                        filesystem[directory].remove((filename,0))
                        filesystem[directory].append((filename, dir_sizes[get_path(path, filename)]))               
                    repeat_computation = True
                
            dir_sizes[directory] = total_size
    
    dir_sizes = {} 
    for directory, contents in filesystem.items():
        total_size = 0
        for filename, size in contents:        
            total_size += size
            
        dir_sizes[directory] = total_size
        
    return dir_sizes
                        

def get_path(path, directory = None):
    if directory == '/':
        directory =''
        
    if path == ['/']:
        if directory != None:
            return '/' + directory
        else:
            return '/'
    else: 
        full_path = '/'.join(path)
        if len(full_path) > 1 and full_path[0] == full_path[1]:
            full_path = full_path[1:]
        if directory == None:
            return full_path    
        else:
            return full_path + '/' + directory

# this is shit parsing
def parse_filesystem(data):
    current_dir = '/'
    filesystem = {'/': []}
    path = [current_dir]
    for line in data:
        if line.startswith("$"):
            command_line = line.split(" ")
            command = command_line[1]
            #print()
            #print_filesystem_data(filesystem)
            if command == 'ls':    
                size = 0
            if command == 'cd':                    
                target_dir = command_line[2]
                
                if target_dir == '..': # cd ..                        
                    #print("cd .. with current dir", current_dir, 'with path', get_path(path))
                    for directory in filesystem.keys():
                        
                        #print("checking if path", directory,"is in filesystem" )
                        if directory not in filesystem.keys():
                            continue
                        
                        #print("checking...",current_dir,"inside path", directory,":",filesystem[directory])
                        
                        if current_dir in [d for d,s in filesystem[directory]]:
                            path.pop()
                            target_dir = path[-1]
                            current_dir = target_dir
                            
                            #print("changed to",get_path(path))                          
                            
                            break
                    
                    if current_dir != '/':
                        current_dir = target_dir
                
                elif target_dir == '/': # cd /
                    #print("cd /")
                    current_dir = '/'
                    path = ['/']
                
                else: # cd new_dir
                    #print("cd", target_dir)
                    #print("current_dir", current_dir, "path:", path)
                    #print("current_path", get_path(path),":",filesystem[get_path(path)])
                    
                    if target_dir in [d for d,s in filesystem[get_path(path)]]:
                        current_dir = target_dir
                        path.append(target_dir)
                        #print("changed to", current_dir, 'new path:', get_path(path))
                    else:
                        print("OOPS")
                        
                
        else: #listing contents of current dir
            #print("ls",current_dir)
            dir_contents = line.split(" ")
            
            if dir_contents[0] == 'dir':
                dir_name = dir_contents[1]
                #print("checking content:", dir_name)
                #print('is',get_path(path, dir_name),'in filesystem', filesystem.keys(),'?')
                if get_path(path, dir_name) not in filesystem.keys():    
                    filesystem[get_path(path, dir_name)] = []
                    #print("added",get_path(path, dir_name),'to filesystem',filesystem.keys())
                
                #print("checking if", dir_name,'is in current directory', get_path(path),':',filesystem[get_path(path)])
                if dir_name not in filesystem[get_path(path)]:
                    filesystem[get_path(path)].append((dir_name, 0))                
                    #print("added",dir_name,'to current directory', get_path(path))
            else:
                filesize = int(dir_contents[0])
                filename = dir_contents[1] 
                #print("listing file", filename)                               
                filesystem[get_path(path)].append((filename, filesize))
                #print("added",filename,'to', get_path(path))
                
    return filesystem


#1259494
#836079
#978424
#Day 7, part 1: 1348005 (0.089 secs)
#Day 7, part 2: 12785886 (0.021 secs)
def day7_1(data):
    #data = read_input(2022, "07t")    
    
    limit = 100000
    result = 0
    filesystem = parse_filesystem(data)    
    #print_filesystem_data(filesystem) 
    dir_sizes = compute_dir_sizes(filesystem)   
    #print(dir_sizes)
    
    for d in dir_sizes.keys():
        if dir_sizes[d] < limit:
            result += dir_sizes[d]     
            
    #print_filesystem_data(filesystem) 
    
    print(result)
    
    AssertExpectedResult(1348005, result)
    return result

def day7_2(data):
    #data = read_input(2022, "07t")  
    
    disk_space = 70000000
    min_unused_space_needed = 30000000        
    
    result = 0
    filesystem = parse_filesystem(data)
    
    #print_filesystem_data(filesystem) 

    dir_sizes = compute_dir_sizes(filesystem)   
    #print_filesystem_data(filesystem)
        
    current_unused_space = disk_space - dir_sizes['/']
    print("current unused space:",current_unused_space)
    
    need_to_free = min_unused_space_needed - current_unused_space
    print("need to free",need_to_free)
    
    min_dir = (disk_space, '.')
    for d in dir_sizes.keys():
        size = dir_sizes[d]
        if size >= need_to_free:
            if size < min_dir[0]:
                min_dir = (size,d)          
     
            
    #print_filesystem_data(filesystem) 
    
    print(min_dir)
    result = min_dir[0]
    AssertExpectedResult(12785886, result)
    return result

#endregion

#region ##### Day 8 #####

def count_visible_trees(trees_map):
    rows = len(trees_map)
    columns = len(trees_map[0])    
    visible = 0  
    
    for y in range(rows):
        for x in range(columns):
            if x == 0 or x == (columns-1) or y == 0 or y == (rows-1):
                visible += 1 
            else:
                tree_size = trees_map[y][x]
                #print("testing",y,x,"with height",tree_size)
                taller_trees_horizontal = 0
                
                for xx in range(0, x):                    
                    size = trees_map[y][xx]
                    
                    if size >= tree_size and xx != x:                       
                        taller_trees_horizontal +=1
                        break
                
                for xx in range(x+1, columns):                    
                    size = trees_map[y][xx]
                    
                    if size >= tree_size and xx != x:                       
                        taller_trees_horizontal +=1  
                        break                       
                        
                #print("horizontal for",y,x,"is:", taller_trees_horizontal)
                
                taller_trees_vertical = 0
                
                for yy in range(0, y):                    
                    size = trees_map[yy][x]
                    if size >= tree_size and yy != y:                       
                        taller_trees_vertical +=1 
                        break
                
                for yy in range(y+1,rows):                    
                    size = trees_map[yy][x]
                    if size >= tree_size and yy != y:                       
                        taller_trees_vertical +=1 
                        break
                        
                #print("vertical for",y,x,"is:", taller_trees_horizontal)
                #print()
                if (taller_trees_horizontal + taller_trees_vertical) <4:
                    visible +=1

    return visible

#Day 8, part 1: 1835 (0.109 secs)
#Day 8, part 2: 263670 (0.050 secs)
def day8_1(data):
    #data = read_input(2022, "08t")    
    trees_map = []
    
    result = 0     
    trees_map = buildMapGrid(data, initValue=0)       
    
    y = 0
    for line in data:
        for x in range(len(line)):
            trees_map[y][x] = int(line[x])
        y += 1
    
    result = count_visible_trees(trees_map)            
    #printMap(trees_map)
           
    AssertExpectedResult(1835, result)
    return result


def get_highest_scenic_score(trees_map):
    rows = len(trees_map)
    columns = len(trees_map[0])
    
    highest_scenic_score = 0
    
    for y in range(rows):
        for x in range(columns):
            tree_size = trees_map[y][x]
            #print("testing (",y,x,") with height ->",tree_size,'<-')
            #print()
            left_visible_trees = 0
            
            #print("left trees:")
            left_trees = list(range(0, x))
            left_trees.reverse()
            for xx in left_trees:                    
                size = trees_map[y][xx]
                #print("checking",y,xx,'with size',size)    
                left_visible_trees +=1                      
                if size >= tree_size: 
                    break                                     
                
            #print("left for",y,x,"is:", left_visible_trees)
            #print()
            scenic_score = left_visible_trees  
            
            right_visible_trees = 0
            #print("right trees:")
            for xx in range(x+1, columns):                    
                size = trees_map[y][xx]
                #print("checking",y,xx,'with size',size) 
                right_visible_trees +=1                     
                if size >= tree_size:  
                    break                   
                
            scenic_score *= right_visible_trees                             
            #print("right for",y,x,"is:", right_visible_trees)
            #print()
            up_visible_trees = 0
            
            up_trees = list(range(0, y))
            up_trees.reverse()
            #print("up trees:")                
            for yy in up_trees:                    
                size = trees_map[yy][x]
                #print("checking",yy,x,'with size',size) 
                up_visible_trees +=1
                if size >= tree_size:
                    break                
                
            #print("up for",y,x,"is:", up_visible_trees)    
            scenic_score *= up_visible_trees
            #print()
            down_visible_trees = 0
              
            #print("down trees:")  
            for yy in range(y+1,rows):                    
                size = trees_map[yy][x]
                #print("checking",yy,x,'with size',size) 
                down_visible_trees +=1 
                if size >= tree_size:                                          
                    break
                
            #print("down for",y,x,"is:", down_visible_trees)
            #print()
            scenic_score *= down_visible_trees   
        
            #print("scenic score", scenic_score)
            #print("highest scenic score", highest_scenic_score)
            #print() 
            #print('********')  
            if scenic_score > 0 and scenic_score > highest_scenic_score:
                highest_scenic_score = scenic_score 
            scenic_score = 1                     
               
    return highest_scenic_score


#2500608 too high
def day8_2(data):
    #data = read_input(2022, "08t")       
   
    trees_map = buildMapGrid(data, initValue=0)       
    
    y = 0
    for line in data:
        for x in range(len(line)):
            trees_map[y][x] = int(line[x])
        y += 1

    result = get_highest_scenic_score(trees_map)          
           
    AssertExpectedResult(263670, result)
    return result

#endregion


#region ##### Day 9 #####

def check_tail_position(head, tails):

    for k,v in tails.items():
        if k != 1:
            head = tails[k-1]
        tail = v
        
        new_x = tail[0]
        new_y = tail[1]
        
        head_x = head[0]
        tail_x = tail[0]
        
        head_y = head[1]
        tail_y = tail[1]
        
        dx = head_x -tail_x
        dy = head_y - tail_y
        #print('checking tail', tail,'against head',head,'. dx =',dx,'dy =',dy)
        
        if head == tail:
            return tail    
        
        if abs(head_x - tail_x) > 1:
            #print("adjusting x of tail", tail)
            if head_x - tail_x < 0:
                new_x = tail_x - 1
            elif head_x - tail_x > 0:
                new_x = tail_x + 1
            
            if head_y != tail_y:
                #print('adjusting y of tail', (new_x, new_y))
                if head_y - tail_y < 0:
                    new_y = tail_y - 1
                elif head_y - tail_y > 0:
                    new_y = tail_y + 1       
                #print("adjusted y of tail to", (new_x, new_y))
                
            #print("adjusted x of tail to", (new_x, new_y))             

            
        if abs(head_y - tail_y) > 1:
            #print("adjusting y of tail", tail)
            if head_y - tail_y < 0:
                new_y = tail_y - 1
            elif head_y - tail_y > 0:
                new_y = tail_y + 1
            #print("adjusted y of tail to", (new_x, new_y))
            
            if head_x != tail_x:
                #print('adjusting x of tail', (new_x, new_y))
                if head_x - tail_x < 0:
                    new_x = tail_x - 1
                elif head_x - tail_x > 0:
                    new_x = tail_x + 1      
                #print("adjusted x of tail to", (new_x, new_y))           
                 
        #print('updated tail #',k,'to', (new_x, new_y))
        tails[k] = (new_x, new_y)


def move_rope(moves, tails):
    head = (0,0)
    visited = {(0,0)}
    
    for move in moves:
        direction = move[0]
        pos = move[1]
        
        #print(move)
        #print("initial:",head,tails)
        #print()
        
        for i in range(pos):
            xx = head[0]
            yy =  head[1]
            
            if direction == 'U':
                yy +=1    
            elif direction == 'D':
                yy -=1
            elif direction == 'L':
                xx -=1
            elif direction == 'R':
                xx +=1
                
            old_head = head
            head = (xx, yy)
            check_tail_position(head, tails)
            
            tail = tails[len(tails)]
            #print("*** after move",old_head,'->',head,'***')
            #print('tails:',tails,tail)
            #print()

            visited.add(tail)
        #print()
        #print("completed move", move,"final tails:", tails)
    #print("tail 9 visited:",visited)
    return len(visited)

    
#Day 9, part 1: 6011 (0.118 secs)
#Day 9, part 2: 2419 (0.094 secs)
def day9_1(data):
    #data = read_input(2022, "09t")    
    
    result = 0 
    moves = []       
     
    for line in data:
        direction = line.split(' ')[0]
        times = int(line.split(' ')[1])
        moves.append((direction, times))
            
    tails = defaultdict()
    tails[1] = (0,0)
    result = move_rope(moves, tails)   
    
    AssertExpectedResult(6011, result)
    return result


# 6185 too high
# 2474 too high
# 2456 ?
# 1945 too low
# 1946
def day9_2(data):
    #data = read_input(2022, "09t")    
    
    result = 0 
    moves = []       
     
    for line in data:
        direction = line.split(' ')[0]
        times = int(line.split(' ')[1])
        moves.append((direction, times))

            
    tails = defaultdict()
    tails[1] = (0,0)    
    tails[2] = (0,0)
    tails[3] = (0,0)
    tails[4] = (0,0)
    tails[5] = (0,0)
    tails[6] = (0,0)
    tails[7] = (0,0)
    tails[8] = (0,0)
    tails[9] = (0,0)
    result = move_rope(moves, tails)   
    
    AssertExpectedResult(2419, result)
    return result

#endregion

#region ##### Day 10 #####

def get_signal_strenghts(instructions, times):
    cycle = 1
    register = 1
    signal_strenghts = []
    signal_strenght = 0
    
    cycles = [20,60,100,140,180,220]
    
    required_cycle = 0
    for instruction in instructions:
        
        if instruction[0] == 'noop':
            required_cycle = 1
        elif instruction[0] == 'addx':
            required_cycle = 2

        for i in range(required_cycle):           
                
            if cycle in cycles:
                print(cycle,cycle*register)
                #print(instruction[0],instruction[1], "register:",register)
                signal_strenghts.append(cycle*register)
            
            cycle += 1
            
        if instruction[0] == 'addx':
            register += instruction[1]
            
    return signal_strenghts

#Day 10, part 1: 16880 (0.077 secs)
#Day 10, part 2: 0 (0.006 secs)
def day10_1(data):
    #data = read_input(2022, "10t")       
    
    result = 0
    instructions = []
    for line in data:
        input = line.split(' ')

        if input[0] == 'addx':
            instructions.append((input[0], int(input[1])))
        else:
            instructions.append((input[0], ''))

    times = 6
    signal_strenghts = get_signal_strenghts(instructions, times)

    result = sum(signal_strenghts)
    #print(instructions)
           
    AssertExpectedResult(16880, result)
    return result


def draw_crt(crt, instructions):
    cycle = 1
    register = 1
    required_cycle = 0
    
    x = 0
    for instruction in instructions:
        
        if instruction[0] == 'noop':
            required_cycle = 1
        elif instruction[0] == 'addx':
            required_cycle = 2
            #register += instruction[1]

        for i in range(required_cycle):           
            x = x % 40
            y = (cycle//40)
            
            #print("cycle:",cycle,"position:", x, y, "register:",register)
            if x-1 <= register <= x+1:
                crt[y][x] = '#'
            
            cycle += 1
            x += 1
            
        if instruction[0] == 'addx':
            register += instruction[1]


import emoji
#https://unicode.org/emoji/charts/emoji-list.html
def day10_2(data):
    #data = read_input(2022, "10t")      

    result = 0
    instructions = []
    for line in data:
        input = line.split(' ')

        if input[0] == 'addx':
            instructions.append((input[0], int(input[1])))
        else:
            instructions.append((input[0], ''))

    rows = len(instructions)
    columns = 40    
    rows += sum([1 for l in instructions if l[0] == 'addx']) * 2
    rows = rows // columns

    crt = [ [ emoji.emojize('.') for i in range(columns) ] for j in range(rows) ]    
    draw_crt(crt,instructions)
    
    printGridsASCII(crt,'#')
           
    AssertExpectedResult('RKAZAJBR', result)
    return result

#endregion



#region ##### Day 11 #####

def play_monkey_turn(monkey, monkey_data, monkeys, part2):
    items = monkey_data[0]
    operation = monkey_data[1]
    test = monkey_data[2]
    inspected = monkeys[monkey][3] + len(items)
    
    #print('Monkey',monkey, 'with items', items)
    for worry_level in items:
        #print("Monkey inspects an item with a worry level of", worry_level)
        worry_level = operation(worry_level)
        #print('Worry level changes to', worry_level)       
        
        if not part2:
            worry_level = worry_level // 3
            #print('Monkey gets bored with item. Worry level is divided by 3 to', worry_level)
        else:
            #hardcoding was not a great idea :'(
            worry_level = worry_level % 9699690 
       
        throw_at_monkey = test(worry_level)
        #print('Item with worry level',worry_level,'is thrown to monkey', throw_at_monkey)
        
        monkeys[throw_at_monkey][0].append(worry_level)
        #print()
       
    #print('Updated monkey to', inspected)
    monkeys[monkey] = ([], operation, monkey_data[2], inspected)
    

def play_round(monkeys, part2 = False):
    for monkey, monkey_data in monkeys.items():
        play_monkey_turn(monkey, monkey_data, monkeys, part2)

#Day 11, part 1: 110264 (0.080 secs)
#Day 11, part 2: 23612457316 (0.132 secs)
def day11_1(data):
    data = read_input(2022, "11t")       
    
    monkeys_s, monkeys = get_monkeys_input()
            
    rounds = 20
    for r in range(rounds):
        play_round(monkeys)
        #print("Round",r)
        #for m,v in monkeys_s.items():
        #    print(m, v[0])
        #print()          
    
    inspected = []
    for m, v in monkeys.items():
        inspected.append(v[3])
    inspected.sort(reverse=True)
    monkey_business = inspected[0] * inspected[1]
    
    print(monkey_business)
           
    AssertExpectedResult(110264, monkey_business)
    return monkey_business


#2,3,5,7,11,13,17,19,23
#wasted effort on trying disibility rules algos for big numbers
def apply_divisibility_rule(number, divisor):
    if divisor == 2:
        #print(str(number)[-1])
        return (int(str(number)[-1]) % 2) == 0
        
    elif divisor == 3:
        list_of_integers = [int(digit) for digit in str(number)]       
        
        return (sum(list_of_integers) % 3) == 0
            
    elif divisor == 5:
        last_digit = int(str(number)[-1])
        
        return last_digit == 0 or last_digit == 5
    
    elif divisor == 7: #ok
        
        list_of_integers = [int(digit) for digit in str(number)]          
        triplets = []
        
        # Append required 0s at the beginning. 
        if (len(list_of_integers) % 3 == 1) : 
            list_of_integers.insert(0,0)
            list_of_integers.insert(0,0)      
        elif (len(list_of_integers) % 3 == 2) : 
            list_of_integers.insert(0,0)        
        
        i = 0
        while i < len(list_of_integers):
            n = int(str(list_of_integers[i]) + str(list_of_integers[i+1]) + str(list_of_integers[i+2]))
            triplets.append(n) 
            i += 3
        triplets.reverse()
       
        total = range(len(triplets))
        sum_subtracts = 0
        for i in total[1::2]:        
            #print(triplets[i-1],'-',triplets[i]  )
            sum_subtracts += triplets[i-1]-triplets[i]  
        
        return (sum_subtracts % 7) == 0
        
    elif divisor == 11: # ok
        list_of_integers = [int(digit) for digit in str(number)]       
        
        total = range(len(list_of_integers))
        sum_subtracts = 0
        for i in total[1::2]:
            print(list_of_integers[i-1],'-',list_of_integers[i])
            sum_subtracts += list_of_integers[i-1]-list_of_integers[i]  
        
        if len(list_of_integers) % 2 != 0:
            sum_subtracts += list_of_integers[-1]
        
        return (sum_subtracts % 11) == 0
    
    elif divisor == 13: #ok
        list_of_integers = [int(digit) for digit in str(number)]          
        triplets = []
        
        # Append required 0s at the beginning. 
        if (len(list_of_integers) % 3 == 1) : 
            list_of_integers.insert(0,0)
            list_of_integers.insert(0,0)      
        elif (len(list_of_integers) % 3 == 2) : 
            list_of_integers.insert(0,0)        
        
        i = 0          
        while i < len(list_of_integers):
            n = int(str(list_of_integers[i]) + str(list_of_integers[i+1]) + str(list_of_integers[i+2]))
            triplets.append(n) 
            i += 3
        triplets.reverse()
        
        sum_subtracts = 0
        while len(triplets) > 0:
            n1 = triplets.pop()
            n2 = triplets.pop()
            sum_subtracts += n1 - n2
            #print(n1,'-',n2)
            if len(triplets) == 1:
                n3 = triplets.pop()
                #print(n3,'- 0')
                sum_subtracts += n3
        
        return (sum_subtracts % 13) == 0
        
    elif divisor == 17:
        
        while(len(str(number)) > 5):
            last_digit = int(str(number)[-1]) * 7
            remainder = int(str(number)[:-1])
            
            number = remainder - last_digit
        
        return (number % 17) == 0
    
    elif divisor == 19:
        
        while(len(str(number)) > 5):
            last_digit = int(str(number)[-1]) * 2
            remainder = int(str(number)[:-1])
            
            number = remainder - last_digit
        
        return (number % 19) == 0
            
    elif divisor == 23:
        
        while(len(str(number)) > 5):
            last_digit = int(str(number)[-1]) * 7
            remainder = int(str(number)[:-1])
            
            number = remainder - last_digit
        
        return (number %23) == 0
           
    return False

def get_monkeys_input():
    monkeys_s = defaultdict()
    monkeys = defaultdict()
    
    # 96577
    monkeys_s[0] = ([79,98], lambda old: old*19, lambda worry: 2 if worry % 23 == 0 else 3, 0)    
    monkeys_s[1] = ([54,65,75,74], lambda old: old+6, lambda worry: 2 if worry % 19 == 0 else 0, 0)    
    monkeys_s[2] = ([79,60,97], lambda old: old*old, lambda worry: 1 if worry % 13 == 0 else 3, 0)    
    monkeys_s[3] = ([74], lambda old: old+3, lambda worry: 0 if worry % 17 == 0 else 1, 0)
    
    # 9699690
    monkeys[0] = ([65,78], lambda old: old*3, lambda worry: 2 if worry % 5 == 0 else 3, 0)
    monkeys[1] = ([54, 78, 86, 79, 73, 64, 85, 88], lambda old: old + 8, lambda worry: 4 if worry % 11 == 0 else 7, 0)
    monkeys[2] = ([69, 97, 77, 88, 87], lambda old: old + 2, lambda worry: 5 if worry % 2 == 0 else 3, 0)
    monkeys[3] = ([99], lambda old: old + 4, lambda worry: 1 if worry % 13 == 0 else 5, 0)
    monkeys[4] = ([60, 57, 52], lambda old: old * 19, lambda worry: 7 if worry % 7 == 0 else 6, 0)
    monkeys[5] = ([91, 82, 85, 73, 84, 53], lambda old: old + 5, lambda worry: 4 if worry % 3 == 0 else 1, 0)
    monkeys[6] = ([88, 74, 68, 56], lambda old: old * old, lambda worry: 0 if worry % 17 == 0 else 2, 0)
    monkeys[7] = ([54, 82, 72, 71, 53, 99, 67], lambda old: old + 1, lambda worry: 6 if worry % 19 == 0 else 0, 0)
    
    return monkeys_s, monkeys

# 32398560007 too high
# 21662625840 too low
def day11_2(data):
    data = read_input(2022, "11t")       
    
    monkeys_s, monkeys = get_monkeys_input()
            
    rounds = 10000
    for r in range(rounds):
        play_round(monkeys, part2=True)
       
        #if r+1 in [1,20,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]:
        #    print("Round",r+1)
        #    for m,v in monkeys.items():
        #        print(m, v[3])
        #    print()             
        
    inspected = []
    for m, v in monkeys.items():
        inspected.append(v[3])
    inspected.sort(reverse=True)
    monkey_business = inspected[0] * inspected[1]
    print(inspected)
    print(monkey_business)
    
    
    #1231243423434534643634545634565436354635636354635463452345235235325235235235
    #print(apply_divisibility_rule(2278  , 17))
           
    AssertExpectedResult(23612457316, monkey_business)
    return monkey_business

#endregion


#region ##### Day 12 #####

def build_graph_from_height_map(map):
    graph = {}
    sizeX = len(map[0])
    sizeY = len(map)

    for y in range(sizeY):
        for x in range(sizeX):

            east = (x+1, y)
            west = (x-1, y)
            north = (x, y-1)
            south = (x, y+1)
            
            neighbours = []
            
            if map[y][x] == 'S':
                current_pos = ord('a')
            elif map[y][x] == 'E':
                current_pos = ord('z')
            else:
                current_pos = ord(map[y][x])

            if map[y][x] != ' ':                
                if east[1] >= 0 and east[0] >= 0 and east[1] < sizeY and east[0] < sizeX and map[east[1]][east[0]] != ' ' and (ord(map[east[1]][east[0]]) - current_pos) <= 1:
                    neighbours.append(east)
                if west[1] >= 0 and west[0] >= 0 and west[1] < sizeY and west[0] < sizeX and map[west[1]][west[0]] != ' ' and (ord(map[west[1]][west[0]]) - current_pos) <= 1:
                    neighbours.append(west)
                if north[1] >= 0 and north[0] >= 0 and north[1] < sizeY and north[0] < sizeX and map[north[1]][north[0]] != ' ' and (ord(map[north[1]][north[0]]) - current_pos) <= 1:
                    neighbours.append(north)
                if south[1] >= 0 and south[0] >= 0 and south[1] < sizeY and south[0] < sizeX and map[south[1]][south[0]] != ' ' and (ord(map[south[1]][south[0]]) - current_pos) <= 1:
                    neighbours.append(south)
            
            graph[(x,y)] = neighbours
    return graph


# shamelessly taken from https://onestepcode.com/graph-shortest-path-python/
def shortest_path(graph, node1, node2):
    path_list = [[node1]]
    path_index = 0
    # To keep track of previously visited nodes
    previous_nodes = {node1}
    if node1 == node2:
        return path_list[0]
        
    while path_index < len(path_list):
        current_path = path_list[path_index]
        last_node = current_path[-1]
        next_nodes = graph[last_node]
        
        # Search goal node
        if node2 in next_nodes:
            current_path.append(node2)
            return current_path
        # Add new paths
        for next_node in next_nodes:
            if not next_node in previous_nodes:
                new_path = current_path[:]
                new_path.append(next_node)
                path_list.append(new_path)
                # To avoid backtracking
                previous_nodes.add(next_node)
        # Continue to next path in list
        path_index += 1
    # No path is found
    return []

#Day 12, part 1: 352 (0.110 secs)
#Day 12, part 2: 345 (0.494 secs)
def day12_1(data):
    #data = read_input(2022, "12t")       
    
    result = 0  
    y = 0
    rows = len(data)
    columns = len(data[0])  
    grid = [ [ ' ' for i in range(columns) ] for j in range(rows) ]  
    initial_position = (0,0)
    final_position = (0,0)
    
    for line in data:
        x=0
        for n in line:
            grid[y][x] = n
            if n == 'S':
                grid[y][x] = 'a'
                initial_position = (x,y)
            if n == 'E':
                grid[y][x] = 'z'
                final_position = (x,y)
            x+=1
        y+=1
        
    #printMap(grid)    
    g = build_graph_from_height_map(grid)    
    #printGraph(g)
    path = shortest_path(g,initial_position,final_position)
    #print(path)
    result = len(path)-1
    
    AssertExpectedResult(352, result)
    return result


def day12_2(data):
    #data = read_input(2022, "12t")       
    
    result = 0  
    y = 0
    rows = len(data)
    columns = len(data[0])  
    grid = [ [ ' ' for i in range(columns) ] for j in range(rows) ]  
    initial_position = (0,0)
    final_position = (0,0)
    a_positions = []
    
    for line in data:
        x=0
        for n in line:
            if n == 'a':
                a_positions.append((x,y))
            grid[y][x] = n
            if n == 'S':
                grid[y][x] = 'a'
                initial_position = (x,y)
            if n == 'E':
                grid[y][x] = 'z'
                final_position = (x,y)
            x+=1
        y+=1
        
    #printMap(grid)    
    g = build_graph_from_height_map(grid)    
    
    result = sys.maxsize
    while len(a_positions) > 0:
        initial_position = a_positions.pop()    
        path = shortest_path(g,initial_position,final_position)
        path_length = len(path)-1
        if path_length > 0 and path_length < result:
            result = path_length
    
    AssertExpectedResult(345, result)
    return result


#endregion


#region ##### Day 13 #####


def is_packet_pair_in_right_order(left_packet, right_packet):
    
    #print("Left packet",left_packet)
    #print("Right packet",right_packet)
    
    if type(left_packet) is list and type(right_packet) is int:
        #print("Converting right packet into list")
        return is_packet_pair_in_right_order(left_packet, [right_packet])
    
    if type(left_packet) is int and type(right_packet) is list:
        #print("Converting left packet into list")
        return is_packet_pair_in_right_order([left_packet], right_packet)
    
    if type(left_packet) is int and type(right_packet) is int:
        #print("Packets are integers:", left_packet, right_packet)
        if left_packet != right_packet:
            #print("Comparing integers!")
            return left_packet < right_packet
        else:
            return None
    
    if type(left_packet) is list and type(right_packet) is list:
        #print("Packets are lists of sizes",len(left_packet), len(right_packet))
        
        if left_packet == []:
            return True
        if right_packet == []:
            return False
        
        left_elem = left_packet.pop(0)
        right_elem = right_packet.pop(0)

        if right_elem == left_elem == []:
            result = None
        else:
            result = is_packet_pair_in_right_order(left_elem, right_elem)
           
        #print("result is", result)
        if result != None:
            return result
        else:
            if left_packet != [] or right_packet != []:
                return is_packet_pair_in_right_order(left_packet, right_packet)
        
    
# 2087
# 5913
#Day 13, part 1: 5580 (0.076 secs)
#Day 13, part 2: 26200 (1.557 secs)
def day13_1(data):
    #data = read_input(2022, "13t")           
    result = 0
    left_packet = None
    right_packet = None
    index = 1
    right_ordered_pairs = []
    for line in data:
        if line:
            if left_packet == None:
                left_packet = eval(line.split("\n")[0])
            elif right_packet == None:
                right_packet = eval(line.split("\n")[0])
                #print("Pair",index,":",left_packet,'vs',right_packet)
                result = is_packet_pair_in_right_order(left_packet, right_packet)
                #print("result of pair comparison is", result)
                if result == True:
                    right_ordered_pairs.append(index)
                #print()
                left_packet = None
                right_packet = None
                index+=1
    #print(right_ordered_pairs)
    result = sum(right_ordered_pairs)
           
    AssertExpectedResult(5580, result)
    return result

# this is inneficient, takes 110s
def bubble_sort(array):
    n = len(array)

    for i in range(n):
        already_sorted = True

        for j in range(n - i - 1):
            c_array = copy.deepcopy(array)            
            res = is_packet_pair_in_right_order(c_array[j+1], c_array[j])
            if res:
                array[j], array[j + 1] = array[j + 1], array[j]
                already_sorted = False

        if already_sorted:
            break

    return array


def merge(left, right):
    if len(left) == 0:
        return right

    if len(right) == 0:
        return left

    result = []
    index_left = index_right = 0

    while len(result) < len(left) + len(right):

        c_left = copy.deepcopy(left) 
        c_right = copy.deepcopy(right) 
        res = is_packet_pair_in_right_order(c_left[index_left], c_right[index_right])
        
        if res: 
            result.append(left[index_left])
            index_left += 1
        else:
            result.append(right[index_right])
            index_right += 1

        if index_right == len(right):
            result += left[index_left:]
            break

        if index_left == len(left):
            result += right[index_right:]
            break

    return result

def merge_sort(array):
    if len(array) < 2:
        return array

    midpoint = len(array) // 2

    return merge(
        left=merge_sort(array[:midpoint]),
        right=merge_sort(array[midpoint:]))


def day13_2(data):
    #data = read_input(2022, "13t")           
    result = 0
    left_packet = None
    right_packet = None
    index = 1
    divider_packets = [[[2]],[[6]]]
    packet_pairs = [[[2]],[[6]]]
    for line in data:
        if line:
            if left_packet == None:
                left_packet = eval(line.split("\n")[0])
            elif right_packet == None:
                right_packet = eval(line.split("\n")[0])
                packet_pairs.append(left_packet)
                packet_pairs.append(right_packet)

                left_packet = None
                right_packet = None
                index+=1
                
    packet_pairs = merge_sort(packet_pairs)
    i = 1
    result = 1
    for packet in packet_pairs:
        if packet in divider_packets:
            result *= i
        i+=1
    print(result)   
           
    AssertExpectedResult(26200, result)
    return result

#endregion



#region ##### Day 14 #####

def generate_cave_system(rock_segments, low_x, high_x, low_y, high_y, part2 = False):
    delta = 500
    
    adjust_x = high_x - delta//2
    adjust_y = high_y 
    adjust_delta = min(adjust_x, adjust_y) 
    
    rows = low_y - high_y + delta 
    columns = high_x - low_x + delta
    cave = [ [ '.' for i in range(columns) ] for j in range(rows) ]      
    
    sand = (500-adjust_x, 0+adjust_y)
    
    lowest_y = 0
    for rock_segment in rock_segments:
        #print(rock_segment)
        start = rock_segment[0]
        end = rock_segment[1]
        xi = min(start[0], end[0])
        xf = max(start[0], end[0])
        
        for x in range(xi, xf+1):
            #print("converting x:", x, "into", x - adjust_x)
            
            x -= adjust_x
            
            
            yi = min(start[1], end[1])
            yf = max(start[1], end[1])
            for y in range(yi, yf+1):  
               # print("converting y:", y, "into", y -adjust_y )
                y+= adjust_y         
                
                cave[y][x] = '#'
                if y > lowest_y:
                    lowest_y = y
    if part2:
        y= lowest_y+2
        for x in range(0, columns):   
            cave[y][x] = '#' 
    
    cave[sand[1]][sand[0]] = '+'
    #printMap(cave)
    return cave, sand, lowest_y

def check_grain_at_rest(cave, lowest_rock, x, y, part2 = False):
    SAND = 'o'
    AIR = '.'
    ROCK = '#'
    obstacles = [SAND, ROCK]
    directions = {"down": (0,1), "left": (-1,1), "right": (1,1)}
    
    dx = directions["down"][0]
    dy = directions["down"][1]
    
    #print("sand FALLING DOWN to position", x, y, "with",cave[y][x], part2)
    if not part2:
        if y+dy > lowest_rock :
            #print("sand fell into ABYSS!", y+dy,'>', lowest_rock)
            return cave, True, True, x, y  
    
    #     .
    #   #?#?#
    if cave[y+dy][x+dx] in obstacles: #hit a rock or sand grain
        #print("hit an obstacle", cave[y+dy][x+dx],"at", x+dx,y+dy, "switching to LEFT")
        dx = directions["left"][0]
        dy = directions["left"][1]                
        #     .
        #  ##.# ###
        if cave[y+dy][x+dx] == AIR: # can cascade left
            return check_grain_at_rest(cave, lowest_rock, x+dx, y+dy, part2)
        else:
            #print("hit an obstacle", cave[y+dy][x+dx],"at", x+dx,y+dy, "switching to RIGHT")
            dx = directions["right"][0]
            dy = directions["right"][1]                
            #     .
            #  ## #. ###
            if cave[y+dy][x+dx] == AIR: # can cascade right
                return check_grain_at_rest(cave, lowest_rock, x+dx, y+dy, part2)
            else: 
                if cave[y][x] in obstacles:
                    #print("hit an obstacle", cave[y][x],"at", x,y, "rebooting!")
                    return cave, False, False, x, y
                else: #it sits on top
                    #print("sand dropped at",x,y,"!")
                    cave[y][x] = SAND
                    return cave, True, False, x, y
   
    return cave, False, False, x, y 

def drop_one_sand(cave, lowest_rock, sand, part2 = False):
    SAND = 'o'
    AIR = '.'
    ROCK = '#'
    sand_point = (sand[0], sand[1])
    x,y = (sand[0], sand[1])        

    while True: 
        dx = 0
        dy = 1         
        cave, is_resting, fell_into_abyss, x, y = check_grain_at_rest(cave, lowest_rock, x, y, part2)

        if not part2:
            if fell_into_abyss:
                break
        
        if not is_resting:
            #print("updating",x,y,"to",x+dx,y+dy)
            x+=dx 
            y+=dy
        else:
            break
 
        
    return cave, fell_into_abyss 

def drop_sand_until_it_flows_into_abyss(cave, lowest_rock, sand, part2=False):
    units = 0
    
    while True:
        #print("unit:", units)
        cave, fell_into_abyss = drop_one_sand(cave, lowest_rock, sand, part2)
        if not part2 and fell_into_abyss:
            break
        units+=1
        #printMap(cave)
        if part2 and cave[sand[1]][sand[0]] == 'o':
            break
        
    return units

def parse_cave(data):
    SAND = 'o'
    rock_segments = []
    lowest_rock = 0
    result = 0
    
    low_y = -1000
    high_y = 1000
    
    low_x = 1000    
    high_x = -1000
    for line in data:
        input = line.split(' -> ')
        start = None
        end = None
        for point in input:
            p = point.split(',')
            x = int(p[0])
            y = int(p[1])
            
            if y > low_y:
                low_y = y
            if y < high_y:
                high_y = y
                
            if x < low_x:
                low_x = x
            if x > high_x:
                high_x = x            
            
            if not start:
                start = (x,y)
            else:
                end = (x,y)
                rock_segments.append([start, end])
                start = end
                end = None
    return rock_segments, low_x, high_x, low_y, high_y

    
#Day 14, part 1: 618 (0.138 secs)
#Day 14, part 2: 26358 (0.255 secs)
def day14_1(data):
    #data = read_input(2022, "14t")       
    rock_segments, low_x, high_x, low_y, high_y = parse_cave(data)    
    cave, sand, lowest_rock = generate_cave_system(rock_segments, low_x, high_x, low_y, high_y)
    result = drop_sand_until_it_flows_into_abyss(cave, lowest_rock, sand)   
   
    AssertExpectedResult(618, result)
    return result

def day14_2(data):
    #data = read_input(2022, "14t")       
    
    rock_segments, low_x, high_x, low_y, high_y = parse_cave(data)    
    
    cave, sand, lowest_rock = generate_cave_system(rock_segments, low_x, high_x, low_y, high_y, part2=True)
    result = drop_sand_until_it_flows_into_abyss(cave, lowest_rock, sand, part2=True)   
   
    AssertExpectedResult(26358, result)
    return result


#enddregion


#region ##### Day 15 #####

def print_dict(d):

    for k,v in d.items():
        print(k,'=>')
        for e in v:
            print(e)
        print()
            
def manhattan_distance(p1, p2, three_dimensions = False):
    if three_dimensions:
        (x,y,z) = p1 
        (xx,yy,zz) = p2
        return abs(xx - x) + abs(yy - y) + abs(zz - z)
    else:
        (x,y) = p1 
        (xx,yy) = p2
        return abs(xx - x) + abs(yy - y)

#naive version
def compute_coverage(sensors, sensors_beacons, beacons_sensors,low_x, high_x, low_y, high_y):
    coverage_area = set()
    for sensor in sensors:
        distance = sensors_beacons[sensor][1]
        for y in range(low_y,high_y+1):
            for x in range(low_x, high_x+1):
                distance2 = manhattan_distance((x,y), sensor)
                if distance2 <= distance and (x,y) not in beacons_sensors.keys():
                    coverage_area.add((x,y))
    return coverage_area                    

def parse_beacons_sensors(data):
    result = 0
    sensors_beacons = defaultdict()
    
    for line in data:
        sensor_data = [parse("Sensor at x={}, y={}: closest beacon is at x={}, y={}", line)][0]
  
        sensor = (int(sensor_data[0]), int(sensor_data[1]))
        beacon = (int(sensor_data[2]), int(sensor_data[3]))
        distance = manhattan_distance(sensor, beacon)
        
        sensors_beacons[sensor] = (beacon, distance)
  
    return sensors_beacons

def compute_coverage_area_for_sensor(sensor, radius, target_y):
    coverage_area = set()
    c_x, c_y =  sensor
    reference = (c_x, target_y)    
    dist = manhattan_distance(sensor, reference)
    
    if dist > radius:
        return coverage_area
    
    delta = dist
    for x in range(c_x - radius + delta, c_x + radius - delta + 1):
        coverage_area.add((x, target_y))
       
    return coverage_area   

# 1995350 low
#Day 15, part 1: 5181556 (4.538 secs)
#Day 15, part 2: 12817603219131 (0.204 secs)
def day15_1(data):
    #data = read_input(2022, "15t")       
    
    result = 0
    sensors_beacons = parse_beacons_sensors(data)
    
    target_y = 2000000
    #target_y = 10
    coverage_area = set()
    
    
    all_coverage_area = set()
    for sensor in sensors_beacons.keys():
        radius = sensors_beacons[sensor][1]
        coverage_area = compute_coverage_area_for_sensor(sensor, radius, target_y)
        all_coverage_area = all_coverage_area.union(coverage_area)
    #print(all_coverage_area) 
    
    result = len(all_coverage_area)-1
    AssertExpectedResult(5181556, result)
    return result


def get_tuning_frequency(distress_beacon):
    x = distress_beacon[0]
    y = distress_beacon[1]
    return (x*4000000 + y)

from z3 import *

#encoding taken from https://stackoverflow.com/questions/22547988/how-to-calculate-absolute-value-in-z3-or-z3py
def z3_abs(x):
    return If(x >= 0,x,-x)

# 60000025 too low 
def day15_2(data):
    #data = read_input(2022, "15t")       
    
    result = 0
    sensors_beacons = parse_beacons_sensors(data)
    
    lower_bound = 0
    upper_bound = 4000000    
    
    solver = Solver()

    x = Int('x')
    y = Int('y')
    solver.add(x >= lower_bound, x <= upper_bound, y >= lower_bound, y <= upper_bound)
        
    for sensor in sensors_beacons.keys():
        radius = sensors_beacons[sensor][1]        
        c_x, c_y = sensor
        
        # for each sensor add equation that x,y must be outside their range
        solver.add( z3_abs(c_x - x) + z3_abs(c_y - y) > radius )
    
    try:    
        solver.check()
        model = solver.model()
        print(model)    
 
        distress_beacon = (model[x].as_long(), model[y].as_long())
        result = get_tuning_frequency(distress_beacon)
    except:
        print("No model found!")
    
    AssertExpectedResult(12817603219131, result)
    return result

#endregion



#region ##### Day 16 #####

def parse_tunel_system(data):
    result = 0
    tunnel_graph = defaultdict()
    
    for line in data:
        tunnel_data = [parse("Valve {} has flow rate={}; tunnels lead to valves {}", line)][0]
        if not tunnel_data:
            tunnel_data = [parse("Valve {} has flow rate={}; tunnel leads to valve {}", line)][0]
        #print(tunnel_data)
        origin = tunnel_data[0]
        flow_rate = int(tunnel_data[1])
        tunnels = list(tunnel_data[2].split(", "))            
        tunnel_graph[origin] = (flow_rate, tunnels)
  
    return tunnel_graph

def open_valve(flow_rate, minutes_left):
    return flow_rate*minutes_left


def walk_tunnels(tunnel_graph, current_position, open_valves, minutes_left, total_pressure_release):
    if minutes_left == 0:
        print("Run out of time!")
        return total_pressure_release, open_valves
    
    flow_rate, possible_tunnels = tunnel_graph[current_position]
    if current_position not in open_valves and flow_rate != 0:
        minutes_left -=1
        total_pressure_release += open_valve(flow_rate, minutes_left)
        print("Opened valve", current_position, "with added pressure",flow_rate*minutes_left)
        open_valves.append(current_position)
        
    
    max_pressure = 0
    for tunnel in possible_tunnels:
        print("Walking into tunnel", tunnel,"open valves are:",open_valves)
        if not tunnel in open_valves:
            pressure_released, open_valves = walk_tunnels(tunnel_graph, tunnel, open_valves, minutes_left-1, total_pressure_release)
            print("Pressure released with tunnel", tunnel, 'is', pressure_released)
        
            if pressure_released > max_pressure:
                max_pressure = pressure_released
            
    minutes_left -=1
    total_pressure_release += max_pressure
    
    return total_pressure_release, open_valves
        

def release_most_pressure(tunnel_graph):
    minutes_left = 30
    total_pressure_release = 0
    current_position = 'AA'
    
    paths = find_all_paths(current_position, lambda n: tunnel_graph[n][1], depth=minutes_left)
    for p in paths:
        print(p)
    print(paths)
    
    return 0

#from functools import lru_cache
#@hashable_lru
def find_all_paths(node, childrenFn, depth, _depth=0, _parents={}):
    if _depth == depth - 1:
        # path found with desired length, create path and stop traversing
        path = []
        parent = node
        for i in range(depth):
            path.insert(0, parent)
            if not parent in _parents:
                continue
            parent = _parents[parent]
            if parent in path:
                return # this path is cyclic, forget
        yield path
        return

    for nb in childrenFn(node):
        _parents[nb] = node # keep track of where we came from
        for p in find_all_paths(nb, childrenFn, depth, _depth + 1, _parents):
            yield p


def day16_1(data):
    data = read_input(2022, "16t")       
    
    result = 0
    tunnel_graph = parse_tunel_system(data)
    #print_dict(tunnel_graph)
    pressure_released = release_most_pressure(tunnel_graph)
    
    AssertExpectedResult(pressure_released, result)
    return result

#endregion



#region ##### Day 17 #####

# ['-', '+','L','|','S']  
'''
####  .#.   ..#   #   ##
      ###   ..#   #   ##
      .#.   ###   #
                  #
'''  
def drop_rock(cave, rock, jet_direction, current_height, occupied, jet_active):
    print("in",current_height)
    left_edge = 3
    lower_edge = len(cave) - (current_height + 3) - 2    
    h = lower_edge
    
    if jet_active:
        left_edge = apply_jet(cave, left_edge, lower_edge, jet_direction)
        
    if rock == '-':
        y = lower_edge
        x = left_edge
        for i in range(4):
            # it's falling down and hit an obstacle
            if not jet_active and rock_is_resting(cave, x, y):            
                #drop rock
                cave[y][x] = '#'
            x+=1
                   
        if not jet_active and rock_is_resting(cave, x, y):            
        #    if (x,y) not in occupied:            
        #        occupied.add((x,y))
            #drop rock
            current_height +=1
            return cave, current_height, True
    print(current_height)
    return cave, current_height, False

def rock_is_resting(cave, x, y):
    #print("is rock resting?")
    #print("***", cave[y+1][x])
    return cave[y+1][x] != '.'
    

def apply_jet(cave, left_edge, y, jet_direction):
    print("applying jet for",left_edge,"in direction", jet_direction)
    if jet_direction == '<':
        if cave[y][left_edge - 1] == '.':
            return left_edge - 1
    
    if jet_direction == '>':
        if cave[y][left_edge + 1] == '.':
            return left_edge + 1
        
def day17_1(data):
    data = read_input(2022, "17t")       
    width = 7
    max_rocks_fallen = 2022
    rock_types = ['-', '+','L','|','S']    
    
    rows = max_rocks_fallen * 4 
    columns = width + 2
    cave = [ [ '.' for i in range(columns) ] for j in range(rows) ] 
    floor = [ '-' for i in range(columns) ]
    floor[0] = '+'
    floor[8] = '+'
    cave.append(floor)
    for y in range(rows):
        cave[y][0] = '|'
        cave[y][columns-1] = '|'
            
    
    
    result = 0
    jet_pattern = data[0]
    
    rocks_fallen = 0
    current_rock_type = 0
    current_jet_direction = 0
    current_height = 0
    
    occupied = set()
    jet_active = False
    height = len(cave) - (current_height + 3) - 2 
    
    #debug
    max_rocks_fallen = 1
    while rocks_fallen != max_rocks_fallen:       
        rock = rock_types[current_rock_type]        
        
        print("dropping rock", rock)
        is_rested = False
        aux = current_height
        while not is_rested:
            jet_direction = jet_pattern[current_jet_direction]
            
            print('current height is', height)
            cave, res, is_rested = drop_rock(cave, rock, jet_direction, current_height, occupied, jet_active)
            print("res",res)
            current_height = res
            current_jet_direction += 1 
            current_jet_direction %= len(jet_pattern)
            jet_active = not jet_active
            if aux == current_height and not jet_active:
                print("something wrong")
                rocks_fallen = max_rocks_fallen
                break
                
        current_rock_type += 1
        current_rock_type %= 5
        
        if is_rested:
            print("rock dropped!")
            rocks_fallen +=1
        
        
    printMap(cave)
    

           
    AssertExpectedResult(0, result)
    return result

#endregion



'''
#region ##### Day 18 #####

def day18_1(data):
    data = read_input(2022, "18t")       
    
    result = 0
    for line in data:
        input = line.split(' ')

           
    AssertExpectedResult(0, result)
    return result

#endregion
'''


if __name__ == "__main__":
    # override timeout
    main(sys.argv, globals(), AOC_EDITION_YEAR, 900)

