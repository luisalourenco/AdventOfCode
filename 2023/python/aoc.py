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
from aocd import get_data
from aocd import submit
from shapely import *
from itertools import combinations
import networkx as nx
import graphviz
from queue import PriorityQueue

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
from common.mapUtils import printMap, buildMapGrid, buildGraphFromMap, buildGraphFromMap_v2, find_starting_point, build_empty_grid
from common.graphUtils import dijsktra, printGraph, find_all_paths, find_path, find_shortest_path, find_shortest_pathOptimal, bfs, dfs, Graph, hashable_lru, BFS_SP, hash_list, hashable_cache, Graph3
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
                
    print("checking if", part_number,"is part number with values (x,y):",x,y)
    
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


# y correcto, x seguinte
def check_gears(engine_map, gears, x, y, part_number):
    for gear in gears.keys():
        (xx,yy) = gear
        print(gear, part_number)
        if yy == y: # same line, must be on left or right
            left = list(part_number) == engine_map[y][x-len(part_number):x]
            right = list(part_number) == engine_map[y][x+1:x+len(part_number)+1]            
            if left or right:
                gears[gear].append(part_number)
        elif yy == y-1: #or yy == y+1:
            left = engine_map[y-1][x-len(part_number):x]
            right = engine_map[y-1][x+1:x+len(part_number)+1]
            print(left)
            print(right)
            if xx-1 <= x <= xx+1:
                gears[gear].append(part_number)

    print()
    return gears


def get_numbers_positions(engine_map):
    rows = len(engine_map)
    columns = len(engine_map[0])      
    part_number = ''
    numbers = {(-1,-1): []}

    uuid = 0
    for y in range(rows):   
        if part_number != '':
            number = int(part_number)
            #numbers[(x,y)] = number
            part_number = ''
            
        for x in range(columns):
            elem = engine_map[y][x]
            if elem.isdigit():
                part_number += elem
                numbers[(-1,-1)].append((x,y))
            elif part_number != '':                
                number = int(part_number)
                #numbers[(x,y)] = number
                for (xx,yy) in numbers[(-1,-1)]:
                    numbers[(xx,yy)] = (number, uuid)
                numbers[(-1,-1)] = []
                uuid += 1
                part_number = '' 
                               
    if part_number != '':
        number = int(part_number)
        #numbers[(x,y)] = number

    #print(numbers)         
    return numbers

def get_gears_ratios(engine_map, gears, numbers_positions):
    rows = len(engine_map)
    columns = len(engine_map[0]) 
    gears_numbers = dict()
    for gear in gears:
        gears_numbers[gear] = []

    res = set()
    ratio = 0
    for (x,y) in gears:

        if x != 0: #not first column
            n = numbers_positions.get((x-1,y))
            
            if n is not None:
                res.add(n)
        if y != 0: #not first row
            n = numbers_positions.get((x,y-1))
            if n is not None:
                res.add(n)
        if x != columns-1: #not last column
            n = numbers_positions.get((x+1,y))
            if n is not None:
                res.add(n)
        if y != rows-1: #not last row
            n = numbers_positions.get((x,y+1))
            if n is not None:
                res.add(n)

        if y != 0 and x != 0: #not first row nor first column
            n = numbers_positions.get((x-1,y-1))
            if n is not None:
                res.add(n)
        
        if y != rows-1 and x != columns-1: #not last row nor last column
            n = numbers_positions.get((x+1,y+1))
            if n is not None:
                res.add(n)

        if y != 0 and x != columns-1: #not first row nor last column
            n = numbers_positions.get((x+1,y-1))
            if n is not None:
                res.add(n)

        if y != rows-1 and x != 0: #not first row nor last column
            n = numbers_positions.get((x-1,y+1))
            if n is not None:
                res.add(n)

        gears_numbers[(x,y)] = res
        if len(res) == 2:
            res = list(res)
            ratio += res[0]*res[1]
            print(res)
        res = set()
    return ratio
    
def get_gears_ratios_v2(engine_map, gears, numbers_positions):
    rows = len(engine_map)
    columns = len(engine_map[0]) 
    gears_numbers = dict()
    for gear in gears:
        gears_numbers[gear] = []

    res = []
    uuids = set()
    ratio = 0
    for (x,y) in gears:

        val = numbers_positions.get((x-1,y))
            
        if val is not None:
            (n, uuid) = val
            if uuid not in uuids:
                res.append(n)
                uuids.add(uuid)
        
        val  = numbers_positions.get((x,y-1))
        if val is not None:
            (n, uuid) = val
            if uuid not in uuids:
                res.append(n)
                uuids.add(uuid)
        
        val = numbers_positions.get((x+1,y))
        if val is not None:
            (n, uuid) = val
            if uuid not in uuids:
                res.append(n)
                uuids.add(uuid)
        
        val = numbers_positions.get((x,y+1))
        if val is not None:
            (n, uuid) = val
            if uuid not in uuids:
                res.append(n)
                uuids.add(uuid)

        val = numbers_positions.get((x-1,y-1))
        if val is not None:
            (n, uuid) = val
            if uuid not in uuids:
                res.append(n)
                uuids.add(uuid)
        
        val = numbers_positions.get((x+1,y+1))
        if val is not None:
            (n, uuid) = val
            if uuid not in uuids:
                res.append(n)
                uuids.add(uuid)

        val = numbers_positions.get((x+1,y-1))
        if val is not None:
            (n, uuid) = val
            if uuid not in uuids:
                res.append(n)
                uuids.add(uuid)

        val = numbers_positions.get((x-1,y+1))
        if val is not None:
            (n, uuid) = val
            if uuid not in uuids:
                res.append(n)
                uuids.add(uuid)

        gears_numbers[(x,y)] = res
        if len(res) == 2:
            res = list(res)
            ratio += res[0]*res[1]
            #print("numbers for gear",x,y,"are",res)
       
       
        res = []
    return ratio


# 80364588 too high
# 79841031 too low
def day3_2(data):
    #data = read_input(2023, "03_teste")     
    result = 0    
    engine_map = buildMapGrid(data, initValue='.')       
    
    y = 0
    gears = {}
    for line in data:
        for x in range(len(line)):
            engine_map[y][x] = line[x]
            if line[x] == '*':
                gears[(x,y)] = []
        y += 1
    
    print(gears)
    #printMap(engine_map)
    numbers_positions = get_numbers_positions(engine_map)
    
    #print numbers positions
    '''
    rows = len(engine_map)
    columns = len(engine_map[0]) 
    for y in range(rows): 
        for x in range(columns):
            val = numbers_positions.get((x,y))
            if val is not None:
                (n, uuid) = val
                print("(",x,y,"):",n,"with uuid",uuid)
        print()
    #####    
    '''


    gears_ratio = get_gears_ratios_v2(engine_map, gears, numbers_positions) 
    result = gears_ratio
    print(gears_ratio)
    
    AssertExpectedResult(0, result)
    return result

#endregion

#region ##### Day 4 #####
#Day 4, part 1: 26443 (0.096 secs)
#Day 4, part 2: 6284877 (1.065 secs)
def day4_1(data):
    #data = read_input(2023, "04_teste")    
    result = 0    
    
    for line in data:
        points = 0
        ndata = parse("Card {}: {}", line)[1]
        numbers = ndata.split("|")
        winning_numbers = ints(numbers[0].split())
        played_numbers = ints(numbers[1].split() )
        
        for n in played_numbers:
            if n in winning_numbers:
                points = 1 if points == 0 else points*2
        result += points
    
    AssertExpectedResult(26443, result)
    return result

def day4_2(data):
    #data = read_input(2023, "04_teste")    
    result = 0    
    cards = {}

    for line in data:
        wins = 0
        ndata = parse("Card {}: {}", line)
        numbers = ndata[1].split("|")
        winning_numbers = ints(numbers[0].split())
        played_numbers = ints(numbers[1].split() )
        
        for n in played_numbers:
            if n in winning_numbers:
                wins+= 1
        cards[int(ndata[0])] = wins
    
    scratchcards = list(cards.keys())
    scratchcards.reverse()
    result = len(scratchcards)
    
    while (scratchcards):
        card = scratchcards.pop()
        wins = cards[card]
        result += wins
        for i in range(card+1, card + wins+1):
            scratchcards.append(i)

    
    AssertExpectedResult(6284877, result)
    return result

#endregion

#region ##### Day 5 #####

def get_map_used(mapIndex):
    # [seed_to_soil, soil_to_fertilizer, 
    # fertilizer_to_water, water_to_light, 
    # light_to_temperature, temperature_to_humidity, 
    # humidity_to_location]
    map_used = ""
    if mapIndex == 0:
        map_used = "seed to soil"
    elif mapIndex == 1:
        map_used = "soil to fertilizer"
    elif mapIndex == 2:
        map_used = "fertilizer to water"
    elif mapIndex == 3:
        map_used = "water to light"
    elif mapIndex == 4:
        map_used = "light to temperature"
    elif mapIndex == 5:
        map_used = "temperature to humidity"
    elif mapIndex == 6:
        map_used = "humidity to location"
    
    return map_used
    

def get_minimum_location_for_all_seeds(seeds, all_maps):
    locations = sys.maxsize

    mapIndex = 0
    for seed in seeds:
        #print("Seed:", seed)
        found = False
        for mapIndex in range(7):
            # (source_start, destination_start, ranges)
            mappings = all_maps[mapIndex]
            #print(get_map_used(seed, mapIndex))
            #print()
            for (source_start, destination_start, ranges) in mappings:                
                #print("source_start:", source_start)
                #print("destination_start:", destination_start)
                #print("range:", ranges)
                
                if source_start <= seed < source_start + ranges:
                    #print("found! moving to next map")
                    diff = seed - source_start
                    seed = destination_start + diff
                    found = True
                if found:
                    found = False
                    break
                #print()
            
            if not found:
                #print("did not find, using same value")
                seed = seed

            #print(map_used, seed)
            #print()
        if seed < locations:
            locations = seed
        #print()
        
    return locations

def parse_mappings(data):
    seed_to_soil = []
    soil_to_fertilizer = []
    fertilizer_to_water = []
    water_to_light = []
    light_to_temperature = []
    temperature_to_humidity = []
    humidity_to_location = []
    mapIndex = 0

    all_maps = [seed_to_soil, soil_to_fertilizer, fertilizer_to_water, water_to_light, light_to_temperature, temperature_to_humidity, humidity_to_location]

    for line in data:   
        line = line.strip()    
        if line != "":

            val = parse("seeds: {}", line)
            if val:
                seeds = ints(val[0].split())
            elif line == "seed-to-soil map:":
                mapIndex = 0
            elif line == "soil-to-fertilizer map:":
                mapIndex = 1
            elif line == "fertilizer-to-water map:":
                mapIndex = 2
            elif line == "water-to-light map:":
                mapIndex = 3
            elif line == "light-to-temperature map:":
                mapIndex = 4
            elif line == "temperature-to-humidity map:":
                mapIndex = 5
            elif line == "humidity-to-location map:":
                mapIndex = 6
            else:
                line_data = line.split(" ")
                destination_start = int(line_data[0])
                source_start = int(line_data[1])
                ranges = int(line_data[2])
                
                #print(get_map_used(mapIndex),":",source_start,"-",source_start+ranges-1," -> ", destination_start, destination_start+ranges-1)
                
                all_maps[mapIndex].append( (source_start, destination_start, ranges) )
    return (seeds, all_maps)

#Day 5, part 1: 240320250 (0.110 secs)
#Day 5, part 2: 28580589 (0.003 secs)
def day5_1(data):
    #data = read_input(2023, "05_teste")        

    (seeds, all_maps) = parse_mappings(data)    
    
    #for mapping in all_maps:
    #    print(mapping)
    result = get_minimum_location_for_all_seeds(seeds, all_maps)
  
    AssertExpectedResult(240320250, result)
    return result


def get_minimum_location_for_all_seeds_v2(seeds, all_maps):
    locations = sys.maxsize

    mapIndex = 0
    new_seeds = []
    for mapIndex in range(7):   
        
        if len(seeds) == 0:
            seeds = new_seeds
            new_seeds = []   
             
        while(len(seeds)> 0):
            (init, end) = seeds.pop()
            #print(get_map_used(mapIndex), init, end)  
            
            found = False            
            mappings = all_maps[mapIndex]
            #print("seed:(",init,",",end,")")
            # check each interval in a given map for each "seed" before moving to the next map
            for (source_start, destination_start, ranges) in mappings:                
                #print("source:", source_start, source_start + ranges-1)
                #print("destination", destination_start, destination_start + ranges-1)
                
                # x <= a < y and x <= b < y
                y = source_start + ranges
                x = source_start
                a = init
                b = end
                if x <= a < y and x <= b < y:
                    #print("found match! moving to next seed")
                    delta = destination_start - source_start
                    #print("delta:", delta)
                    init = init + delta
                    end = end + delta
                    #print("mapping to", init, end)
                    #print()
                    new_seeds.append((init,end))
                    found = True
                    if init < locations and mapIndex == 6:
                        locations = init
                    # a <= x < b and x < b < y
                elif a <= x < b and x < b < y:
                    found = True
                    seeds.append((a, x-1))
                    seeds.append((x, b))
                    #print("partial match on left, new seeds:",a,x-1,'and',x,b)
                elif x <= a < y and a < y < b:
                    found = True
                    seeds.append((a, y-1))
                    seeds.append((y, b))
                    #print("partial match on right, new seeds:",a,y-1,'and',y,b)
                    
                if found:
                    #found = False
                    break
            
            if not found:
                #print("did not find, using same value:",init,end)
                new_seeds.append((init,end))
                if init < locations and mapIndex == 6:
                    locations = init
            
            
    return locations


def day5_2(data):
    #data = read_input(2023, "05_teste")        

    (seeds_ranges, all_maps) = parse_mappings(data)
    seeds = []
    rr = []
    for i in range(0, len(seeds_ranges), 2):
        seed, r = seeds_ranges[i : i + 2]
        seeds.append((seed, seed+r-1))
    
    result = get_minimum_location_for_all_seeds_v2(seeds, all_maps)

    AssertExpectedResult(28580589, result)
    return result

#endregion

#region ##### Day 6 #####

def how_many_times_beats_record(time, distance):
    time_left = time
    total_distance = 0
    speed = 0
    wins = 0
    for i in range(1, time):
        speed = i
        time_left = time - i
        total_distance = speed * time_left

        if total_distance > distance:
            wins += 1

    return wins

#Day 6, part 1: 393120 (0.034 secs)
#Day 6, part 2: 36872656 (3.633 secs)
def day6_1(data):
    #data = read_input(2023, "06_teste")        

    times = ints(data[0].split(':')[1].split())
    distances = ints(data[1].split(':')[1].split())
    result =  1
    for i in range(len(times)):
        result *= how_many_times_beats_record(times[i], distances[i])
  
    AssertExpectedResult(393120, result)
    return result


def day6_2(data):
    #data = read_input(2023, "06_teste")        

    time = data[0].split(':')[1].split()
    time = int(''.join(time))
    distance = data[1].split(':')[1].split()
    distance = int(''.join(distance))

    result =  1
    result = how_many_times_beats_record(time, distance)
  
    AssertExpectedResult(36872656, result)
    return result

#endregion


#region ##### Day 7 #####

# ranges from 0 to 12
def get_label_strenght(label):
    labels = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    labels.reverse()
    return labels.index(label)

# ranges from 0 to 13
def get_label_strenght_v2(label):
    labels = ['A', 'K', 'Q', 'T', '9', '8', '7', '6', '5', '4', '3', '2', 'J']
    labels.reverse()
    return labels.index(label)

# all labels are equal
def is_five_of_a_kind(hand):
    cards = set(hand)
    return len(cards) == 1

def is_four_of_a_kind(hand):
    jokers = list(hand).count('J')
    cards = list(set(hand))
    if len(cards) == 2:
        return list(hand).count(cards[0]) == 4 or list(hand).count(cards[1]) == 4       
    else:
        return False

def is_full_house(hand):
    jokers = list(hand).count('J')
    cards = list(set(hand))
    if len(cards) == 2:
        return (list(hand).count(cards[0]) == 3 and list(hand).count(cards[1]) == 2) or (list(hand).count(cards[0]) == 2 and list(hand).count(cards[1]) == 3) 
    else:
        return False

def is_three_of_a_kind(hand):
    jokers = list(hand).count('J')
    cards = list(set(hand))
    if len(cards) == 3:
        return list(hand).count(cards[0]) == 3 or list(hand).count(cards[1]) == 3 or list(hand).count(cards[2]) == 3
    else:
        return False

def is_two_pair(hand):
    jokers = list(hand).count('J')
    cards = list(set(hand))
    if len(cards) == 3:
        return (list(hand).count(cards[0]) == 2 and list(hand).count(cards[1]) == 2 and list(hand).count(cards[2]) == 1) or (list(hand).count(cards[0]) == 2 and list(hand).count(cards[1]) == 1 and list(hand).count(cards[2]) == 2) or (list(hand).count(cards[0]) == 1 and list(hand).count(cards[1]) == 2 and list(hand).count(cards[2]) == 2)
    else:
        return False

def is_pair(hand):
    cards = list(set(hand))
    jokers = list(hand).count('J')
    if len(cards) == 4:
        return (list(hand).count(cards[0]) == 2 and list(hand).count(cards[1]) == 1 and list(hand).count(cards[2]) == 1 and list(hand).count(cards[3]) == 1) or (list(hand).count(cards[0]) == 1 and list(hand).count(cards[1]) == 2 and list(hand).count(cards[2]) == 1 and list(hand).count(cards[3]) == 1) or (list(hand).count(cards[0]) == 1 and list(hand).count(cards[1]) == 1 and list(hand).count(cards[2]) == 2 and list(hand).count(cards[3]) == 1) or (list(hand).count(cards[0]) == 1 and list(hand).count(cards[1]) == 1 and list(hand).count(cards[2]) == 1 and list(hand).count(cards[3]) == 2)
    else:
        return False

def is_high_card(hand):
    return len(set(hand)) == 5

#### With jokers for part 2

# all labels are equal
def is_five_of_a_kind_v2(hand):
    jokers = list(hand).count('J')
    cards = set(hand)
    return len(cards) == 1 or (len(cards) == 2 and jokers > 0)

def is_four_of_a_kind_v2(hand):
    jokers = list(hand).count('J')
    cards = list(set(hand))
    hand = list(hand)
    #print(cards)
    cards.sort(key=functools.cmp_to_key(compare_cards_in_hands_v2))
    cards.reverse()
    
    if len(cards) == 2:
        return hand.count(cards[0]) == 4 or hand.count(cards[1]) == 4    
    elif len(cards) == 3 and jokers > 0:
        hand1 = [cards[0] if card == 'J' and cards[0] != 'J' else card for card in hand]
        hand2 = [cards[1] if card == 'J' and cards[1] != 'J' else card for card in hand]
        return hand1.count(cards[0]) == 4 or hand2.count(cards[1]) == 4    
    else:
        return False

def is_full_house_v2(hand):
    jokers = list(hand).count('J')
    cards = list(set(hand))
    hand = list(hand)
    cards.sort(key=functools.cmp_to_key(compare_cards_in_hands_v2))
    cards.reverse()
    
    if len(cards) == 2:
        return (hand.count(cards[0]) == 3 and hand.count(cards[1]) == 2) or (hand.count(cards[0]) == 2 and hand.count(cards[1]) == 3) 
    elif len(cards) == 3 and jokers > 0:
        hand1 = [cards[0] if card == 'J' and cards[0] != 'J' else card for card in hand]
        hand2 = [cards[1] if card == 'J' and cards[1] != 'J' else card for card in hand]

        return (hand1.count(cards[0]) == 3 and hand.count(cards[1]) == 2) or (hand1.count(cards[0]) == 2 and hand.count(cards[1]) == 3) or (hand.count(cards[0]) == 3 and hand2.count(cards[1]) == 2) or (hand.count(cards[0]) == 2 and hand2.count(cards[1]) == 3) 
    else:
        return False

def is_three_of_a_kind_v2(hand):
    jokers = list(hand).count('J')
    cards = list(set(hand))
    hand = list(hand)
    cards.sort(key=functools.cmp_to_key(compare_cards_in_hands_v2))
    cards.reverse()
    
    if len(cards) == 3:
        return hand.count(cards[0]) == 3 or hand.count(cards[1]) == 3 or hand.count(cards[2]) == 3
    elif len(cards) == 4 and jokers > 0:
        hand1 = [cards[0] if card == 'J' and cards[0] != 'J' else card for card in hand]
        hand2 = [cards[1] if card == 'J' and cards[1] != 'J' else card for card in hand]
        hand3 = [cards[2] if card == 'J' and cards[2] != 'J' else card for card in hand]
        return hand1.count(cards[0]) == 3 or hand2.count(cards[1]) == 3 or hand3.count(cards[2]) == 3
    else:
        return False

def is_two_pair_v2(hand):
    jokers = list(hand).count('J')
    cards = list(set(hand))
    hand = list(hand)
    cards.sort(key=functools.cmp_to_key(compare_cards_in_hands_v2))
    cards.reverse()
    
    if len(cards) == 3:
        return (hand.count(cards[0]) == 2 and hand.count(cards[1]) == 2 and hand.count(cards[2]) == 1) or (hand.count(cards[0]) == 2 and hand.count(cards[1]) == 1 and hand.count(cards[2]) == 2) or (hand.count(cards[0]) == 1 and hand.count(cards[1]) == 2 and hand.count(cards[2]) == 2)
    elif len(cards) == 4 and jokers > 0:
        hand1 = [cards[0] if card == 'J' and cards[0] != 'J' else card for card in hand]
        hand2 = [cards[1] if card == 'J' and cards[1] != 'J' else card for card in hand]
        hand3 = [cards[2] if card == 'J' and cards[2] != 'J' else card for card in hand]
        
        return (hand1.count(cards[0]) == 2 and hand.count(cards[1]) == 2 and hand.count(cards[2]) == 1) or (hand1.count(cards[0]) == 2 and hand.count(cards[1]) == 1 and hand.count(cards[2]) == 2) or (hand1.count(cards[0]) == 1 and hand.count(cards[1]) == 2 and hand.count(cards[2]) == 2) or (hand.count(cards[0]) == 2 and hand2.count(cards[1]) == 2 and hand.count(cards[2]) == 1) or (hand.count(cards[0]) == 2 and hand2.count(cards[1]) == 1 and hand.count(cards[2]) == 2) or (hand.count(cards[0]) == 1 and hand2.count(cards[1]) == 2 and hand.count(cards[2]) == 2) or (hand.count(cards[0]) == 2 and hand.count(cards[1]) == 2 and hand3.count(cards[2]) == 1) or (hand.count(cards[0]) == 2 and hand.count(cards[1]) == 1 and hand3.count(cards[2]) == 2) or (hand.count(cards[0]) == 1 and hand.count(cards[1]) == 2 and hand3.count(cards[2]) == 2)
    else:
        return False

def is_pair_v2(hand):
    cards = list(set(hand))
    jokers = list(hand).count('J')
    hand = list(hand)
    cards.sort(key=functools.cmp_to_key(compare_cards_in_hands_v2))
    cards.reverse()
    
    if len(cards) == 4:
        return (hand.count(cards[0]) == 2 and hand.count(cards[1]) == 1 and hand.count(cards[2]) == 1 and hand.count(cards[3]) == 1) or (hand.count(cards[0]) == 1 and hand.count(cards[1]) == 2 and hand.count(cards[2]) == 1 and hand.count(cards[3]) == 1) or (hand.count(cards[0]) == 1 and hand.count(cards[1]) == 1 and hand.count(cards[2]) == 2 and hand.count(cards[3]) == 1) or (hand.count(cards[0]) == 1 and hand.count(cards[1]) == 1 and hand.count(cards[2]) == 1 and hand.count(cards[3]) == 2)
    elif len(cards) == 5 and jokers > 0:
        hand1 = [cards[0] if card == 'J' and cards[0] != 'J' else card for card in hand]
        hand2 = [cards[1] if card == 'J' and cards[1] != 'J' else card for card in hand]
        hand3 = [cards[2] if card == 'J' and cards[2] != 'J' else card for card in hand]
        hand4 = [cards[3] if card == 'J' and cards[3] != 'J' else card for card in hand]
        
        return (hand1.count(cards[0]) == 2 and hand.count(cards[1]) == 1 and hand.count(cards[2]) == 1 and hand.count(cards[3]) == 1) or (hand1.count(cards[0]) == 1 and hand.count(cards[1]) == 2 and hand.count(cards[2]) == 1 and hand.count(cards[3]) == 1) or (hand1.count(cards[0]) == 1 and hand.count(cards[1]) == 1 and hand.count(cards[2]) == 2 and hand.count(cards[3]) == 1) or (hand1.count(cards[0]) == 1 and hand.count(cards[1]) == 1 and hand.count(cards[2]) == 1 and hand.count(cards[3]) == 2) or (hand.count(cards[0]) == 2 and hand2.count(cards[1]) == 1 and hand.count(cards[2]) == 1 and hand.count(cards[3]) == 1) or (hand.count(cards[0]) == 1 and hand2.count(cards[1]) == 2 and hand.count(cards[2]) == 1 and hand.count(cards[3]) == 1) or (hand.count(cards[0]) == 1 and hand2.count(cards[1]) == 1 and hand.count(cards[2]) == 2 and hand.count(cards[3]) == 1) or (hand.count(cards[0]) == 1 and hand2.count(cards[1]) == 1 and hand.count(cards[2]) == 1 and hand.count(cards[3]) == 2) or (hand.count(cards[0]) == 2 and hand.count(cards[1]) == 1 and hand3.count(cards[2]) == 1 and hand.count(cards[3]) == 1) or (hand.count(cards[0]) == 1 and hand.count(cards[1]) == 2 and hand3.count(cards[2]) == 1 and hand.count(cards[3]) == 1) or (hand.count(cards[0]) == 1 and hand.count(cards[1]) == 1 and hand3.count(cards[2]) == 2 and hand.count(cards[3]) == 1) or (hand.count(cards[0]) == 1 and hand.count(cards[1]) == 1 and hand3.count(cards[2]) == 1 and hand.count(cards[3]) == 2) or (hand.count(cards[0]) == 2 and hand.count(cards[1]) == 1 and hand.count(cards[2]) == 1 and hand4.count(cards[3]) == 1) or (hand.count(cards[0]) == 1 and hand.count(cards[1]) == 2 and hand.count(cards[2]) == 1 and hand4.count(cards[3]) == 1) or (hand.count(cards[0]) == 1 and hand.count(cards[1]) == 1 and hand.count(cards[2]) == 2 and hand4.count(cards[3]) == 1) or (hand.count(cards[0]) == 1 and hand.count(cards[1]) == 1 and hand.count(cards[2]) == 1 and hand4.count(cards[3]) == 2)
    
    
    else:
        return False




# ranges from 1 to 7
def get_type_strenght(hand):
    if is_five_of_a_kind(hand):
        return 7
    elif is_four_of_a_kind(hand):
        return  6
    elif is_full_house(hand):
        return 5
    elif is_three_of_a_kind(hand):
        return 4
    elif is_two_pair(hand):
        return 3
    elif is_pair(hand):
        return 2
    else:
        return 1

def get_type_strenght_v2(hand):
    if is_five_of_a_kind_v2(hand):
        return 7
    elif is_four_of_a_kind_v2(hand):
        return  6
    elif is_full_house_v2(hand):
        return 5
    elif is_three_of_a_kind_v2(hand):
        return 4
    elif is_two_pair_v2(hand):
        return 3
    elif is_pair_v2(hand):
        return 2
    else:
        return 1

def compare_cards_in_hands(hand1, hand2):
    cards1 = list(hand1)
    cards2 = list(hand2)
    #print(hand1,'comparing cards with',hand2)
    for card1, card2 in zip(cards1, cards2):
        #print(card1,'compare with',card2) 
        if get_label_strenght(card1) != get_label_strenght(card2):
            if get_label_strenght(card1) > get_label_strenght(card2):
                return 1
            else:
                return -1

def compare_cards_in_hands_v2(hand1, hand2):
    cards1 = list(hand1)
    cards2 = list(hand2)
    #print(hand1,'comparing cards with',hand2)
    for card1, card2 in zip(cards1, cards2):
        #print(card1,'compare with',card2)
        if get_label_strenght_v2(card1) != get_label_strenght_v2(card2):
            if get_label_strenght_v2(card1) > get_label_strenght_v2(card2):
                return 1
            else:
                return -1
        
def compare_hands_v2(hand1, hand2):
    if get_type_strenght_v2(hand1) < get_type_strenght_v2(hand2):
        return -1
    elif get_type_strenght_v2(hand1) > get_type_strenght_v2(hand2):
        return 1
    else: # same type strength
        return compare_cards_in_hands_v2(hand1, hand2)


def compare_hands(hand1, hand2):
    if get_type_strenght(hand1) < get_type_strenght(hand2):
        return -1
    elif get_type_strenght(hand1) > get_type_strenght(hand2):
        return 1
    else: # same type strength
        return compare_cards_in_hands(hand1, hand2)



#Day 7, part 1: 251136060 (0.191 secs)
def day7_1(data):
    #data = read_input(2023, "07_teste")
    hands = []
    bets = dict()

    for line in data:
        input = line.split(' ')
        hands.append(input[0])
        bets[input[0]] = int(input[1])    

    hands.sort(key=functools.cmp_to_key(compare_hands))
    result = 0
    for i in range(len(hands)):
        k = hands[i]
        result += (i+1)*bets[k]    
  
    AssertExpectedResult(251136060, result)
    return result


def test_cases_part2_day7():
    five_kind = ['JJJJJ','AAAAA','JAAAA','AJAAA','AAJAA','AAAJA','AAAAJ']
    four_kind = ['AA8AA','TTTT8','JTTT8','TJTT8','TTJT8','TTTJ8','TTT8J','T55J5','KTJJT','QQQJA','QJJQ2','JJQJ4','JJ2J9','JTJ55']
    full_house = ['23332','J2233','2J233','22J33','223J3','2233J','22333','25J52']
    three_kind = ['AJKJ4','TTT98','JTT98','TJT98','TTJ98','TT9J8','TT98J','T9T8J','T98TJ','T98JT','TQJQ8']
    two_pair = ['23432','KK677','KK677']
    pair = ['32T3K','A23A4','32T3K','J2345','2J345','23J45','234J5','2345J','5TK4J']
    high_card = ['23456']
    
    for c in five_kind:
        if not is_five_of_a_kind_v2(c):
            print(c,"should be five of a kind")

    for c in four_kind:
        if not is_four_of_a_kind_v2(c):
            print(c,"should be four of a kind")

  
    for c in full_house:
        if not is_full_house_v2(c):
            print(c,"should be full house")
            
    for c in three_kind:
        if not is_three_of_a_kind_v2(c):
            print(c,"should be three of a kind")
    
    for c in two_pair:
        if not is_two_pair_v2(c):
            print(c,"should be two pair")
            
    for c in pair:
        if not is_pair_v2(c):
            print(c,"should be pair")


#249605933 too high
#249682966 --
#250771385
#249899797
#248913887 too low
def day7_2(data):
    #data = read_input(2023, "07_teste")
    hands = []
    bets = dict()

    for line in data:
        input = line.split(' ')
        hands.append(input[0])
        bets[input[0]] = int(input[1])    

    hands.sort(key=functools.cmp_to_key(compare_hands_v2))
    
    
    #print(hands)
    result = 0
    for i in range(len(hands)):
        k = hands[i]
        result += (i+1)*bets[k]  
        
        
    test_cases_part2_day7()    
    #print(get_type_strenght('QQQJA'))  
    #print(get_type_strenght('T55J5'))
    #print(get_label_strenght_v2('Q'))  
    #print(get_label_strenght_v2('T'))    
  
    AssertExpectedResult(251136060, result)
    return result

#endregion


#region ##### Day 8 #####

def how_many_steps_until_destination(instructions, desert_map):
    position = 'AAA'
    end = 'ZZZ'
    steps = 0
    size = len(instructions)

    while position != end:
        next_instruction = instructions[steps%size]

        (left, right) = desert_map.get(position)

        if next_instruction == 'L':
            position = left
        elif next_instruction == 'R':
            position = right
        steps += 1
    
    return steps

#Day 8, part 1: 16043 (0.042 secs)
#Day 8, part 2: 15726453850399 (0.584 secs)
def day8_1(data):
    #data = read_input(2023, "04_teste")    
    result = 0    
    instructions = list(data[0])
    desert_map = dict()

    for line in data[2:]:
        nodes = line.split('=')
        destinations = parse("({}, {})", nodes[1].strip())
        desert_map[nodes[0].strip()] = (destinations[0], destinations[1])

    result = how_many_steps_until_destination(instructions, desert_map)

    AssertExpectedResult(16043, result)
    return result

def how_many_steps_until_destination_v2(instructions, desert_map, node):
    position = node
    steps = 0
    size = len(instructions)

    while position[2] != 'Z':
        next_instruction = instructions[steps%size]
        (left, right) = desert_map.get(position)

        if next_instruction == 'L':
            position = left
        elif next_instruction == 'R':
            position = right

        steps += 1
    
    return steps

#19788348363346328285369657 too high
def day8_2(data):
    #data = read_input(2023, "08_teste")    
    result = 0    

    instructions = list(data[0])
    desert_map = dict()
    starting_nodes = []
    for line in data[2:]:
        destinations = parse("{} = ({}, {})", line)
        desert_map[destinations[0]] = (destinations[1], destinations[2])

    for k in desert_map.keys():
        if k[2] == 'A':
            starting_nodes.append(k)
    
    all_steps = []
    for node in starting_nodes:
        steps =  how_many_steps_until_destination_v2(instructions, desert_map, node)
        all_steps.append(steps)

    result = 1
    for steps in all_steps:
        result = lcm(result, steps)

    AssertExpectedResult(15726453850399, result)
    return result

#endregion

#region ##### Day 9 #####

def get_differences(history):
    rows = [copy.deepcopy(history)]

    i = 1
    
    while sum([1 for n in rows[i-1] if n != 0]) != 0:
        row = rows[i-1]
        
        next_numbers = []
        for n,m in zip(row, row[1:]):
            next_numbers.append(m-n) 
       
        rows.append(next_numbers)
        i += 1

    return rows

def extrapolate_next_number(history, part2 = False):    
    rows = get_differences(history)
    size = len(rows)-1
    
    i = size
    while i != 0:
        previous_row = rows[i-1]
        pos = 0 if part2 else len(previous_row)-1 
        previous_row_number = previous_row[pos]

        if size == i:
            current_row_number = 0
        else: 
            current_row_number = extrapolated_number
        extrapolated_number = previous_row_number - current_row_number if part2 else current_row_number + previous_row_number

        i-= 1

    return extrapolated_number


#Day 9, part 1: 1681758908 (0.037 secs)
#Day 9, part 2: 803 (0.005 secs)
def day9_1(data):
    #data = read_input(2023, "09_teste")    
    result = 0
    histories = []

    for line in data:
        histories.append(ints(line.split()))

    for history in histories:
        result += extrapolate_next_number(history)

    AssertExpectedResult(1681758908, result)
    return result

def day9_2(data):
    #data = read_input(2023, "09_teste")    
    result = 0     
    histories = []

    for line in data:
        histories.append(ints(line.split()))

    for history in histories:
        result += extrapolate_next_number(history, True)   

    AssertExpectedResult(803, result)
    return result

#endregion

#region ##### Day 10 #####

def is_pipe_connected(grid, this_pipe_coords, that_pipe_coords):
    (x,y) = this_pipe_coords
    (xx, yy) = that_pipe_coords
    this_pipe = grid[y][x]
    that_pipe = grid[yy][xx]
    is_connected = False

    east = (x+1, y)
    west = (x-1, y)
    north = (x, y-1)
    south = (x, y+1)

    north_pipes = ['|','7','F']
    south_pipes = ['J','|','L']
    west_pipes = ['-','L','F']
    east_pipes = ['-','J','7']

    if this_pipe == '|': #connected to north or south
        if that_pipe_coords == north:
            #print(this_pipe, "is connected to",that_pipe,"?", that_pipe in north_pipes)
            is_connected = that_pipe in north_pipes
        elif that_pipe_coords == south:
            #print(this_pipe, "is connected to",that_pipe,"?", that_pipe in south_pipes)
            is_connected = that_pipe in south_pipes
    elif this_pipe == '-': #connected to west or east
        if that_pipe_coords == west:
            is_connected = that_pipe in west_pipes
        elif that_pipe_coords == east:
            is_connected = that_pipe in east_pipes
    elif this_pipe == 'L': #connected to north or east
        if that_pipe_coords == north:
            is_connected = that_pipe in north_pipes
        elif that_pipe_coords == east:
            is_connected = that_pipe in east_pipes
    elif this_pipe == 'J': #connected to north or west
        if that_pipe_coords == north:
            is_connected = that_pipe in north_pipes
        elif that_pipe_coords == west:
            is_connected = that_pipe in west_pipes
    elif this_pipe == '7': #connected to west or south        
        if that_pipe_coords == south:
            is_connected = that_pipe in south_pipes
        elif that_pipe_coords == west:
            is_connected = that_pipe in west_pipes
    elif this_pipe == 'F': #connected to east or south
        if that_pipe_coords == south:
            is_connected = that_pipe in south_pipes
        elif that_pipe_coords == east:
            is_connected = that_pipe in east_pipes

    return is_connected

def find_connecting_pipes(grid, s_node, graph):
    grid_s = copy.deepcopy(grid)
    (x,y) = s_node
    east = (x+1, y)
    west = (x-1, y)
    north = (x, y-1)
    south = (x, y+1)

    possible_pipes = ['|', '-','L','J','7','F']
    
    possible_connections = {
        '|': [], 
        '-': [],
        'L': [],
        'J': [],
        '7': [],
        'F': []
    }

    for pipe in possible_pipes:
        grid_s[y][x] = pipe
        if is_pipe_connected(grid_s, s_node, north):
            possible_connections[pipe].append(north)
        if is_pipe_connected(grid_s, s_node, south):
            possible_connections[pipe].append(south)
        if is_pipe_connected(grid_s, s_node, east):
            possible_connections[pipe].append(east)
        if is_pipe_connected(grid_s, s_node, west):
            possible_connections[pipe].append(west)
        
        if len(possible_connections[pipe]) == 2:
            graph[(x,y)] = possible_connections[pipe]
            return (graph, pipe)
        
    return graph,pipe
    
#Day 10, part 1: 6979 (0.104 secs)
#Day 10, part 2: 443 (55.916 secs)
def day10_1(data):
    #data = read_input(2023, "10_teste")    
    result = 0
    grid = []

    grid = buildMapGrid(data, initValue='.')  
    graph = buildGraphFromMap_v2(grid, '.', is_pipe_connected)

    s_node = find_starting_point(grid, 'S')
    print("starting point at", s_node)
    graph, _ = find_connecting_pipes(grid, s_node, graph)

    loop = bfs(graph, s_node, compute_distances=True)
    #print(loop)
    
    max_steps = -1
    for (_, steps) in loop:
        if steps > max_steps:
            max_steps = steps
    result = max_steps

    '''
    for k in graph.keys():
        if len(graph[k]) > 0:
            print(k,"->",graph[k],grid[k[1]][k[0]])
    '''
    
    AssertExpectedResult(6979, result)
    return result

# 40 wrong
def day10_2(data):
    #data = read_input(2023, "10_teste")    
    result = 0    
    grid = []

    symbols_map = {
        "F": "\u250F",
        "J": "\u251B",
        "L": "\u2517",
        "7": "\u2513",
        "|": "\u2503",
        "-": "\u2501"
    }

    grid = buildMapGrid(data, initValue='.', withPadding=False) 
    graph = buildGraphFromMap_v2(grid, '.', is_pipe_connected)

    s_node = find_starting_point(grid, 'S')
    print("starting point at", s_node)
    graph, pipe = find_connecting_pipes(grid, s_node, graph)
    grid[s_node[1]][s_node[0]] = symbols_map[pipe] # replace S with correct pipe

    #printMap(grid)

    loop = bfs(graph, s_node)
    #print("loop:",loop)
    
    rows = len(grid)
    columns = len(grid[0])    
    #hack to account for squeezes
    for y in range(rows):           
        for x in range(columns):
            tile = grid[y][x]
            if tile != '.' and (x,y) not in loop:
                grid[y][x] = '.'

    inside_points = 0
    # going from left to right, then change row
    for yy in range(rows):           
        boundaries = 0
        for xx in range(columns):
            tile = grid[yy][xx]

            # for each empty space, "draw" a ray forward and count boundaries of the loop other than -
            if tile == '.':
                
                boundaries = 0
                expect_seven = False
                expect_j = False
                for x in range(xx,columns):
                    point = (x,yy)
                    ttile = grid[yy][x]
                    
                    # "FJ", and "L7" are corners that should count as 1
                    if point in loop and ttile != '-':
                        
                        if ttile =='L':
                            expect_seven = True
                        elif expect_seven:
                            if ttile == '7':
                                boundaries += 1
                            expect_seven = False

                        elif ttile =='F':
                            expect_j = True
                        elif expect_j:
                            if ttile == 'J':
                                boundaries += 1
                            expect_j = False
                        else:
                            boundaries +=1

                # point is inside if the number of boundaries crossed is odd (impar)       
                if boundaries % 2 != 0: # inside point
                    inside_points += 1  
                    grid[yy][xx] = "I"


    print("inside points:", inside_points)
    #printMap(grid, symbolsMap=symbols_map)
    #printMap(grid) 

    result = inside_points    
    AssertExpectedResult(443, result)
    return result

#endregion


#region ##### Day 11 #####


def expand_univere(universe, delta = 2):
    rows = len(universe)
    columns = len(universe[0])    
    
    expand_ys = []
    expand_xs = []
    
    galaxies_row = 0
    for y in range(rows):
        galaxies_row = sum([1 for atom in universe[y] if atom == "#"])
        if galaxies_row == 0:
            expand_ys.append(y)
       
    for x in range(columns):
        galaxies_col = 0 
        #search by column              
        for y in range(rows):
            if universe[y][x] == "#":
                galaxies_col += 1
                
        if galaxies_col == 0:
            expand_xs.append(x)
       
    #print(expand_xs) 
    #print(expand_ys) 
    galaxies = set() 
    delta=delta-1
    
    for y in range(rows):
        for x in range(columns):
            if universe[y][x] == "#":
                xx = x
                for l in expand_xs:
                    if x > l:
                        xx+= delta         
                yy = y
                for l in expand_ys:               
                    if y > l:
                        yy+= delta

                galaxies.add((xx,yy))
                
    return galaxies

#Day 11, part 1: 9724940 (0.135 secs)
#Day 11, part 2: 569052586852 (0.030 secs)
def day11_1(data):
    #data = read_input(2023, "11_teste")    
    result = 0  

    universe = buildMapGrid(data, initValue='.', withPadding=False)
    galaxies = expand_univere(universe)
    
    galaxies_combinations = combinations(galaxies, 2)
    for (galaxy_a, galaxy_b) in list(galaxies_combinations):
        #print("finding shortest path from a:", galaxy_a,"to b:", galaxy_b)
        min_path = sys.maxsize
     
        if galaxy_a != galaxy_b:
            dist = abs(galaxy_a[0] - galaxy_b[0]) + abs(galaxy_a[1] - galaxy_b[1])
            if dist < min_path:
                    min_path = dist
        result += min_path
        
    AssertExpectedResult(9724940, result)
    return result


def day11_2(data):
    #data = read_input(2023, "11_teste")    
    result = 0    
             
    universe = buildMapGrid(data, initValue='.', withPadding=False)
    galaxies = expand_univere(universe, delta = 1000000)
  
    galaxies_combinations = combinations(galaxies, 2)
    
    for (galaxy_a, galaxy_b) in list(galaxies_combinations):
        #print("finding shortest path from a:", galaxy_a,"to b:", galaxy_b)
        min_path = sys.maxsize
     
        if galaxy_a != galaxy_b:
            (x,y) = galaxy_a
            (xx,yy) = galaxy_b
            
            dist = abs(x - xx) + abs(y - yy) 
            if dist < min_path:
                    min_path = dist
        result += min_path
        
    AssertExpectedResult(569052586852, result)
    return result

#endregion

#region ##### Day 12 #####


''' test cases
#.#.### 1,1,3
.#...#....###. 1,1,3
.#.###.#.###### 1,3,1,6
####.#...#... 4,1,1
#....######..#####. 1,6,5
.###.##....# 3,2,1
'''
def check_condition(springs, condition):
    condition_copy = copy.deepcopy(condition)
    #print("testing if", springs,"meets condition", condition_copy)
    condition_copy.reverse()

    if sum([1 for s in springs if s == '#']) != sum(condition):
        return False

    contiguous = 0
    start = False
    for spring in springs:
        if spring == "#":
            contiguous+=1
            if not start:
                start = True
        elif start and contiguous > 0 and len(condition_copy) > 0:
            expected = condition_copy.pop()
            #print("checking contiguous", contiguous,"equals expected",expected,":", contiguous != expected)
            if contiguous != expected:
                return False
            contiguous = 0

    if len(condition_copy) > 0:
        expected = condition_copy.pop()
        #print("checking contiguous", contiguous,"equals expected",expected,":", contiguous != expected)
        if contiguous != expected:
            return False
        
    return True


def run_test_cases():
    ''' test cases
        #.#.### 1,1,3
        .#...#....###. 1,1,3
        .#.###.#.###### 1,3,1,6
        ####.#...#... 4,1,1
        #....######..#####. 1,6,5
        .###.##....# 3,2,1
    '''
    print(check_condition(list('#.#.###'), [1,1,3]))
    print()
    print(check_condition(list('.#...#....###.'), [1,1,3]))
    print()
    print(check_condition(list('.#.###.#.######'), [1,3,1,6]))
    print()
    print(check_condition(list('####.#...#...'), [4,1,1]))
    print()
    print(check_condition(list('#....######..#####.'), [1,6,5]))
    print()
    print(check_condition(list('.###.##....#'), [ 3,2,1]))
    
    test_cases = ['.###.##.#...',
                    '.###.##..#..',
                    '.###.##...#.',
                    '.###.##....#',
                    '.###..##.#..',
                    '.###..##..#.',
                    '.###..##...#',
                    '.###...##.#.',
                    '.###...##..#',
                    '.#.#....##.#']
    for test in test_cases:
        print(check_condition(list(test),[3,2,1]))
        print()


def combine_springs(springs):
     size = len(springs)
     for i in itertools.product(*(['.#'] * size)) :
         yield i


def find_unknown_springs(springs):
    unkowns = []
    for i in range(len(springs)):
        if springs[i] == '?':
            unkowns.append(i)
    return unkowns

#\.*\#{3}\.+\#{2}\.+\#{1}\.*
def verify_condition_regex(replaced, condition):
    pattern = '\\.*'
    for n in condition:
        pattern += '\\#{'+str(n)+'}\\.+'
    pattern = pattern[:-1]+'*'
    #print(pattern)

    return re.match(pattern, replaced)


def verify_condition(replaced, condition):
    #condition = [int(c) for c in list(condition) if c != ',']
    replaced = list(replaced)
    for cond in condition:

        try:
            i = replaced.index('#')
        except ValueError:
            return 0
        test = replaced[i:i+cond]
            
        #print("testing:",test,(['#']*cond),cond)
        if test != (['#']*cond):
            #print(cond,"condition not met")
            #print()
            return 0
        else:        

            #print(cond,"condition met!")
            replaced = replaced[i+cond:]
            #print("-->",replaced)
            if len(condition)> 0 and len(replaced) > 0:
                if replaced[0] == '#':
                    #print(cond,"condition not met")
                    #print()
                    return 0                
        
    if "#" in replaced:
        return 0

    #print(replaced,"satisfied all conditions:",condition)
    #print()
    return 1

#@lru_cache(maxsize=128)    
#@hashable_cache
def check_condition_rec(springs, condition, counter, replaced, cache):    
    #print("call with",springs, condition, counter)
      
    if len(springs) == 0:
        if replaced in cache:
            return cache[replaced]
        else:
            res = verify_condition(replaced, condition)            
            cache[replaced] = res
            return res
        
    elif springs[0] == '#':        
        return check_condition_rec(springs[1:], condition, counter, replaced + springs[0], cache)
        
    elif springs[0] == '.':       
        return check_condition_rec(springs[1:], condition, counter, replaced + springs[0], cache) 
       
    elif springs[0] == '?':
        #print("Replacing ?")        
        return check_condition_rec(springs[1:], condition, counter, replaced + '#', cache) + check_condition_rec(springs[1:], condition, counter, replaced + '.', cache)

def check_condition_main(s, c):   
    return check_condition_rec(s, c, 0, '', cache)

def check_condition_v2(line):

    data_split = line.split(" ")
    
    s = list(data_split[0])
    c = ints(data_split[1].split(','))
    #print(s)
    #print(c)

    res = check_condition_main(s, c)
    #print("call check_condition_v2",s, c, res)
    #print()
    return res

# takes 61s on real input
def naive_solution(records):
    size = 0
    result = 0
    for (springs, condition) in records:
        
        result += check_condition(springs, condition)
        unkowns = find_unknown_springs(springs)
        #print(unkowns)
        possible_combinatios = combine_springs(unkowns)
        
        arrangments = 0
        
        for comb in possible_combinatios:
            size+=1
            j = 0
            springs_copy = copy.deepcopy(springs)
            for i in unkowns:
                val = comb[j]
                j +=1
                springs_copy[i] = val
            if check_condition(springs_copy, condition):
                arrangments +=1
               # print()
        #print(springs,"has",arrangments,"possible arrangements")
        result += arrangments
    return result

def day12_1(data):
    #data = read_input(2023, "12_teste")    
    result = 0  
    global cache
    cache = defaultdict(int)

    for line in data:        
        result += check_condition_v2(line)
        
    #print(size)
    AssertExpectedResult(7251, result)
    return result

def check_condition_v3(line):
    global cache
    cache = defaultdict(int)

    data_split = line.split(" ")

    s =''
    for _ in range(5):
        s += data_split[0]+'?'
    s = list(s[:-1])

    c =''
    for _ in range(5):
        c += data_split[1]+','
    c = ints(c[:-1].split(','))
    
    #print(s)
    #print(c)

    res = check_condition_main(s, c)
    #print("call check_condition_v2",s, c, res)
    #print()
    return res

def day12_2(data):
    data = read_input(2023, "12_teste")    
    result = 0    
    records = []
    
    
    #for line in data:        
    #    result += check_condition_v3(line)
             
    AssertExpectedResult(0, result)
    return result

#endregion


#region ##### Day 13 #####

def search_vertical_line_all_sizes(max_size, i, l, r, candidates, row):
    
    size = max_size
    if True:
        #rows = set() 
        left = copy.deepcopy(l)
        right = copy.deepcopy(r)
        
        left = left[len(left)-size :]
        right = right[:size] #len(left)-size

        #print("slice size:", size)
        #print("left:",left,)
        #print("right:",right)

        right.reverse()
        #print("col:",i,":",left,"==",right)
        #print()
        
        if left == right and len(left) > 0:
            if i not in candidates:                
                candidates[i] = (size, row)
            else:
                candidates[i] = (size, row)
            #candidates.append(i)
        
    #print("candidates for row, col",row,i,":", candidates)

    return candidates

def search_vertical_line(grid, row):
    columns = len(grid[0])

    #print("row:",row)
    candidates = dict()
    for i in range(columns):
        left = grid[row][:i]
        right = grid[row][i:]
        
        #print("col:",i)
        #print("left:",left,)
        #print("right:",right)

        size = min(len(left), len(right))
        
        candidates = search_vertical_line_all_sizes(size, i, left, right, candidates, row)
        
        '''
        left = left[len(left)-size : ]
        right = right[:i] #len(left)-size

        print("slice size:", size)
        print("left:",left,)
        print("right:",right)

        right.reverse()
        print(i,":",left,"==",right)
        print()
        
        if left == right and len(left) > 0:
            candidates.append(i)
        '''

    return candidates


def search_horizontal_line(grid, row):
    up = grid[:row]
    down = grid[row:]
        
    #print(row,":")
    #print("up:",up,)
    #print()
    #print("down:",down)

    size = min(len(down), len(up))
    up = up[len(up)-size:]
    down = down[:size]

    #print("size:",size)
    #print("up:",up,)
    #print()
    #print("down:",down)
   
    down.reverse()
    #print(row,":",up,"==",down)
    #print()
        
    if down == up and len(down) > 0:
        return row
    
    return None

def update_mirror_candidates(mirrors, mirror_point):

    #{2: (1, 6), 3: (3, 6), 4: (1, 6), 7: (2, 6)}
    for k in mirror_point.keys():
        (size, row) = mirror_point.get(k)
        
        if k not in mirrors:
            mirrors[k] = (size, set())
        
        (size, rows) = mirrors[k]
        rows.add(row)
        mirrors[k] = (size, rows)

def extract_mirror_point(mirrors, rows):

    #{2: (1, list(rows)), 3: (3, 6), 4: (1, 6), 7: (2, 6)}
    vertical_line = []
    for k in mirrors.keys():
        (_, rr) = mirrors.get(k)
        if len(rr) == rows:
            vertical_line.append(k)

    return vertical_line

def find_mirror_point(grid):
    rows = len(grid)

    mirror_point = None
    mirrors = dict()
    horizontal_mirror = []

    for r in range(rows):
        mirror_point = search_vertical_line(grid, r)
        update_mirror_candidates(mirrors, mirror_point)
    
    vertical_mirror = extract_mirror_point(mirrors, rows)
    
    mirrors = dict()
    for r in range(rows):
        mirror_point = search_horizontal_line(grid, r)
        
        if mirror_point:
            horizontal_mirror.append(mirror_point)
    
    
    return vertical_mirror, horizontal_mirror    

def swap_tile(gg, i, j):
    g = copy.deepcopy(gg)
    
    if  g[j][i] == '.':
        g[j][i] = '#'
    elif g[j][i] == '#':
        g[j][i] = '.'
        
    return g

def summarise_pattern_notes(grids, swap=False):    
    result = 0    

    test_case = 0
    for g in grids:
        grid = buildMapGrid(g, '.', withPadding=False)
        
        vertical, horizontal = find_mirror_point(grid) 
        
        original_v = vertical[0] if vertical else vertical
        original_h = horizontal[0] if horizontal else horizontal

        original_grid = copy.deepcopy(grid)
        count = 0
        rows = len(original_grid)
        columns = len(original_grid[0])

        test_case += 1

        #print("Test case:", test_case)
        
        if swap:          
            
            end = False
            for y in range(rows):
                
                for x in range(columns):
                    #end = False
                    use_v = False
                    use_h = False
                    
                    #print("swapping:",x,y)
                    count+=1
                    grid = swap_tile(original_grid,x,y)
                    vertical, horizontal = find_mirror_point(grid)
                    
                    #printMap(grid)

                    #print("horizontal main",horizontal)
                    
                    new_v = [v for v in vertical if v != original_v] if vertical else vertical
                    new_h = [h for h in horizontal if h != original_h] if horizontal else horizontal

                    #print("new_h main",new_h)

                    #print("originals main:",original_v,original_h)

                    
                    #print(new_v, new_h)
                    if new_v != original_v and len(vertical) > 0 and len(new_v) > 0:
                        #print("vertical:",new_v)
                        #print("orignal_v:",original_v)
                        if original_v in vertical:
                            vertical.remove(original_v)
                        end = True
                        use_v = True
                        break
                    
                    
                    if new_h != original_h and len(horizontal) > 0 and len(new_h) > 0:
                        #print("len(horizontal):",len(horizontal))
                        #print("horizontal:",horizontal)
                        #print("new_h:",new_h)
                        #print("original_h:",original_h)
                        if original_h in horizontal:
                            horizontal.remove(original_h)
                        end = True
                        use_h = True
                        break
                    
                if end:
                    break
               
        
            #print("count, rows, coluns, rows*columns,",count, rows, columns, rows*columns)
            #if count == rows * columns:
            #    print("BOOM! Test case:", test_case)
        
        if vertical and not horizontal:            
            v = original_v if len(vertical) == 0 else vertical
            result += sum(v)
            print("Vertical mirror point found at",vertical)
        
        if horizontal and not vertical:
            hh = original_h if len(horizontal) == 0 else horizontal
            
            result += sum([h*100 for h in hh])
            print("Horizontal mirror point found at",horizontal)
        
        if not horizontal and not vertical:
            print("No mirror found")
            print()
        if horizontal and vertical:
            if use_h:
                hh = original_h if len(horizontal) == 0 else horizontal
                result += sum([h*100 for h in hh])
                print("Horizontal mirror point found at",hh)
            elif use_v:
                v = original_v if len(vertical) == 0 else vertical
                result += sum(v)
                print("Vertical mirror point found at",v)
                
           # print("Two mirrors found!")
            print()
            
    return result

# too low 28612, 28729, 28818, 28818
#Day 13, part 1: 34918 (0.119 secs)
#Day 13, part 2: 33054 (6.659 secs)
def day13_1(data):
    #data = read_input(2023, "13_t")    
    result = 0  
    grids = []

    while(data):
        try:
            i = data.index('')
        except ValueError:
            grids.append(data)
            break
        grid = data[:i]
        grids.append(grid)
        data = data[i+1:]

    result = summarise_pattern_notes(grids)

    AssertExpectedResult(34918, result)
    return result

# too low 25245 
def day13_2(data):
    #data = read_input(2023, "13_t")    
    result = 0  
    grids = []

    while(data):
        try:
            i = data.index('')
        except ValueError:
            grids.append(data)
            break
        grid = data[:i]
        grids.append(grid)
        data = data[i+1:]

    result = summarise_pattern_notes(grids, swap = True)

    AssertExpectedResult(0, result)
    return result

#endregion

#region ##### Day 14 #####

def total_load(platform):
    rows = len(platform)
    columns = len(platform[0])
    load = 0

    line = rows
    for y in range(rows):
        for x in range(columns):
            if platform[y][x] == 'O':
                load += line
        line -= 1
    return load

def tilt_platform(platform):
    rows = len(platform)
    columns = len(platform[0])
    new_platform = copy.deepcopy(platform)
    # column -> sorted_list(rows)
    empty_tiles = dict()
    empty_rows = []

    for x in range(columns):
        empty_rows = []
        found_hard_rock = False
        first_rock = False
        for y in range(rows-1, 0, -1):
            if platform[y][x] == '.' and not found_hard_rock:
                if x not in empty_tiles:
                    empty_tiles[x] = []
                empty_rows.append(y)
            if platform[y][x] == 'O' and not first_rock:
                first_rock = True
            if platform[y][x] == '#' and first_rock:
                found_hard_rock = True
                break
        empty_rows.sort()
        empty_tiles[x] = empty_rows
    
    for k in empty_tiles.keys():
        print("column", k, "has empty rows", empty_tiles[k])

    for x in range(columns):
        for y in range(rows):
            if platform[y][x] == 'O':
                empty_rows = empty_tiles[x]
                next_row = empty_rows[0]
                new_platform[y][x] = '.' #empty current tile
                new_platform[next_row][x] = 'O' #move rock to empty tile
                empty_tiles[x] = empty_rows[1:] #update empty tiles
    
   
    return new_platform

def move_rock_north(x,y,platform):
    new_platform = copy.deepcopy(platform)

    new_y = y
    for i in range(y-1, -1, -1):
        if platform[i][x] == '.':
            new_y = i
        elif platform[i][x] == '#':
            break

    new_platform[y][x] = '.' #empty current tile
    new_platform[new_y][x] = 'O' #move rock to empty tile
    #print("Moved from",x,y,"->",x,new_y)
    
    return new_platform

def move_rock_south(x,y,platform):
    new_platform = copy.deepcopy(platform)

    new_y = y
    for i in range(y,len(platform)):
        if platform[i][x] == '.':
            new_y = i
        elif platform[i][x] == '#':
            break

    new_platform[y][x] = '.' #empty current tile
    new_platform[new_y][x] = 'O' #move rock to empty tile
    #print("Moved from",x,y,"->",x,new_y)
    
    return new_platform

def move_rock_west(x,y,platform):
    new_platform = copy.deepcopy(platform)
    #print("moving west")
    
    new_x = x
    for i in range(x, -1, -1):
        #print("testing",x,y,"->",platform[y][x])
        if platform[y][i] == '.':
            new_x = i
        elif platform[y][i] == '#':
            #print("hit a wall at",x,y)
            break

    new_platform[y][x] = '.' #empty current tile
    new_platform[y][new_x] = 'O' #move rock to empty tile
    #print("Moved from",x,y,"->",new_x, y)
    
    return new_platform

def move_rock_east(x,y,platform):
    new_platform = copy.deepcopy(platform)

    new_x = x
    for i in range(x, len(platform[0])):
        if platform[y][i] == '.':
            new_x = i
        elif platform[y][i] == '#':
            break

    new_platform[y][x] = '.' #empty current tile
    new_platform[y][new_x] = 'O' #move rock to empty tile
    #print("Moved from",x,y,"->",x,new_y)
    
    return new_platform

def tilt_platformv2(platform, cycles=0):
    #north, then west, then south, then east
    move_operations = [move_rock_north, move_rock_west, move_rock_south, move_rock_east]
    
    rows = len(platform)
    columns = len(platform[0])
    new_platform = copy.deepcopy(platform)

    if cycles == 0:
        move_operations = [move_operations[0]]
        cycles = 1

    #printMap(platform)
    
    for c in range(cycles):
        for move_rock in move_operations:                
            for x in range(columns):
                for y in range(rows):
                    if platform[y][x] == 'O':
                        new_platform = move_rock(x,y,new_platform)
            
            #printMap(new_platform)
            platform = new_platform
        #printMap(new_platform)
        print("cycle", c, ":",total_load(new_platform))
    return new_platform


def day14_1(data):
    data = read_input(2023, "14_teste")    
    result = 0  

    platform = buildMapGrid(data, '.', withPadding=False)
    platform = tilt_platformv2(platform)
    result = total_load(platform)

    AssertExpectedResult(110128, result)
    return result


def day14_2(data):
    #data = read_input(2023, "14_teste")    
    result = 0    
    cycles = 1000 #1000000000
    
    '''
     Strategy for part 2 was to print the first 1k cycles and check por a pattern, it timed out after 237 iterations that took 90m
     but was enough to detect a pattern and compute the result:
     
        cycle 179 : 103865
        cycle 180 : 103863
        cycle 181 : 103876
        cycle 182 : 103860
        cycle 183 : 103854
        cycle 184 : 103845
        cycle 185 : 103855
        cycle 186 : 103867
        cycle 187 : 103861
        cycle 188 : 103878
        cycle 189 : 103858
        cycle 190 : 103856
        cycle 191 : 103843
        cycle 192 : 103857

        (1000000000 - 179) % 14     
        
        Still unsure about the programmatic approach to this problem.
        
        TIP: should study cycle detection algorithms for these problems
    '''
    platform = buildMapGrid(data, '.', withPadding=False)
       
    platform = tilt_platformv2(platform, cycles)
    result = total_load(platform)
    #1 - 87
    #2,3,4 - 69
    #5 - 65
    #6 - 64
    
    AssertExpectedResult(103861, result)
    return result

#endregion


#region ##### Day 15 #####

def HASH_algorithm(value):
    current_value = 0
    for c in value:
        current_value += ord(c)
        current_value *= 17
        current_value %= 256
    return current_value



def day15_1(data):
    #data = read_input(2023, "15_teste")    
    result = 0  

    for value in data[0].split(','):
        result += HASH_algorithm(value)

    AssertExpectedResult(513214, result)
    return result


def focusing_power(boxes):
    total_power = 0
    for box in boxes.keys():        
        lenses = boxes[box]

        for i in range(len(lenses)):
            power = 1
            power *= (1 + box)
            power *= (i+1)
            (_, focal_length) = lenses[i]
            power *= int(focal_length)
            #print(power)
            total_power += power
    return total_power



def day15_2(data):
    #data = read_input(2023, "15_teste")    
    result = 0    

    boxes = dict()

    for step in data[0].split(','):
        focal_lenght = None
        add_lens = False

        if '=' in step:
            value = [parse("{}={}", step)][0]
            focal_lenght = value[1]
            add_lens = True
        else:
            value = [parse("{}-", step)][0]
        
        label = value[0]
        box = HASH_algorithm(label)

        if box not in boxes:
            boxes[box] = []
        
        lenses = boxes[box]
        if add_lens:
            elem = (label, focal_lenght)            
            lenses_replaced = False
            #if lens already exists
            i = 0
            for (lens, _) in lenses:
                if label == lens:
                    #replace lenses                
                    boxes[box] = lenses[:i] + [elem] + lenses[i+1:]
                    lenses_replaced = True
                i +=1
            if not lenses_replaced:
                boxes[box].append(elem)
        else:
            i = 0
            for (lens, _) in lenses:
                if label == lens:
                    boxes[box] = lenses[:i] + lenses[i+1:]
                i += 1


    #print(boxes)
    result = focusing_power(boxes)
             
    AssertExpectedResult(0, result)
    return result

#endregion

#region ##### Day 16 #####


def update_current_beam_position(beams, rows, columns):
    new_beams = []

    for ((x,y), direction) in beams:

        if direction == '->':
            if x + 1 < columns:
                new_beams.append(((x+1 , y), direction))
        elif direction == '<-':
            if x-1 >= 0:
                new_beams.append(((x-1 , y), direction))
        elif direction == '^':
            if y-1 >= 0:
                new_beams.append(((x, y-1), direction))
        elif direction == 'v':
            if y + 1 < rows:
                new_beams.append(((x, y+1), direction))
    
    return new_beams

def update_beam_direction(beams, contraption):
    new_beams = []    

    while len(beams) > 0:
        (position, direction) = beams.pop()
        (x, y) = position
        tile = contraption[y][x]
        #print("beam",x,y,"[",direction,"]", "on tile", tile)

        if tile == '/':
            if direction == '->':
                new_beams.append((position, '^'))
            elif direction == '<-':
                new_beams.append((position, 'v'))
            elif direction == '^':
                new_beams.append((position, '->'))
            elif direction == 'v':
                new_beams.append((position, '<-'))
        elif tile == '\\':
            if direction == '->':
                new_beams.append((position, 'v'))
            elif direction == '<-':
                new_beams.append((position, '^'))
            elif direction == '^':
                new_beams.append((position, '<-'))
            elif direction == 'v':
                new_beams.append((position, '->'))
        elif tile == '|':
            if direction == '->' or direction == '<-':
                new_beams.append((position, 'v'))
                new_beams.append((position, '^'))
            else:
                new_beams.append((position, direction))
        elif tile == '-':      
            if direction == '^' or direction == 'v':
                new_beams.append((position, '<-'))
                new_beams.append((position, '->'))
            else:
                new_beams.append((position, direction))
        else:
            new_beams.append((position, direction))
    
    return new_beams

def count_energized_tiles(beam, contraption): 
    rows = len(contraption)
    columns = len(contraption[0])
    beams = [ beam ]
    enerized_contraption = build_empty_grid(rows, columns, '.', withPadding=False)
    energized_tiles = set()

    i = 0
    while True:
        before = len(energized_tiles)

        for ((xx, yy), _) in beams:
            enerized_contraption[yy][xx] = "#"
            energized_tiles.add((xx,yy))
                   
        beams = update_beam_direction(beams, contraption)        
        beams = update_current_beam_position(beams, rows, columns)   
        after = len(energized_tiles)

        if before == after:
            i += 1
            if i == 10:
                break
        else:
            i = 0

    return after



def day16_1(data):
    #data = read_input(2023, "16_teste")    
    result = 0  
    beam = ((0,0), '->')

    contraption = buildMapGrid(data, '.', withPadding=False)

    result = count_energized_tiles(beam, contraption)


    AssertExpectedResult(8116, result)
    return result


def day16_2(data):
    data = read_input(2023, "16_teste")    
    result = 0    
    contraption = buildMapGrid(data, '.', withPadding=False)
    rows = len(contraption)
    columns = len(contraption[0])
    rows = len(contraption)
    columns = len(contraption[0])

    
    for x in range(columns):
        # top row
        beam = ((x,0), 'v')
        res = count_energized_tiles(beam,contraption)
        if res > result:
            result = res
        
        # bottom row
        beam = ((x,columns-1), '^')
        res = count_energized_tiles(beam,contraption)
        if res > result:
            result = res
    
    for y in range(rows):
        # first column
        beam = ((0, y), '->')
        res = count_energized_tiles(beam,contraption)
        if res > result:
            result = res
        
        # last column
        beam = ((rows-1, y), '<-')
        res = count_energized_tiles(beam,contraption)
        if res > result:
            result = res
    
             
    AssertExpectedResult(8383, result)
    return result

#endregion

#region ##### Day 17 #####


def get_dir(dir):
    # (1,0) right
    # (-1,0) left
    # (0,-1) up
    # (0,1) down
    if dir == (1,0):
        return "right"
    elif dir == (-1,0):
        return "left"
    elif dir == (0,-1):
        return "up"
    elif dir == (0,1):
        return "down"

def build_adjacency_list(rows, columns):

    # (1,0) right
    # (-1,0) left
    # (0,-1) up
    # (0,1) down
    right_coords = (1,0)
    left_coords = (-1,0)
    up_coords = (0,-1)
    down_coords = (0,1)

    # positions, direction delta, count in direction
    builder = [( (0,0) , (1, 0), 0 ), ( (0,0) , (0, 1), 0 )]
    
    adjacency_list = dict()
    #for y in range(rows):
    #    for x in range(columns):
    visited = set()
    cc = 0
    while len(builder) > 0:
            cc+=1

            coords, dir, length = builder.pop()
            (x,y) = coords  
   
                
            #print(coords, dir)
            if  cc >1000:
                break        

            right = (x+1, y) if x+1 <= columns-1 else None
            left = (x-1, y) if x-1 >= 0 else None
            up = (x, y-1) if y-1 >= 0 else None
            down = (x, y+1) if y+1 <= rows-1 else None


            key = (coords, get_dir(dir))

            if key not in adjacency_list:
                adjacency_list[key] = set()


            if dir == right_coords:
                #within boundaries and hasn't taken 3 steps in same direction
                if right and length < 3: 
                    adjacency_list[key].add((right, get_dir(dir)))

                    # add to queue if new coords haven't been processed yet
                    if (right, dir) not in visited:
                        builder.append( (right, dir, length + 1) )
                
                if True: # outside boundaries OR has taken more than 3 steps, time to turn 90º
                    
                    if up:
                        if (up, up_coords) not in visited:
                            builder.append( (up, up_coords, 0) )

                        val = (up, get_dir(up_coords))
                        adjacency_list[key].add(val)
                    
                    if down:
                        if (down, down_coords) not in visited:
                            builder.append( (down, down_coords, 0) )

                        val = (down, get_dir(down_coords))
                        adjacency_list[key].add(val)
            
            if dir == left_coords:

                if left and length < 3:
                    adjacency_list[key].add((left, get_dir(dir)))
                    if (left, dir) not in visited:
                        builder.append( (left, dir, length + 1) )
                if True:

                    if up:
                        if (up, up_coords) not in visited:
                            builder.append( (up, up_coords, 0) )

                        val = (up, get_dir(up_coords))
                        adjacency_list[key].add(val)
                    
                    if down:
                        if (down, down_coords) not in visited:
                            builder.append( (down, down_coords, 0) )

                        val = (down, get_dir(down_coords))
                        adjacency_list[key].add(val)

            if dir == up_coords:

                if up and length < 3:
                    adjacency_list[key].add((up, get_dir(dir)))
                    if (up, dir) not in visited:
                        builder.append( (up, dir, length + 1) )
                
                if True:
                    if left:
                        val = (left, get_dir(left_coords))
                        adjacency_list[key].add(val)
                        if (left, left_coords) not in visited:
                            builder.append( (left, left_coords, 0) )

                    if right:
                        if (right, right_coords) not in visited:
                            builder.append( (right, right_coords, 0) )
                        val = (right, get_dir(right_coords))
                        adjacency_list[key].add(val)

            if dir == down_coords:
                if down and length < 3:
                    adjacency_list[key].add((down, get_dir(dir)))
                    if (down, dir) not in visited:
                        builder.append( (down, dir, length + 1) )
                
                if True:
                    if left:
                        val = (left, get_dir(left_coords))
                        adjacency_list[key].add(val)
                        if (left, left_coords) not in visited:
                            builder.append( (left, left_coords, 0) )

                    if right:
                        if (right, right_coords) not in visited:
                            builder.append( (right, right_coords, 0) )
                        val = (right, get_dir(right_coords))
                        adjacency_list[key].add(val)
    
            visited.add( (coords, dir) )

    return adjacency_list


def turn_left(x, y, direction, rows, columns):
    # (1,0) right
    # (-1,0) left
    # (0,-1) up
    # (0,1) down
    right_coords = (1,0)
    left_coords = (-1,0)
    up_coords = (0,-1)
    down_coords = (0,1)
    
    right = (x+1, y) if x+1 <= columns-1 else None
    left = (x-1, y) if x-1 >= 0 else None
    up = (x, y-1) if y-1 >= 0 else None
    down = (x, y+1) if y+1 <= rows-1 else None
    
    if direction == right_coords:
        return up, up_coords
    if direction == left_coords:
        return down, down_coords
    if direction == up_coords:
        return left, left_coords
    if direction == down_coords:
        return right, right_coords
    
def turn_right(x, y, direction, rows, columns):
    # (1,0) right
    # (-1,0) left
    # (0,-1) up
    # (0,1) down
    right_coords = (1,0)
    left_coords = (-1,0)
    up_coords = (0,-1)
    down_coords = (0,1)
    
    right = (x+1, y) if x+1 <= columns-1 else None
    left = (x-1, y) if x-1 >= 0 else None
    up = (x, y-1) if y-1 >= 0 else None
    down = (x, y+1) if y+1 <= rows-1 else None
    
    if direction == right_coords:
        return down, down_coords
    if direction == left_coords:
        return up, up_coords
    if direction == up_coords:
        return right, right_coords
    if direction == down_coords:
        return left, left_coords    
    
def continue_straight(x, y, direction, rows, columns):
    # (1,0) right
    # (-1,0) left
    # (0,-1) up
    # (0,1) down
    right_coords = (1,0)
    left_coords = (-1,0)
    up_coords = (0,-1)
    down_coords = (0,1)
    
    right = (x+1, y) if x+1 <= columns-1 else None
    left = (x-1, y) if x-1 >= 0 else None
    up = (x, y-1) if y-1 >= 0 else None
    down = (x, y+1) if y+1 <= rows-1 else None
    
    if direction == right_coords:
        return right, direction
    if direction == left_coords:
        return left, direction
    if direction == up_coords:
        return up, direction
    if direction == down_coords:
        return down, direction

def get_neighbours(x, y, direction, length, rows, columns, part2):
    neighbours = []
    
    if (part2 and length >= 4) or not part2:
        #turn left
        lcoords, ldir = turn_left(x, y, direction, rows, columns)
        if lcoords:
            neighbours.append( (lcoords, ldir, 1) )
        
        #turn right
        rcoords, rdir = turn_right(x, y, direction, rows, columns)
        if rcoords:
            neighbours.append( (rcoords, rdir, 1) )
        
    #straight if < 3 for part1 or if < 10 for part2
    if part2:
        max_length = 10
    else:
        max_length = 3
    if length < max_length:
        scoords, sdir = continue_straight(x, y, direction, rows, columns)
        if scoords:
            neighbours.append( (scoords, sdir, length+1) )
    
    return neighbours

def dijkstra_atmost_steps(graph, start, target, part2=False): 
    rows = len(graph)
    columns = len(graph[0])  
    visited = set()
        
    right_coords = (1,0)
    down_coords = (0,1)

    queue = PriorityQueue()
    # total heat loss, coordinates, direction, length in that direction
    queue.put( (0, start, right_coords, 0) )
    queue.put( (0, start, down_coords, 0) )
    
    while not queue.empty():

        # get next lowest heat cost point
        (heat_cost, coords, direction, length) = queue.get()
        (x,y) = coords
        
        # check if we have already visited this pair of coordinates, direction
        if (coords, direction, length) in visited:
            continue
        visited.add( (coords, direction, length) )
        
        #grid[y][x] = '#'
        #print("Dequeue:", (heat_cost, coords, direction, length))
        # if we reached our target then it must be the lowest heat cost
        if coords == target:
            print("puff", coords, get_dir(direction), length, heat_cost)           
            return heat_cost
        
        # get neighbours points: turn left, turn right and go straight if possible
        neighbours = get_neighbours(x, y, direction, length, rows, columns, part2)
        
        for (n_coords, n_dir, n_length) in neighbours:

            (xx, yy) = n_coords
            # get new heat cost for the neighbour point
            heat = heat_cost + int(graph[yy][xx])
            
            #print("Checking neighbour", n_coords, "with heat", heat, "heat_cost",heat_cost)            
            # add neighbour point with corresponding heat cost to priority queue
            queue.put( (heat, n_coords, n_dir, n_length) )
    
    return heat_cost

# 654, 655 too low
# 696 too high
# 656, 670, 657
#Day 17, part 1: 686 (1.775 secs)
def day17_1(data):
    #data = read_input(2023, "17_teste")    
    result = 0  

    rows = len(data)
    columns = len(data[0])

    start = (0,0)
    end = (rows-1, columns-1)

    grid = buildMapGrid(data,withPadding=False)
    result = dijkstra_atmost_steps(grid, start, end)
    
    #adjacency_list = build_adjacency_list(rows, columns)


    AssertExpectedResult(686, result)
    return result


def day17_2(data):
    #data = read_input(2023, "17_teste")    
    result = 0  

    rows = len(data)
    columns = len(data[0])

    start = (0,0)
    end = (rows-1, columns-1)

    grid = buildMapGrid(data,withPadding=False)
    result = dijkstra_atmost_steps(grid, start, end, part2=True)
    
    #adjacency_list = build_adjacency_list(rows, columns)


    AssertExpectedResult(686, result)
    return result

#endregion

#region ##### Day 18 #####


def follow_dig_plan(dig_plan, dig_plan_instructions):    
    rows = len(dig_plan)
    columns = len(dig_plan[0])
    (x,y) = (columns//3,rows//3)
    dig_plan[y][x] = '#'
    points = [(x,y)]
    
    for direction, val in dig_plan_instructions:
        if direction == 'R':
            for i in range(val):
                x+=1
                dig_plan[y][x] = '#'
                points.append((x,y))
        elif direction == 'L':
            for i in range(val):
                x-=1
                dig_plan[y][x] = '#'
                points.append((x,y))
        elif direction == 'U':
            for i in range(val):
                y-=1
                dig_plan[y][x] = '#'
                points.append((x,y))
        elif direction == 'D':
            for i in range(val):
                y+=1
                dig_plan[y][x] = '#'
                points.append((x,y))
    return dig_plan, points


def day18_1(data):
    data = read_input(2023, "18_teste")

    dig_plan_instruction = []
    rows = 10
    columns = 10

    for line in data:
        r = parse("{} {} ({})", line)
        direction = r[0]
        val = int(r[1])
        dig_plan_instruction.append((direction, val))
        if direction in ['D']:
            rows+=val
        elif direction == 'R':
            columns+=val
    
    # original solution, works but is a bit slow
    '''
    dig_plan = build_empty_grid(rows, columns, '.')
    dig_plan, points = follow_dig_plan(dig_plan, dig_plan_instruction)
    p = Polygon(points)
    
    result = 0
    for y in range(rows):
        for x in range(columns):
            point = Point(x,y)
            if p.contains(point):
                result += 1

    result += p.boundary.length
    '''
    
    # second approach is smarter in building the polygon :sweat
    (x,y) = (columns//3,rows//3)
    points = write_down_dig_plan(dig_plan_instruction,x,y)
    polygon = Polygon(points)
       
    # it restricts the generated polygon in a box to reduce the search space to check for interior points, still not optimal
    '''
    box = polygon.envelope
    result = 0
    minx, miny, maxx, maxy = int(box.bounds[0]), int(box.bounds[1]), int(box.bounds[2]), int(box.bounds[3])
    for y in range(miny, maxy+1):
        for x in range(minx, maxx+1):
            point = Point(x,y)
            if polygon.contains(point):
                result += 1
    result += polygon.boundary.length
    '''
    
    # third approach seems to be the only viable option for part 2 :\ 
    # which is basically a derivation of shoelace formula (Pick's theorem I think)
    result = polygon.area + polygon.length//2+1

    AssertExpectedResult(0, result)
    return result

def get_dir_hex(hex):
    if hex == 0:
        return 'R'
    elif hex == 1:
        return 'D'
    elif hex == 2:
        return 'L'
    elif hex == 3:
        return 'U'

def write_down_dig_plan(dig_plan_instructions,x,y):
    points = []
    
    for direction, val in dig_plan_instructions:
        if direction == 'R':
            points.append(Point(x,y))
            x+=val
            points.append(Point(x,y))
        elif direction == 'L':
            points.append(Point(x,y))
            x-=val
            points.append(Point(x,y))
        elif direction == 'U':
            points.append(Point(x,y))
            y-=val
            points.append(Point(x,y))
        elif direction == 'D':
            points.append(Point(x,y))
            y+=val
            points.append(Point(x,y))
    return points

def day18_2(data):
    #data = read_input(2023, "18_teste")    
    result = 0

    dig_plan_instructions = []
    rows = 1000
    columns = 1000

    for line in data:
        r = parse("{} {} ({})", line)
        hex = r[2]
        val = int(hex[1:6],16)
        direction = get_dir_hex(int(hex[6]))
        
        dig_plan_instructions.append((direction, val))
        if direction in ['D']:
            rows+=val
        elif direction == 'R':
            columns+=val
    
 
    (x,y) = (columns//3,rows//3)
    points = write_down_dig_plan(dig_plan_instructions,x,y)
    polygon = Polygon(points)
 
    # this approach does not work for such a big search space even for the sample data
    '''
    box = polygon.envelope    
    
    result = 0
    minx, miny, maxx, maxy = int(box.bounds[0]), int(box.bounds[1]), int(box.bounds[2]), int(box.bounds[3])
    for y in range(miny, maxy+1):
        for x in range(minx, maxx+1):
            point = Point(x,y)
            if polygon.contains(point):
                result += 1
    
    result += polygon.boundary.length
    '''
    
    result = polygon.area + polygon.length//2+1
    
             
    AssertExpectedResult(0, result)
    return result

#endregion


#region ##### Day 19 #####

def start_workflows(workflows, parts, start):
    accepted = []
    dest = start
    for part in parts:
        dest = start
        while dest != 'A' and dest != 'R':
            rules = workflows[dest]
            for rule in rules:
                #print(part)
                r = rule.split(":")
                if len(r) > 1:
                    cond = r[0]
                    rule_eval = eval(cond, {'x': part.x, 'm': part.m, 'a': part.a, 's': part.s } )
                    #print(cond,"for part", part, "is", t)
                    if rule_eval:
                        dest = r[1]
                        break
                else:
                    dest = r[0]
            if dest == 'A':
                accepted.append(part)

    #print("accepted",accepted)
    rating = sum([part.x + part.m + part.a + part.s for part in accepted])
    return rating
        


def day19_1(data):
    #data = read_input(2023, "19_teste")    
    result = 0  

    Part = namedtuple("Part",['x','m','a','s'])
    workflows = dict()
    parts = []

    i = data.index('')
    workflows_data = data[:i]
    parts_data = data[i+1:]
    
    for line in workflows_data:
        wk = parse("{name}{{{rules}}}", line)
        workflows[wk['name']] = wk['rules'].split(",")
        
    for line in parts_data:
        wk = parse("{{x={x:d},m={m:d},a={a:d},s={s:d}}}", line)
        parts.append(Part(wk['x'], wk['m'], wk['a'], wk['s'] ))

    start = 'in'
    result = start_workflows(workflows, parts, start)
    
    AssertExpectedResult(342650, result)
    return result

def print_xmas(r,x,m,a,s, just_result = False):
    if not just_result:
        print("rule:", r)
        print("x:",len(x))
        print("m:",len(m))
        print("a:",len(a))
        print("s:",len(s))
    
    res = len(x)*len(m)*len(a)*len(s)
    r = "{:,}".format(res)
    print("combinations:",r)


def evaluate_workflow_condition(cond,xx,mm,aa,ss, negate_condition = False):
    xxx = xx
    mmm = mm 
    aaa = aa 
    sss = ss
    
    next_x = xxx
    next_m = mmm 
    next_a = aaa 
    next_s = sss
    neg_cond = cond
    

    if '>' in cond:
        neg_cond = neg_cond.replace('>','<=')
    else:
        neg_cond = neg_cond.replace('<','>=')
    
    #print("Evaluating condition", cond)
    if 'x' in cond:
        xxx = [x for x in xx if eval(cond)]
        next_x = [x for x in xx if eval(neg_cond)]
        #print("x changed?",len(xxx)!=len(next_x))
    elif 'm' in cond:
        mmm = [m for m in mm if eval(cond)]
        next_m = [m for m in mm if eval(neg_cond)]
        #print("m changed?",len(mmm)!=len(next_m))
    elif 'a' in cond:
        aaa = [a for a in aa if eval(cond)]
        next_a = [a for a in aa if eval(neg_cond)]
        #print("a changed?",len(aaa)!=len(next_a))
    elif 's' in cond:
        sss = [s for s in ss if eval(cond)]
        next_s = [s for s in ss if eval(neg_cond)]
        #print("s changed?",len(sss)!=len(next_s)) 
    
    
    return (xxx, next_x), (mmm, next_m), (aaa, next_a), (sss, next_s)

def start_workflows_v2(workflows, parts, start):
    x = list(range(1,4001))
    m = list(range(1,4001))
    a = list(range(1,4001))
    s = list(range(1,4001))

    def process_workflows_rec(curr, workflows, xx, mm, aa, ss, result):
       
        #print("Curr:", curr)
        #print_xmas(curr, xx, mm, aa, ss, just_result=False)
        #print()
        
        if curr == 'A':
            res = len(xx)*len(mm)*len(aa)*len(ss)
            
            #total = "{:,}".format(sum(accepted))            
            #print_xmas(curr,xx,mm,aa,ss, just_result=True)           
            #print("total",total)
            
            return res
        elif curr == 'R':       
            return 0
        else:
            rules = workflows[curr]            
            res = 0            
                
            for rule in rules:
                r = rule.split(":")                

                if len(r) > 1:
                    cond = r[0]                   

                    # returns pairs of ranges for cond as True and cond as False for each variable      
                    (xxx, next_x), (mmm, next_m), (aaa, next_a), (sss, next_s) = evaluate_workflow_condition(cond, xx, mm, aa, ss)
                    
                    if xxx or mmm or aaa or sss:                
                        dest = r[1]       
                        #print("Processing rule",curr,"->",dest,"with condition",cond)                 
                        #print_xmas(dest,xxx,mmm,aaa,sss) 
                        #print("---")
                        res += process_workflows_rec(dest, workflows, xxx, mmm, aaa, sss, result)
                                                
                        xxx = next_x
                        mmm = next_m 
                        aaa = next_a
                        sss = next_s
                        
                        # if we applied this rule with cond as True then the next rule for this workflow needs to consider this cond as False
                        
                        #print("curr:", curr)
                        if xxx: 
                            #print("x changed?",len(xxx)!=len(xx))
                            xx = next_x
                        
                        if mmm:
                            #print("m changed?",len(mmm)!=len(mm))
                            mm = next_m

                        if aaa:
                            #print("a changed?",len(aaa)!=len(aa))
                            aa = next_a

                        if sss:
                            #print("s changed?",len(sss)!=len(ss))
                            ss = next_s
                        
                        #print_xmas(curr,xxx,mmm,aaa,sss)
                        
                        
                else:
                    dest = r[0]
                    #print("Processing rule",curr,"->",dest,"with no condition")                 
                    #print_xmas(dest,xxx,mmm,aaa,sss) 
                    #print("---")
                   
                    res += process_workflows_rec(dest, workflows, xxx, mmm, aaa, sss, result)

            return res
           
    result = process_workflows_rec(start, workflows, x, m, a, s, 0)
    #result = "{:,}".format(result)
    return result
    

    #return accepted

def day19_2(data):
    #data = read_input(2023, "19_teste")    
    result = 0    
             
    Part = namedtuple("Part",['x','m','a','s'])
    workflows = dict()
    parts = []

    i = data.index('')
    workflows_data = data[:i]
    parts_data = data[i+1:]
    
    for line in workflows_data:
        wk = parse("{name}{{{rules}}}", line)
        workflows[wk['name']] = wk['rules'].split(",")
        
    for line in parts_data:
        wk = parse("{{x={x:d},m={m:d},a={a:d},s={s:d}}}", line)
        parts.append(Part(wk['x'], wk['m'], wk['a'], wk['s'] ))

    start = 'in'
    result = start_workflows_v2(workflows, parts, start)
        
    
    AssertExpectedResult(130303473508222, result)
    return result

#endregion


#region ##### Day 20 #####


def update_target_modules(module, targets, pulse, conjunctions):

    for mod in targets:       
        if mod in conjunctions:
            conjunctions[mod].update({module: pulse})            
            
    return conjunctions          


def process_modules(press_button_times, modules, flip_flops, conjunctions, modules_needed = []):
    
    low_pulses = 0
    high_pulses = 0
    found = []
    
    for i in range(press_button_times):
        queue = [('broadcaster', False)]

        while len(queue) > 0 :
            # process next module
            mod, pulse = queue.pop(0)
            
            if pulse:
                high_pulses+=1
            else:
                low_pulses+=1

            #print()
            #pulse_type = 'High' if pulse else 'Low'    
            #print("Processing module", mod,"with current state:", flip_flops, conjunctions)
            
            if mod in modules:
                targets = modules[mod]
                
                #pulse_type = 'High' if pulse else 'Low'                
                #print(mod,"received pulse",pulse_type)
            
                if mod == 'broadcaster':
                    queue = queue + [(target, pulse) for target in targets]
                    
                    #print(mod,"--",pulse_type,"-->", targets)

                elif mod in flip_flops: 
                    turned_on = flip_flops[mod]                    
                    
                    if not pulse: # if low pulse               
                        flip_flops[mod] = not turned_on #switch
                        if not turned_on:
                            pulse = not pulse # send high pulse
                       
                        #updates targets
                        conjunctions = update_target_modules(mod, targets, pulse, conjunctions)       
                        queue = queue + [(target, pulse) for target in targets]
                        
                        #pulse_type = 'High' if pulse else 'Low'
                        #print(mod,"--", pulse_type,"-->", targets)
                    if pulse and mod in modules_needed:
                        modules_needed.remove(mod)
                        found.append(i+1)
                        
                
                elif mod in conjunctions:
                    inputs = conjunctions[mod]
                    
                    #print("conjunction",mod,"has inputs state:", inputs)                    
                    
                    pulse = True 
                    for k in inputs:
                        p = inputs[k]
                        pulse = pulse and p
                    
                    # it remembered high pulses for ALL inputs, change to Low
                    pulse = not pulse
                    
                    conjunctions = update_target_modules(mod, targets, pulse, conjunctions)       
                    queue = queue + [(target, pulse) for target in targets]
                    
                    #pulse_type = 'High' if pulse else 'Low'
                    #print(mod,"--", pulse_type,"-->", targets)
           
        #print()

    print("low:", low_pulses)
    print("high:", high_pulses)
    print(modules_needed)
    print(found)
    r = 1
    for f in found:
     r = lcm(r,2)
    print("lcm:",r)
    return low_pulses*high_pulses


def parse_day20_data(data):
    modules = {'button': ['broadcaster']}
    flip_flops = {}
    conjunctions = {}

    for line in data:
        mods = parse("{} -> {}", line)
        module = mods[0]
        targets = mods[1]

        if module != 'broadcaster':
            mod_type = module[0]
            module = module[1:]
            if mod_type == '%':
                # turned on 
                flip_flops[module] = False
            elif mod_type == '&':   
                conjunctions[module] = dict()

        if module not in modules:
            modules[module] = []
        modules[module] = [target.strip() for target in targets.split(',')]
    
    for k in modules.keys():
        target = modules[k]
        for c in conjunctions.keys():
            if c in target:
                conjunctions[c].update({k : False})
    
    return modules, flip_flops, conjunctions

def day20_1(data):
    #data = read_input(2023, "20_teste")    
    result = 0     
    modules, flip_flops, conjunctions = parse_day20_data(data)

    #print(flip_flops)
    #print(conjunctions)
    #print(modules)
    
    press_button_times = 1000
    result = process_modules(press_button_times, modules, flip_flops, conjunctions)


    AssertExpectedResult(703315117, result)
    return result


def day20_2(data):
    #data = read_input(2023, "20_teste")    
    result = 0    
    
    modules, flip_flops, conjunctions = parse_day20_data(data)

    modules_needed = list(conjunctions['cs'].keys())
    
    press_button_times = 100000000
    result = process_modules(press_button_times, modules, flip_flops, conjunctions, modules_needed)
             
    AssertExpectedResult(0, result)
    return result

#endregion

#region ##### Day 21 #####

def bfs2(graph, root, s, compute_distances = False):
    path=[]
    if compute_distances:
        visited, queue = set(), deque([(root, 0)])
    else:
        visited, queue = set(), deque([root])
    visited.add(root)

    steps = 0
    while queue:

        # Dequeue a vertex from queue
        if compute_distances:
            (vertex, steps) = queue.popleft()
        else:
            vertex = queue.popleft() 

        path.append((vertex, steps)) if compute_distances else path.append(vertex)
        #print(str(vertex) + ' ', end='')

        # If not visited, mark it as visited, and
        # enqueue it
        for neighbour in graph[vertex]:
            if steps > s:
                return path
            #if neighbour not in visited:
            visited.add(neighbour)
            if compute_distances:
                queue.append((neighbour, steps + 1))
            else:
                queue.append(neighbour)

    return path


# 7522 7259 high
# 1045 low
def day21_1(data):
    data = read_input(2023, "21_teste")    
    result = 0  

    grid = buildMapGrid(data,'.', withPadding=False)
    graph = buildGraphFromMap(grid, noPadding=True)

    rows = len(grid)
    columns = len(grid[0])
    start = (0,0)
    
    for y in range(rows):
        for x in range(columns):
            if grid[y][x] == 'S':
                start = (x,y)
                break

    
    steps = 64
    p = bfs2(graph,start,steps,compute_distances=True)
    #print(p)
    
    l = set()
    for coords, s in p:
        #print(coords,s)
        if s == steps:
            l.add(coords)
            l.union(set(graph[coords]))
    
    result = len(l)
    #print(p)

    AssertExpectedResult(0, result)
    return result


def day21_2(data):
    data = read_input(2023, "21_teste")    
    result = 0    
             
    AssertExpectedResult(0, result)
    return result

#endregion

#region ##### Day 22 #####

# taken from someone's solution, I was struggling with this math :'(
def intersects(brick, test_brick):
    _,(p_min,p_max) = brick
    _,(o_min,o_max) = test_brick
    
    """Return if brick intersects with test_brick"""
    if (p_max.x < o_min.x or
        p_min.x > o_max.x):
        return False        # doesn't intersect in x dimension
    if (p_max.y < o_min.y or
        p_min.y > o_max.y):
        return False        # doesn't intersect in y dimension
    return True


def brick_falling_at_z(brick, highest_z, z_bricks, init_z):
    l,(pmin,pmax) = brick
    delta = pmax.z - pmin.z
    for curr_z in range(init_z, init_z+delta+1):
        #print("Brick", l,"fell on level", curr_z)
        z_bricks[curr_z].add(brick)
        if curr_z > highest_z:
            highest_z = curr_z
    
    return z_bricks, highest_z

def fall_bricks(bricks):
    z_bricks = defaultdict(set)
    brick_is_supported_by = defaultdict(set)
    brick_supports = defaultdict(set)
    highest_z = 0
    
    while len(bricks) > 0:
        (z_index, l, pmin, pmax) = bricks.pop(0)
        brick = l,(pmin,pmax)

        # ensures we can count bricks that do not support another bricks (including the ones on top)
        if brick not in brick_supports.keys():
            brick_supports[brick] = set()

        if z_index == 1:            
            z_bricks[1].add(brick)
            highest_z = 1

            # when a bricks lands, fill z_bricks with brick coordinates for all its z coordinate values
            z_bricks, highest_z = brick_falling_at_z(brick, highest_z, z_bricks, 1)     
            #print("Brick", l,"fell on the ground")
        else: 
            
            same_level = True
            zz = z_index
            keep_falling = True
            
            while keep_falling:
                 
                bricks_below = z_bricks[zz]
                while len(bricks_below) == 0:
                    zz-=1
                    bricks_below = z_bricks[zz]
                #print(l,"found bricks at level", zz)                    
                same_level = True
                
                for ll, (omin, omax) in bricks_below:
                    
                    #print("Checking bricks below Brick",l,"for level",zz)               
            
                    brick_below = ll, (omin, omax)
                    if intersects(brick, brick_below):                
                        #print("Brick",ll,"intersects with brick",l)                    
                        same_level = False
                        brick_supports[brick_below].add(brick)
                        brick_is_supported_by[brick].add(brick_below)
                    #else:
                        #print("Brick",ll,"DOES NOT intersect with brick",l)
                
                #if same_level is True it means it did not fall into a brick on this level, we need to keep falling
                if same_level:
                    keep_falling = True
                    zz -= 1
                else:
                    keep_falling = False
                
                if not same_level or zz == 0:
                    keep_falling = False

            
            init_z = -1
            if same_level:
                #print("Brick", l,"fell on level", zz)
                init_z = zz
            else:
                #print("Brick", l,"fell on level", zz+1)
                init_z = zz + 1
            
            # when a bricks lands, fill z_bricks with brick coordinates for all its z coordinate values
            z_bricks, highest_z = brick_falling_at_z(brick, highest_z, z_bricks, init_z+1)           
      
                    
    return brick_supports, brick_is_supported_by
            


def check_bricks_to_desintegrate(brick_supports, brick_is_supported_by):
    count = 0
    bricks_to_desintegrate = set()
    
    for brick in brick_supports.keys():                
        bricks_supported = brick_supports[brick]        
        can_desintegrate = True

        for supported_brick in bricks_supported:
            copy_supported_by = [b for b in brick_is_supported_by[supported_brick] if b != brick]
            
            if len(copy_supported_by) == 0:
                can_desintegrate = False                
                #break
                
        if can_desintegrate:
            #print("Brick", brick, "can be desintegrated")
            count +=1
            bricks_to_desintegrate.add(brick)
    
    bricks_that_are_needed = [b for b in brick_supports.keys() if b not in bricks_to_desintegrate]
    #print(len(bricks_that_are_needed))
    
    result_part2 = 0
    for brick in bricks_that_are_needed:
        l,(pmin,pmax) = brick
        bricks_that_might_fall = dfs(brick_supports, brick)
        bricks_that_might_fall = [b for b in bricks_that_might_fall if b != brick]

        #for ll,_ in bricks_that_might_fall:
            #print(l,"might make brick",ll,"fall")

        #copy_bricks_that_will_fall = [b for b in bricks_that_will_fall]

        bricks_that_will_fall = set()
        for brick_might_fall in bricks_that_might_fall:

            copy_supported_by = [b for b in brick_is_supported_by[brick_might_fall] if b != brick]
            #copy_supported_by2 = [b for b in brick_is_supported_by[brick_might_fall] if b != brick]

            will_fall = True
            for  b in copy_supported_by:
                if b not in bricks_that_might_fall:
                    will_fall = False

            #if len(copy_supported_by) == 0:
            if will_fall:
                bricks_that_will_fall.add(brick_might_fall)
            
        #print(l,"will make bricks",[l for l,_ in bricks_that_will_fall], "fall")
        result_part2 += len(bricks_that_will_fall)

    return count, result_part2
        
def print_debug_bricks(brick_supports, brick_is_supported_by):
    for k in brick_supports.keys():   
        (l, _) = k
        ll = brick_supports[k]
        print(l,"supports", ll)
    
    for k in brick_is_supported_by.keys():  
        (l, _) = k
        ll = brick_is_supported_by[k] 
        print(l,"is supported by",ll)    
    


import uuid

def read_bricks(data):
    bricks = []
    Point = namedtuple('Point',['x','y','z'])
    
    i = 0
    for line in data:
        point_range = parse('{min_x},{min_y},{min_z}~{max_x},{max_y},{max_z}', line)
        min_x = int(point_range['min_x'])
        min_y = int(point_range['min_y'])
        min_z = int(point_range['min_z'])
        max_x = int(point_range['max_x'])
        max_y = int(point_range['max_y'])
        max_z = int(point_range['max_z'])
        
        label = str(uuid.uuid4())
        label = chr(ord('A') + i)
        i+=1
        
        # order ascending by z coordinate usinf priority queue    
        bricks.append( (min_z, label, Point(min_x, min_y, min_z), Point(max_x, max_y, max_z)  ) )
    
    # order ascending by z coordinate
    bricks.sort(key=lambda points: points[2].z)
    return bricks


# too low 113
# too high 471 
# 463 correct answer
def day22_1(data):
    #data = read_input(2023, "22_teste")    
    result = 0  
    bricks = []    
    bricks = read_bricks(data)
        
    brick_supports, brick_is_supported_by = fall_bricks(bricks)
    result, _ = check_bricks_to_desintegrate(brick_supports, brick_is_supported_by)    

    #print_debug_bricks(brick_supports, brick_is_supported_by)
    
    AssertExpectedResult(463, result)
    return result


#184, 82938 too low
# 89727
def day22_2(data):
    #data = read_input(2023, "22_teste")    
    result = 0  
    bricks = []    
    bricks = read_bricks(data)
        
    brick_supports, brick_is_supported_by = fall_bricks(bricks)
    _, result = check_bricks_to_desintegrate(brick_supports, brick_is_supported_by)

    #print(bricks_that_are_needed)
    
    #print_debug_bricks(brick_supports, brick_is_supported_by)
             
    AssertExpectedResult(0, result)
    return result

#endregion

#region ##### Day 23 #####


def build_graph_from_hike_map(map, part2=False):
    graph = {}
    sizeX = len(map[0])
    sizeY = len(map)

    for y in range(sizeY):
        for x in range(sizeX):

            east = (x+1, y)
            west = (x-1, y)
            north = (x, y-1)
            south = (x, y+1)

            heast = True
            hwest = True 
            hnorth = True
            hsouth = True
            empty_cells = ['.', '<', '>', '^', 'v']
            
            if not part2:
                if map[y][x] == '<':
                    heast = False 
                    hnorth = False
                    hsouth = False
                elif map[y][x] == '>':
                    hwest = False 
                    hnorth = False
                    hsouth = False
                elif map[y][x] == '^':
                    hwest = False 
                    heast = False
                    hsouth = False
                elif map[y][x] == 'v':
                    hwest = False 
                    heast = False
                    hnorth = False
            
            neighbours = []
            if map[y][x] in empty_cells:
                if 0 <= east[0] < sizeX and 0 <= east[1] < sizeY:
                    if heast:
                        if map[east[1]][east[0]] in empty_cells:
                            neighbours.append(east)
                
                if 0 <= west[0] < sizeX and 0 <= west[1] < sizeY:                        
                    if hwest:
                        if map[west[1]][west[0]] in empty_cells:
                            neighbours.append(west)
                
                if 0 <= north[0] < sizeX and 0 <= north[1] < sizeY: 
                    if hnorth:
                        if map[north[1]][north[0]] in empty_cells:
                            neighbours.append(north)
                
                if 0 <= south[0] < sizeX and 0 <= south[1] < sizeY:
                    if hsouth: 
                        if map[south[1]][south[0]] in empty_cells:
                            neighbours.append(south)
            
            graph[(x,y)] = neighbours
    return graph

def find_longest_path(graph, start, end, path=[]):
        path = path + [start]

        if start == end:
            return path
        
        if start not in graph:
            return None
        
        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = find_longest_path(graph, node, end, path)
                if newpath:
                    if not shortest or len(newpath) > len(shortest):
                        shortest = newpath
        
        if start in path:
            path.remove(start)

        return shortest


def day23_1(data):
    data = read_input(2023, "23_teste")    
    result = 0  

    hike_map = buildMapGrid(data, initValue='.', withPadding=False)
    graph = build_graph_from_hike_map(hike_map)

    for k in graph.keys():
        print(k,"->", graph[k])

    start = (1,0)
    end = (len(hike_map[0])-2, len(hike_map)-1)

    sys.setrecursionlimit(10000)
    path = find_longest_path(graph, start, end)

    result = len(path)-1

    AssertExpectedResult(2230, result)
    return result


def DFSv3(G,v,seen=None,path=None):
    if seen is None: seen = []
    if path is None: path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(DFSv3(G, t, seen[:], t_path))
    return paths



def day23_2(data):
    #data = read_input(2023, "23_teste")    
    result = 0   

    hike_map = buildMapGrid(data, initValue='.', withPadding=False)
    graph = build_graph_from_hike_map(hike_map, part2=True)

    #for k in graph.keys():
    #    print(k,"->", graph[k])


    start = (1,0)
    end = (len(hike_map[0])-2, len(hike_map)-1)

    sys.setrecursionlimit(10000)
    path = find_longest_path(graph, start, end)
    result = len(path)-1 
    
    # Run DFS, compute metrics
    #all_paths = DFSv3(graph, start)
    #max_len   = max(len(p) for p in all_paths)
    #max_paths = [p for p in all_paths if len(p) == max_len]

    
    print("Longest Path Length:")
    #print(max_len)
    
             
    AssertExpectedResult(0, result)
    return result

#endregion

#region ##### Day 24 #####

#too low 7787
# too high 11161
def day24_1(data):
    #data = read_input(2023, "24_teste")    
    result = 0  
    hailstones = []

    for line in data:
        hs = parse("{px}, {py}, {pz} @ {vx}, {vy}, {vz}", line)
        hailstone = ( (int(hs['px'].strip()), int(hs['py'].strip()), int(hs['pz'].strip())), (int(hs['vx'].strip()), int(hs['vy'].strip()), int(hs['vz'].strip()) ) )
        hailstones.append(hailstone)

    bounds = (200000000000000, 400000000000000)
    #bounds = (7, 27)
    coords = [ Point(bounds[0], bounds[0]), Point(bounds[0],bounds[1]), Point(bounds[1],bounds[0]), Point(bounds[1],bounds[1])]
    test_area = box(200000000000000, 200000000000000, 400000000000000, 400000000000000)
    #test_area = Polygon(coords)
    
    time_forward = 200000000000000 + 400000000000000
 
    pairs = list(itertools.combinations(hailstones, 2))
    for h1, h2 in pairs:
        pos1, vel1 = h1 
        pos2, vel2 = h2
        x1,y1,_ = pos1
        x2,y2,_ = pos2
        vx1,vy1,_ = vel1
        vx2,vy2,_ = vel2

        h1_start = (x1,y1)
        h2_start = (x2,y2)
 
        x1 += (vx1 * time_forward)
        x2 += (vx2 * time_forward)
        y1 += (vy1 * time_forward)
        y2 += (vy2 * time_forward)

        h1_end = (x1,y1)
        h2_end = (x2,y2)

        h1_line = LineString([h1_start, h1_end])
        h2_line = LineString([h2_start, h2_end])
    
        intersection_point = h1_line.intersection(h2_line)
        
        if test_area.contains(intersection_point):
            result += 1
        


    AssertExpectedResult(11098, result)
    return result


def day24_2(data):
    data = read_input(2023, "24_teste")    
    result = 0  
    hailstones = []

    for line in data:
        hs = parse("{px}, {py}, {pz} @ {vx}, {vy}, {vz}", line)
        hailstone = ( (int(hs['px'].strip()), int(hs['py'].strip()), int(hs['pz'].strip())), (int(hs['vx'].strip()), int(hs['vy'].strip()), int(hs['vz'].strip()) ) )
        hailstones.append(hailstone)

  
    AssertExpectedResult(0, result)
    return result

#endregion

#region ##### Day 25 #####



# 519672 too low
# 520406 too high
# 520380
# kzh -> rks , ddc -> blb, dgt-> tnz
def day25_1(data):
    #data = read_input(2023, "25_teste")    
     
    components = dict()

    for line in data:
        r = parse("{}: {}", line)
        component = r[0].strip()
        cps = [c.strip() for c  in r[1].split(' ')]

        if component in components:
            components[component] = components[component] + cps 
        else:
            components[component] = cps 
        

    dot = graphviz.Digraph('aoc', filename='aoc.gv',
                     node_attr={'color': 'lightblue2', 'style': 'filled'})
    dot.attr(size='6,6')
    graph = nx.Graph()    

    for key in components.keys():
        graph.add_node(key)
        for c in components[key]:
            # kzh -> rks , ddc -> gqm, dgt-> tnz
            graph.add_node(c)
            #if (key != 'kzh' and c != 'rks') and (key != 'ddc' and c != 'gqm') and  (key != 'dgt' and c != 'tnz'):
            graph.add_edge(key,c)
            dot.edge(key, c)

    #dot.view()

    print(nx.number_connected_components(graph))

    for u,v in nx.minimum_edge_cut(graph):
        graph.remove_edge(u,v)
        
    print(nx.number_connected_components(graph))
    
    result = 1
    for c in nx.connected_components(graph):
        result *= len(c)

    AssertExpectedResult(520380, result)
    return result


#endregion

if __name__ == "__main__":
    # override timeout
    main(sys.argv, globals(), AOC_EDITION_YEAR, 28800)

