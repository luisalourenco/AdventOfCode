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
from shapely import Polygon, Point, contains_xy
from itertools import combinations
import networkx as nx

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
from common.graphUtils import dijsktra, printGraph, find_all_paths, find_path, find_shortest_path, find_shortest_pathOptimal, bfs, dfs, Graph, hashable_lru, BFS_SP
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


def aux(replaced, condition):
    
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
def check_condition_rec(springs, condition, counter, replaced):    
    #print("call with",springs, condition, counter)
    
    if len(springs) == 0:
        return aux(replaced, condition)
        
    elif springs[0] == '#':        
        return check_condition_rec(springs[1:], condition, counter, replaced + [springs[0]])
        
    elif springs[0] == '.':       
        return check_condition_rec(springs[1:], condition, counter, replaced + [springs[0]]) 
       
    elif springs[0] == '?':
        #print("Replacing ?")        
        return check_condition_rec(springs[1:], condition, counter, replaced + ['#']) + check_condition_rec(springs[1:], condition, counter, replaced + ['.'])
    
def check_condition_v2(line):

    data_split = line.split(" ")
    s = list(data_split[0])
    c = ints(data_split[1].split(','))

    res = check_condition_rec(s, c, 0, [])
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
    records = []

    for line in data:
        data_split = line.split(" ")
        records.append((list(data_split[0]), ints(data_split[1].split(','))))
    
    #print(records)
    #result = naive_solution(records)

    #for (springs, condition) in records:
    for line in data:        
        result += check_condition_v2(line)#springs, condition)
        
    #print(size)
    AssertExpectedResult(7251, result)
    return result

def day12_2(data):
    data = read_input(2023, "12_teste")    
    result = 0    
             
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

def day17_1(data):
    data = read_input(2023, "17_teste")    
    result = 0  

    rows = len(data)
    columns = len(data[0])

    #g = buildGraphFromMap(data, withWeights=True, noPadding=True)
    start = (0,0)
    end = (rows-1, columns-1)

    grid = buildMapGrid(data,withPadding=False)

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

    print()
    print("Generated graph:")
    graph = Graph()
    for k in adjacency_list.keys():
        print(k,":", adjacency_list[k])
        neighbours = adjacency_list[k]
        for (n, d) in neighbours:            
            graph.add_edge(k, (n, d), int(grid[n[1]][n[0]])  )


    start = ((0, 0), 'down') 
    end1 = ((12, 12), 'down')
    end2 = ((12, 12), 'right')
    path1 = dijsktra(graph, start, end1 )
    path2 = dijsktra(graph, start, end2 )

    print()
    print("Shortest path1 found:")
    result = 0
    for v, w in zip(path1[:-1],path1[1:]):
        print(v,"->",w)
        result += graph.weights[v,w]

    print()
    print(result)
    print("Shortest path2 found:")
    result = 0
    for v, w in zip(path2[:-1],path2[1:]):
        print(v,"->",w)
        result += graph.weights[v,w]


    AssertExpectedResult(0, result)
    return result


def day17_2(data):
    data = read_input(2023, "17_teste")    
    result = 0    
             
    AssertExpectedResult(0, result)
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

def day19_1(data):
    data = read_input(2023, "19_teste")    
    result = 0  

    AssertExpectedResult(0, result)
    return result


def day19_2(data):
    data = read_input(2023, "19_teste")    
    result = 0    
             
    AssertExpectedResult(0, result)
    return result

#endregion

if __name__ == "__main__":
    # override timeout
    main(sys.argv, globals(), AOC_EDITION_YEAR, 28800)

