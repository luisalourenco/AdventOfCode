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
                    numbers[(xx,yy)] = number
                numbers[(-1,-1)] = []
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

    res = set()
    ratio = 0
    for (x,y) in gears:

        n = numbers_positions.get((x-1,y))
            
        if n is not None:
            res.add(n)
        
        n = numbers_positions.get((x,y-1))
        if n is not None:
            res.add(n)
        
        n = numbers_positions.get((x+1,y))
        if n is not None:
            res.add(n)
        
        n = numbers_positions.get((x,y+1))
        if n is not None:
            res.add(n)

        n = numbers_positions.get((x-1,y-1))
        if n is not None:
            res.add(n)
        
        n = numbers_positions.get((x+1,y+1))
        if n is not None:
            res.add(n)

        n = numbers_positions.get((x+1,y-1))
        if n is not None:
            res.add(n)

        n = numbers_positions.get((x-1,y+1))
        if n is not None:
            res.add(n)

        gears_numbers[(x,y)] = res
        if len(res) == 2:
            res = list(res)
            ratio += res[0]*res[1]
            print("numbers for gear",x,y,"are",res)
       
       
        res = set()
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
    rows = len(engine_map)
    columns = len(engine_map[0]) 
    for y in range(rows): 
        for x in range(columns):
            n = numbers_positions.get((x,y))
            if n is not None:
                print("(",x,y,"):",n)
        print()
    #####    


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

# all labels are equal
def is_five_of_a_kind(hand):
    return len(set(hand)) == 1

def is_four_of_a_kind(hand):
    cards = list(set(hand))
    if len(cards) == 2:
        return list(hand).count(cards[0]) == 4 or list(hand).count(cards[1]) == 4    
    else:
        return False

def is_full_house(hand):
    cards = list(set(hand))
    if len(cards) == 2:
        return (list(hand).count(cards[0]) == 3 and list(hand).count(cards[1]) == 2) or (list(hand).count(cards[0]) == 2 and list(hand).count(cards[1]) == 3) 
    else:
        return False

def is_three_of_a_kind(hand):
    cards = list(set(hand))
    if len(cards) == 3:
        return list(hand).count(cards[0]) == 3 or list(hand).count(cards[1]) == 3 or list(hand).count(cards[2]) == 3
    else:
        return False

def is_two_pair(hand):
    cards = list(set(hand))
    if len(cards) == 3:
        return (list(hand).count(cards[0]) == 2 and list(hand).count(cards[1]) == 2 and list(hand).count(cards[2]) == 1) or (list(hand).count(cards[0]) == 2 and list(hand).count(cards[1]) == 1 and list(hand).count(cards[2]) == 2) or (list(hand).count(cards[0]) == 1 and list(hand).count(cards[1]) == 2 and list(hand).count(cards[2]) == 2)
    else:
        return False

def is_pair(hand):
    cards = list(set(hand))

    if len(cards) == 4:
        return (list(hand).count(cards[0]) == 2 and list(hand).count(cards[1]) == 1 and list(hand).count(cards[2]) == 1 and list(hand).count(cards[3]) == 1) or (list(hand).count(cards[0]) == 1 and list(hand).count(cards[1]) == 2 and list(hand).count(cards[2]) == 1 and list(hand).count(cards[3]) == 1) or (list(hand).count(cards[0]) == 1 and list(hand).count(cards[1]) == 1 and list(hand).count(cards[2]) == 2 and list(hand).count(cards[3]) == 1) or (list(hand).count(cards[0]) == 1 and list(hand).count(cards[1]) == 1 and list(hand).count(cards[2]) == 1 and list(hand).count(cards[3]) == 2)
    else:
        return False

def is_high_card(hand):
    return len(set(hand)) == 5

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


def compare_hands(hand1, hand2):
    if get_type_strenght(hand1) < get_type_strenght(hand2):
        return -1
    elif get_type_strenght(hand1) > get_type_strenght(hand2):
        return 1
    else: # same type strength
        return compare_cards_in_hands(hand1, hand2)

# 250119709 too low
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
    
  
    AssertExpectedResult(0, result)
    return result

#endregion

if __name__ == "__main__":
    # override timeout
    main(sys.argv, globals(), AOC_EDITION_YEAR, 900)

