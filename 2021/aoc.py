# Based on template from https://github.com/scout719/adventOfCode/
# -*- coding: utf-8 -*
# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=wildcard-import
# pylint: disable=wrong-import-position
# pylint: disable=consider-using-enumerate-

import functools
import math
import os
import sys
import time
import copy
import re
import itertools 
import numpy as np
from functools import lru_cache
import operator
from itertools import takewhile
from turtle import Turtle, Screen
from math import remainder, sqrt

# UPDATE THIS VARIABLE
AOC_EDITION_YEAR = 2021

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
print(FILE_DIR)
sys.path.insert(0, FILE_DIR + "/")
sys.path.insert(0, FILE_DIR + "/../")
sys.path.insert(0, FILE_DIR + "/../../")

from common.utils import read_input, main, clear, AssertExpectedResult, ints  # NOQA: E402
from common.mapUtils import printMap, buildMapGrid, buildGraphFromMap
from common.graphUtils import printGraph, find_all_paths, find_path, find_shortest_path, find_shortest_pathOptimal, bfs, dfs, Graph, hashable_lru
from common.aocVM import HandheldMachine
from lark import Lark, Transformer, v_args
from pyformlang.cfg import Production, Variable, Terminal, CFG, Epsilon
from itertools import islice

# pylint: enable=import-error
# pylint: enable=wrong-import-position

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

WHITE_SQUARE = "█"
WHITE_CIRCLE = "•"
BLUE_CIRCLE = f"{bcolors.OKBLUE}{bcolors.BOLD}•{bcolors.ENDC}"
RED_SMALL_SQUARE = f"{bcolors.FAIL}{bcolors.BOLD}■{bcolors.ENDC}"

################# Advent of Code - 2021 Edition #################


#Day 1, part 1: 1548 (0.016 secs)
#Day 1, part 2: 1589 (0.003 secs)
def day1_1(data):
    #data = read_input(2021, "1")    
    count = 0
    currentDepth = sys.maxsize
    for line in data:
        depth = int(line)

        if depth > currentDepth:
            count += 1
        currentDepth = depth

    result = count  
    
    AssertExpectedResult(1548, result)
    return result

def day1_2(data):
    #data = read_input(2021, "1")    
    window_size = 3
    count = 0
    currentDepth = sys.maxsize

    for i in range(len(data) - window_size + 1):
        window = [int(x) for x in data[i: i + window_size]]
        depth = sum(window)
  
        if depth > currentDepth:
            count += 1
        currentDepth = depth    

    result = count  
    
    AssertExpectedResult(1589, result)
    return result   

# Day 2, part 1: 1488669 (0.050 secs)
def day2_1(data):
    #data = read_input(2021, "21")    
    horizontalPos = 0
    depth = 0
    
    for line in data:
        inputData = line.split(" ")
        move = inputData[0]
        value = int(inputData[1])

        if (move == "forward"):
            horizontalPos += value
        elif (move == "down"):
            depth += value
        elif (move == "up"):
            depth -= value

    result = horizontalPos * depth
    
    AssertExpectedResult(1488669, result)
    return result

# Day 2, part 2: 1176514794 (0.003 secs)
def day2_2(data):
    #data = read_input(2021, "21")    
    horizontalPos = 0
    depth = 0
    aim = 0

    for line in data:
        inputData = line.split(" ")
        move = inputData[0]
        value = int(inputData[1])

        if (move == "forward"):
            horizontalPos += value
            depth += (aim * value)
        elif (move == "down"):
            aim += value
        elif (move == "up"):
            aim -= value

    result = horizontalPos * depth
    
    AssertExpectedResult(1176514794, result)
    return result

# Improved version for Day 3 part 1
def alternativeDay3_1(data):
    # reads each line from data, convert each line to an int using map 
    # then convert the resulting int into a list of numbers
    bitsMatrix = [ list(map(int, line)) for line in data ]   

    gammaRate = ""
    epsilonRate = ""

    for i in range(len(bitsMatrix[0])):
        ones = [item[i] for item in bitsMatrix].count(1)
        zeros = [item[i] for item in bitsMatrix].count(0)
        #print("ones:",ones, "zeros:",zeros)
        if ones > zeros:
            gammaRate += '1'
            epsilonRate += '0'
        else:
            gammaRate += '0'
            epsilonRate += '1'

    return int(gammaRate, 2) * int(epsilonRate, 2)

def originalDay3_1(data):
    gammaRate = []
    epsilonRate = []
    bitsPosition = []

    size = 0
    for line in data:        
        bitsPosition.append( [int(n) for n in line] )
        size += 1
    
    bits = np.array(bitsPosition)
    for i in range(len(bits[0])):
        ones = list(bits[:,i]).count(1)
        zeros = size - ones
        if ones > zeros:
            gammaRate.append(1)
            epsilonRate.append(0)
        else:
            gammaRate.append(0)
            epsilonRate.append(1)

    gammaRateBin = [str(n) for n in gammaRate]
    epsilonRateBin = [str(n) for n in epsilonRate]

    g = ''.join(format(int(i,2)) for i in gammaRateBin)
    e = ''.join(format(int(i,2)) for i in epsilonRateBin)
    
    result = int(g, 2) * int(e,2)

    return result

def day3_1(data):
    #data = read_input(2021, "31")    
    result = alternativeDay3_1(data)
    
    AssertExpectedResult(2648450, result)

    return result

def findRating(bitsPosition, bits, rating):
 
    remainder = bitsPosition
    for i in range(len(bits[0])):
        bits = np.array(remainder)
        ones = list(bits[:,i]).count(1)
        zeros = list(bits[:,i]).count(0)

        #print("index",i,"ones:", ones, "zeros:",zeros, "rating",rating)
        #print(bits)
        #print()

        if(rating == 'oxygen'):
            criteria = 1
            criteria2 = 0
        else:
            criteria = 0
            criteria2 = 1

        aux = []
        if ones >= zeros:
            for bit in remainder:                 
                if bit[i] == criteria:                       
                     aux.append(bit)
        else:                
            for bit in remainder:
                if bit[i] == criteria2:               
                    aux.append(bit)
        remainder = aux
        if len(remainder) == 1:
            return remainder
    return remainder

def getNewBitsMatrix(bitsMatrix, cond, rating, position):
    getNewBitsMatrix = []
    for bit in bitsMatrix:  
        if(rating == 'oxygen'):
            criteria = 1
            criteria2 = 0
        else:
            criteria = 0
            criteria2 = 1

        if cond:      
            if bit[position] == criteria:
                getNewBitsMatrix.append(bit)
        else:
            if bit[position] == criteria2:
                getNewBitsMatrix.append(bit)       
     
    return getNewBitsMatrix

def getRating(bitsMatrix, rating):
    size = len(bitsMatrix[0])
    for i in range(size):
        ones = [item[i] for item in bitsMatrix].count(1)
        zeros = [item[i] for item in bitsMatrix].count(0)
        #print("ones:",ones, "zeros:",zeros, "criteria:", ones > zeros)  
        bitsMatrix = getNewBitsMatrix(bitsMatrix, ones >= zeros, rating, i)
        if len(bitsMatrix) == 1:
            break

    return int("".join(map(str, bitsMatrix.pop())),2)

def alternativeDay3_2(data):
    # reads each line from data, convert each line to an int using map 
    # then convert the resulting int into a list of numbers
    bitsMatrix = [ list(map(int, line)) for line in data ]   

    oxygenRating = getRating(bitsMatrix, 'oxygen')
    co2Rating = getRating(bitsMatrix, 'co2')
    
    return oxygenRating * co2Rating

def oldDay3_2(data):
    #data = read_input(2021, "31")  

    bitsPosition = []
    oxygenGeneratorRating = []
    co2ScrubberRating = []

    size = 0
    for line in data:        
        bitsPosition.append( [int(n) for n in line] )
        size += 1
    
    bits = np.array(bitsPosition)
    oxygenGeneratorRating = findRating(bitsPosition, bits, 'oxygen').pop()
    co2ScrubberRating = findRating(bitsPosition, bits, 'co2').pop()

    oxygenGeneratorRating = [str(n) for n in oxygenGeneratorRating]
    co2ScrubberRating = [str(n) for n in co2ScrubberRating]

    o = ''.join(format(int(i,2)) for i in oxygenGeneratorRating)
    c = ''.join(format(int(i,2)) for i in co2ScrubberRating)
    
    result = int(o,2) * int(c, 2) 
    
    AssertExpectedResult(2845944, result)
    return result

def day3_2(data):
    #data = read_input(2021, "31") 
        
    result = alternativeDay3_2(data)    
    AssertExpectedResult(2845944, result)
    return result

if __name__ == "__main__":
    main(sys.argv, globals(), AOC_EDITION_YEAR)

