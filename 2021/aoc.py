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
import itertools, collections
from turtle import Turtle, Screen
from math import remainder, sqrt
from collections import namedtuple


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

##### Day 1 #####

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


##### Day 2 #####

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

##### Day 3 #####

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

##### Day 4 #####

def readInputAndBoards(data):
    boards = []
    results = []
    plays = []
    firstLine = True
    boardLine = 0
    for line in data:
        if (firstLine):
            plays = list([int(play) for play in line.split(",")])
            firstLine = False
        else: 
            if line == '':
                continue
            if boardLine == 0:
                board = [] 
                result = []           

            row = [ int(elem.strip()) for elem in line.split(" ") if elem != '']
            board.append(row)
            result.append([1]*5)
            boardLine += 1

            if boardLine == 5:
                boards.append(board)
                results.append(result)
                boardLine = 0
    return boards, plays, results

def playBoard(play, board, results):
    for i in range(5):
        for j in range(5):
            if (board[i][j] == play):
                results[i][j] = 0
                return results
    return results

def checkBoards(results):
    #print(results)
    for n in range(len(results)):
        result = results[n]
        for i in range(5):
            countCol = [item[i] for item in result].count(0)
            countRow = result[i].count(0)
            if countCol == 5 or countRow == 5:
                return (n, True)
    return (n, False)

def getScore(board, results, play):
    score = 0
    for i in range(5):
        score += sum([x*y for x,y in zip(board[i], results[i])])
    return score * play

# Day 4, part 1: 35711 (0.063 secs)
def day4_1(data):
    #data = read_input(2021, "41") 
    boards, plays, results = readInputAndBoards(data)
    
    for play in plays:       
        for i in range(len(boards)):
            board = boards[i]
            results[i] = playBoard(play, board, results[i])

        n, hasWon = checkBoards(results)
        if (hasWon):
            result = getScore(boards[n], results[n], play)
            #print("winning play:", play, "on board:", n)    
            break  
                 
    AssertExpectedResult(35711, result)

    return result

# 45440 high 
# 13936 high
# Day 4, part 2: 5586 (0.043 secs)
def day4_2(data):
    #data = read_input(2021, "41") 
    boards, plays, results = readInputAndBoards(data)
    winners = []
    
    for play in plays:       

        for i in range(len(boards)):
            board = boards[i]
            results[i] = playBoard(play, board, results[i])

        n, hasWon = checkBoards(results)
        if (hasWon):
            if n not in winners:
                winners.append(n)
            else:
                continue
            result = getScore(boards[n], results[n], play)
            #print("last winning play:", play, "on board:", n)    
  
            results[n] = [ [1] * 5 for i in range(5)]
            
    AssertExpectedResult(5586, result)
    
    return result


##### Day 5 #####

def readLines(data):
    lines = []
    Point = namedtuple('Point', 'x y')
    for line in data:
        inputData = line.split("->")
        start = inputData[0].strip().split(",")
        end = inputData[1].strip().split(",")
        lines.append( (Point(int(start[0]), int(start[1])),  Point(int(end[0]), int(end[1]))) )
    return lines

def fillMap(lines, rows, columns, fillDiagonal = False):

    map = [ [ (0) for i in range(columns) ] for j in range(rows) ]    

    for (p1, p2) in lines:
        if (p1.x <= p2.x):
            startX = p1.x
            endX = p2.x
        else:
            startX = p2.x
            endX = p1.x

        if (p1.y <= p2.y):
            startY = p1.y
            endY = p2.y
        else:
            startY = p2.y
            endY = p1.y

        #print("Fill",p1," ->",p2,"from x:",startX, endX,"to y:",startY, endY)
        if (startX == endX):
            for j in range(startY, endY+1):
                map[j][startX] += 1
        elif (startY == endY):
            for i in range(startX, endX+1):
                map[startY][i] += 1
        elif (fillDiagonal):
            #print("Fill diag",p1," ->",p2,"from x:",startX, endX,"to y:",startY, endY)
            i = p1.x 
            j = p1.y

            control = startX
            while (control <= endX):
                #print("filling",i,j)
                map[j][i] += 1
                control +=1
                if (p1.x <= p2.x):
                    i+=1
                else:
                    i-=1
                if (p1.y <= p2.y):
                    j+=1
                else:
                    j-=1
                          
    return map

def countIntersections(map, rows, columns):
    result = 0
    for i in range(rows):
        for j in range(columns):
            if map[i][j] >= 2:
                result += 1
    return result

# 953911
# Day 5, part 1: 5 (0.047 secs)
def day5_1(data):
    #data = read_input(2021, "51")  
    columns = 1000
    rows = 1000
    lines = readLines(data)
    map = fillMap(lines, rows, columns)
    result = countIntersections(map, rows, columns)                 
    AssertExpectedResult(6225, result)

    return result


# 24778
# Day 5, part 2: 22116 (0.151 secs)
def day5_2(data):
    #data = read_input(2021, "51")  
    columns = 1000
    rows = 1000
    lines = readLines(data)
    map = fillMap(lines, rows, columns, True)
    result = countIntersections(map, rows, columns)                 
    AssertExpectedResult(22116, result)

    return result


if __name__ == "__main__":
    main(sys.argv, globals(), AOC_EDITION_YEAR)

