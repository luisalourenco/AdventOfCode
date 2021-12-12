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
from turtle import Turtle, Screen, heading
from math import pi, remainder, sqrt
from collections import namedtuple
from collections import Counter
from collections import defaultdict

# UPDATE THIS VARIABLE
AOC_EDITION_YEAR = 2021

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
print(FILE_DIR)
sys.path.insert(0, FILE_DIR + "/")
sys.path.insert(0, FILE_DIR + "/../")
sys.path.insert(0, FILE_DIR + "/../../")

from common.utils import read_input, main, clear, AssertExpectedResult, ints, setTimeout  # NOQA: E402
from common.mapUtils import printMap, buildMapGrid, buildGraphFromMap
from common.graphUtils import printGraph, find_all_paths, find_path, find_shortest_path, find_shortest_pathOptimal, bfs, dfs, Graph, hashable_lru
from common.aocVM import HandheldMachine
from lark import Lark, Transformer, v_args, UnexpectedCharacters, UnexpectedToken
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
    map = [ [ 0 for _ in range(columns) ] for _ in range(rows) ]    

    for (p1, p2) in lines:        

        #print("Fill",p1," ->",p2,"from x:",startX, endX,"to y:",startY, endY)
        if (p1.x == p2.x): 
            for j in range(min(p1.y, p2.y), max(p1.y, p2.y)+1):
                map[j][p1.x] += 1
        elif (p1.y == p2.y):
            for i in range(min(p1.x, p2.x), max(p1.x, p2.x)+1):
                map[p1.y][i] += 1
        elif (fillDiagonal):
            #print("Fill diag",p1," ->",p2,"from x:",startX, endX,"to y:",startY, endY)
            i = p1.x 
            j = p1.y

            control = min(p1.x, p2.x)
            while (control <= max(p1.x,p2.x)):
                #print("filling",i,j)
                map[j][i] += 1
                control +=1
                dx = 1 if (p1.x <= p2.x) else -1
                dy = 1 if (p1.y <= p2.y) else -1
                i += dx 
                j += dy
                
                          
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

##### Day 6 #####


def simulateLanterfish(lanternfishs, daysSimulation, printData = False):
    newFishTimerValue = 8
    resetFishTimerValue = 6  
    prev = len(lanternfishs)
    sum = 0

    for day in range(daysSimulation):
        
        if (printData):   
            #print("Day",day,":", len(lanternfishs),"diff:", len(lanternfishs)-prev)
            print("Day",day,":", (lanternfishs))
            if (day%7 == 0):                         
                print("Day",day,":", len(lanternfishs),"diff:", len(lanternfishs)-prev)
                prev = len(lanternfishs)
            else:
                sum += len(lanternfishs)
            
            #print("Day",day,":", lanternfishs)

        newFishes = [ 1 if fish == 0 else 0 for fish in lanternfishs].count(1) 
        lanternfishs = [ resetFishTimerValue if fish == 0 else fish -1 for fish in lanternfishs]       
        for _ in range(newFishes):
            lanternfishs.append(newFishTimerValue)

    return lanternfishs

#Day 6, part 1: 387413 (0.922 secs)
def day6_1(data):
    #data = read_input(2021, "61")
    daysSimulation = 80    
    lanternfishs = [int(fish) for fish in data.pop().split(",")]
    lanternfishs = simulateLanterfish(lanternfishs, daysSimulation)
    
    result = len(lanternfishs)               
    AssertExpectedResult(387413, result)

    return result

def simulateLanterfishOptimized(lanternfishs, daysSimulation, printData = False):
    newFishTimerValue = 8
    resetFishTimerValue = 6      
    timers = Counter(lanternfishs)
    
    for _ in range(daysSimulation):       
        newTimers = defaultdict(int)

        for (timer, fishes) in sorted(timers.items()):
            if timer == 0:
                newTimers[newFishTimerValue] = fishes
                newTimers[resetFishTimerValue] = fishes
            elif timer == resetFishTimerValue + 1:
                newTimers[resetFishTimerValue] += fishes
            else:
                newTimers[timer - 1] = fishes
        timers = newTimers
        
        '''
        print("Day",day+1,":", sum([f for f in timers.values()]))
        for (n,f) in sorted(timers.items()):
            print(n,":",f)
        '''
    
    return sum(timers.values())

# Day 6, part 2: 1738377086345 (0.003 secs)       
def day6_2(data):
    #data = read_input(2021, "61")
    daysSimulation = 256    
    lanternfishs = [int(fish) for fish in data.pop().split(",")]
    result = simulateLanterfishOptimized(lanternfishs, daysSimulation)
    
    AssertExpectedResult(1738377086345, result)

    return result


##### Day 7 #####

def computeLeastFuel(horizontalPositions):
    minFuel = sys.maxsize
    maxPos = max(horizontalPositions)

    alignedPosition = maxPos
    while alignedPosition >= 0:
        fuel = 0
        for position in horizontalPositions:
            fuel += abs(alignedPosition - position)
        
        if fuel < minFuel:
            minFuel = fuel
        
        alignedPosition -= 1
    
    return minFuel

# Day 7, part 1: 344605 (0.220 secs)
def day7_1(data):
    #data = read_input(2021, "71")

    horizontalPositions = [int(position) for position in data.pop().split(",")]
    result = computeLeastFuel(horizontalPositions)  
    AssertExpectedResult(344605, result)

    return result

def computeLeastFuelV2(horizontalPositions, fuelCosts):
    minFuel = sys.maxsize
    maxPos = max(horizontalPositions)
    alignedPosition = maxPos
    while alignedPosition >= 0:
        fuel = 0
        for position in horizontalPositions:
            val = abs(alignedPosition - position)
            if val != 0:
                fuel += fuelCosts[val]
            else:
                fuel += 0
        
        if fuel < minFuel:
            minFuel = fuel
        
        alignedPosition -= 1
    
    return minFuel

def computeFuelCosts(maxPos):
    fuelCosts = defaultdict()  
    increment = 1
    fuel = 0
    for j in range(1, maxPos+1):       
        fuel+= increment
        increment += 1
        fuelCosts[j] = fuel
    
    return fuelCosts

# Day 7, part 2: 93699985 (80.608 secs)
# Day 7, part 2: 93699985 (0.267 secs)
def day7_2(data):
    #data = read_input(2021, "71")

    horizontalPositions = [int(position) for position in data.pop().split(",")]
    maxPos = max(horizontalPositions)
    fuelCosts = computeFuelCosts(maxPos)
    result = computeLeastFuelV2(horizontalPositions, fuelCosts)    
    AssertExpectedResult(93699985, result)

    return result


##### Day 8 #####

def parseEntries(data):
    Entry = namedtuple('Entry', 'input output')
    entries = []
    for line in data:
        signals = line.split(" | ")
        input = signals[0].split(" ")
        output = signals[1].split(" ")
        entries.append(Entry(input, output))
    
    return entries

def interpretEntries(entries):
    segments = defaultdict(int)
    segments[7] = 3
    segments[4] = 4
    segments[1] = 2
    segments[8] = 7
    unique = 0
   
    for entry in entries:
        ones = sum([1 for i in entry.output if len(i) == segments[1]])
        fours = sum([1 for i in entry.output if len(i) == segments[4]])
        sevens = sum([1 for i in entry.output if len(i) == segments[7]])
        eights = sum([1 for i in entry.output if len(i) == segments[8]])
        
        unique += ones + fours + sevens + eights        
    
    return unique

# Day 8, part 1: 375 (0.055 secs)
def day8_1(data):
    #data = read_input(2021, "81")

    entries = parseEntries(data)
    result = interpretEntries(entries)

    AssertExpectedResult(375, result)

    return result


def cleanupResult(l):
    if len(l) == 0:
        return ''
    else:
        return l.pop()


def sortString(str):
    return ''.join(sorted(str))


# based on 1s, find 3s since 2 and 5s will not contain 1s
def findThrees(ones, sizeFive, numbers):   
    for three in sizeFive:
        if len(set(three).intersection(ones)) == 2:
            numbers[three] = 3    
            numbers[3] = three


# at this point we know 3s, lenght 6 is either a 0, 6 or 9 but only 9 has 3s in it
def findNines(sizeSixes, numbers):
    three = numbers[3]
    for nine in sizeSixes:
        if len(set(nine).intersection(three)) == 5:
            numbers[sortString(nine)] = 9    
            numbers[9] = sortString(nine)

# at this point we know 9s, lenght 5 is either a 2, 5 or 3 but 9s contains 5
def findFives(sizeFives, numbers):
    nine = numbers[9]
    for five in sizeFives:
        if len(set(nine).intersection(five)) == 5:
            numbers[sortString(five)] = 5    
            numbers[5] = sortString(five)

# we can decide 6s from 5s at this point
def findSixes(sizeSixes, numbers):
    five = numbers[5]
    for six in sizeSixes:
        if len(set(six).intersection(five)) == 5:
            numbers[sortString(six)] = 6    
            numbers[6] = sortString(six)

def computeValue(numbers, entry):
    val = ''.join([str(numbers[sortString(number)]) for number in entry])
    return int(val)   


def findDigitsAndGetValue(entries):
    
    sum = 0
    for entry in entries:
        numbers = dict()
        onesOut = cleanupResult([sortString(i) for i in entry.output if len(i) == 2])
        onesIn = cleanupResult([sortString(i) for i in entry.input if len(i) == 2])
        ones = set(onesIn).union(set(onesOut))
        sortedOne = sortString(''.join(ones))
        numbers[sortedOne] = 1
        numbers[1] = sortedOne
        
        foursIn = cleanupResult([sortString(i) for i in entry.input if len(i) == 4])
        foursOut = cleanupResult([sortString(i) for i in entry.output if len(i) == 4])
        fours = set(foursIn).union(set(foursOut))
        sortedFour = sortString(''.join(fours))
        numbers[sortedFour] = 4
        numbers[4] = sortedFour

        sevensIn = cleanupResult([sortString(i) for i in entry.input if len(i) == 3])
        sevensOut = cleanupResult([sortString(i) for i in entry.output if len(i) == 3])
        sevens = set(sevensIn).union(set(sevensOut))
        sortedSeven = sortString(''.join(sevens))
        numbers[sortedSeven] = 7
        numbers[7] = sortedSeven

        eightsIn = cleanupResult([sortString(i) for i in entry.input if len(i) == 7])       
        eightsOut = cleanupResult([sortString(i) for i in entry.output if len(i) == 7]) 
        eights = set(eightsIn).union(set(eightsOut))
        sortedEights = sortString(''.join(eights))
        numbers[sortedEights] = 8      
        numbers[8] = sortedEights

        sizeFiveIn = [sortString(i) for i in entry.input if len(i) == 5]     
        sizeFiveOut = [sortString(i) for i in entry.output if len(i) == 5]    
        sizeFives = set(sizeFiveIn).union(set(sizeFiveOut))

        sizeSixIn = [sortString(i) for i in entry.input if len(i) == 6]     
        sizeSixOut = [sortString(i) for i in entry.output if len(i) == 6]    
        sizeSixes = set(sizeSixIn).union(set(sizeSixOut))
       
        # based on 1s, find 3s since 2 and 5s will not contain 1s
        findThrees(ones, sizeFives, numbers)
        sizeFives.remove(numbers[3])
        # based on 3s, find 9s since 6s and 0s will not contain 3s in it
        findNines(sizeSixes, numbers)
        sizeSixes.remove(numbers[9])
        # based on 9s, find 5s since 2s or 3s will not contains 5s in it
        findFives(sizeFives, numbers)
        sizeFives.remove(numbers[5])        
        # we can infer 6s from 5s at this point
        findSixes(sizeSixes, numbers)
        sizeSixes.remove(numbers[6])

        # only zero is left on list of lenght 6
        zero = sizeSixes.pop()
        numbers[0] = zero
        numbers[zero] = 0
       
        # only two is left on list of lenght 5
        two = sizeFives.pop()
        numbers[2] = two
        numbers[two] = 2

        sum += computeValue(numbers, entry.output)
    return sum


# Day 8, part 2: 1019355 (0.013 secs)
def day8_2(data):
    #data = read_input(2021, "81")
    entries = parseEntries(data)
    result = interpretEntries(entries)
    result = findDigitsAndGetValue(entries)

    AssertExpectedResult(1019355, result)

    return result


##### Day 9 #####

def findLowPoints(heightMap):
    Position = namedtuple('Position', 'height x y')
    lowPoints = []    
    columns = len(heightMap[0])
    rows = len(heightMap)
    newHeightMap =  [[9 for x in range(columns)] for y in range(rows)] 

    for j in range(columns):
        for i in range(rows):
            adjacent = []
            height = heightMap[i][j]
            newHeightMap[i][j] = Position(height, i, j)
           
            if (i != 0):
               adjacent.append(heightMap[i-1][j])
            if (i != rows-1):
                adjacent.append(heightMap[i+1][j])
            if (j != 0):
                adjacent.append(heightMap[i][j-1])
            if (j != columns - 1):
                adjacent.append(heightMap[i][j+1])
            
            if height < min(adjacent):
                lowPoints.append(Position(height, i, j))

    return lowPoints, newHeightMap


# Day 9, part 1: 516 (0.066 secs)
def day9_1(data):
    #data = read_input(2021, "91")
    heightMap = []    
    for line in data:
        heightMap.append([int(position) for position in list(line)])
    
    lowPoints, _ = findLowPoints(heightMap)
    result = sum([risk.height + 1 for risk in lowPoints])
  
    AssertExpectedResult(516, result)

    return result

def findBasins(heightMap, lowPoints, newHeightMap):
    basins = []
    
    for lowPoint in lowPoints:
        #print("**** lowpoint ****", lowPoint)        
        basins.append(findBasinRec(lowPoint, newHeightMap, [], []))       

    return basins  
        

def findBasinRec(lowPoint, map, basin, visited):
    columns = len(map[0])
    rows = len(map)
    x = lowPoint.x
    y = lowPoint.y
    height = lowPoint.height   

    #print("rec with:", lowPoint)    
    #print("visited?", lowPoint in visited)
    #print("visited:",visited)
    if lowPoint in visited:
        #print("basin:", basin,len(visited) )
        return []
    else:
        visited.append(lowPoint)    
    
    if (height == 9):
        return []
    else:          
        basin.append(height)
        findBasinRec(map[x-1][y], map, basin, visited) if x-1 >= 0 else [] 
        findBasinRec(map[x+1][y], map, basin, visited) if x+1 <= rows-1 else []
        findBasinRec(map[x][y-1], map, basin, visited) if y-1 >= 0 else [] 
        findBasinRec(map[x][y+1], map, basin, visited) if y+1 <= columns - 1 else []
        return basin

# Day 9, part 2: 1023660 (0.060 secs)
def day9_2(data):
    #data = read_input(2021, "91")
    heightMap = []    
    for line in data:
        heightMap.append([int(position) for position in list(line)])
    
    lowPoints, newHeightMap = findLowPoints(heightMap)
    basins = findBasins(heightMap, lowPoints, newHeightMap)

    #print("basins found:",basins)    

    zipped = zip(map(len, basins), basins)
    res = sorted(list(zipped), key = lambda x: x[0])
    res.reverse()
    result = functools.reduce(operator.mul, [x for x,_ in res][:3] , 1)

    AssertExpectedResult(1023660, result)

    return result


##### Day 10 #####

grammar = """
start: leftside+
leftside: lparan+ | lcparan+ | langleparan+ | lregparan+

lparan: "[" leftside* rparan 
lcparan:  "{" leftside* rcparan
langleparan:  "<" leftside* rangleparan
lregparan:  "(" leftside* rregparan

rparan:"]" 
rcparan: "}" 
rangleparan:">" 
rregparan:  ")"

%import common.WS_INLINE
%ignore WS_INLINE
%ignore " "     
"""

def computeSyntaxErrorScore(e):
    excp = str(e)
    st = excp.split("Token(")[1]
    st = st.split(", ")[1]    
    s = st.split("'")[1].strip()
    
    #print("token:",s)
    if s == ')':
        val = 3
    elif s == ']':
        val = 57
    elif s == '}':
        val = 1197
    elif s ==  '>':
        val = 25137
    else:
        val = 0
    return val

# Day 10, part 1: 323613 (0.183 secs)
def day10_1(data):
    #data = read_input(2021, "101")
    
    parser = Lark(grammar, parser='lalr')
    calc = parser.parse

    syntaxErrorScore = 0
    for line in data:
        try:
            calc(line)
        except Exception as e:
            score = computeSyntaxErrorScore(e)
            #print("score:",score)
            syntaxErrorScore += score  

    result = syntaxErrorScore
  
    AssertExpectedResult(323613, result)

    return result

def getTokenValue(token):
    if token == ')':
        return 1
    elif token == ']':
        return  2
    elif token == '}':
        return  3
    elif token ==  '>':
        return  4
    else:
        return 0

def computeAutoCompleteScore(e, token):
    excp = str(e)
    st = excp.split("Token(")[1]
    st = st.split(", ")[1]    
    s = st.split("'")[1].strip()
    
    #print("token:",s)
    if len(s) == 0:        
        return getTokenValue(token)
    else:
        return 0

def getIncompleteLines(data, calc):
    incompleteLines = []
    for line in data:
        try:
            calc(line)
        except Exception as e:
            score = computeSyntaxErrorScore(e)
            if score == 0:
                incompleteLines.append(line)
    return incompleteLines

def fixLine(line, calc):
    candidateTokens = [')','}','>',']']
   
    for token in candidateTokens:
        try:
            newLine = line + token
            #print("line:",line,"with token:", token)
            calc(newLine)  
            score = getTokenValue(token)
            break
        except Exception as e:   
            score = computeAutoCompleteScore(e, token)            
            if score != 0: # means we found a match    
                #print("added token",token)
                break
    return newLine, score

# Day 10, part 2: 3103006161 (2.543 secs)
def day10_2(data):
    #data = read_input(2021, "101")
    
    parser = Lark(grammar, parser='lalr')
    calc = parser.parse

    incompleteLines = getIncompleteLines(data, calc)         
    fixedLine = False
    scores = []
    
    for line in incompleteLines:
        autoCompleteScore = 0
        fixedLine = False
        while not fixedLine:           
            try:
                calc(line)
                fixedLine = True                
            except Exception:
                line, score = fixLine(line, calc)
                autoCompleteScore = (5 * autoCompleteScore) + score
        scores.append(autoCompleteScore)

    scores.sort()
    #print(scores)
    #print(scores[math.floor(len(scores)/2)])
    result = scores[math.floor(len(scores)/2)]      
  
    AssertExpectedResult(323613, result)

    return result


##### Day 11 #####

def printGrid(grid):
    for row in grid:
        print(row)


def takeStepAt(oldGrid, grid, row, column, flashes):
    if row < 0 or row > 9 or column < 0 or column > 9:
        return grid, flashes
    
    energy = oldGrid[row][column]
            
    if energy + 1 > 9: # FLASH!

        if (row,column) not in flashes:
            #print("FLASH!", (row, column))
            flashes.append((row,column))

            grid, flashes = takeStepAt(grid, copy.deepcopy(grid), row + 1, column, flashes)
            grid, flashes = takeStepAt(grid, copy.deepcopy(grid), row + 1, column + 1, flashes)
            grid, flashes = takeStepAt(grid, copy.deepcopy(grid), row + 1, column - 1, flashes)            
            grid, flashes = takeStepAt(grid, copy.deepcopy(grid), row - 1, column, flashes)            
            grid, flashes = takeStepAt(grid, copy.deepcopy(grid), row - 1, column + 1, flashes)
            grid, flashes = takeStepAt(grid, copy.deepcopy(grid), row - 1, column - 1, flashes)            
            grid, flashes = takeStepAt(grid, copy.deepcopy(grid), row, column + 1, flashes)
            grid, flashes = takeStepAt(grid, copy.deepcopy(grid), row, column - 1, flashes)
        
        energy = 0       

    else:
        energy += 1

    grid[row][column] = energy

    return grid, flashes

def takeStep(grid):
    flashes = []
    for row in range(10):
        for column in range(10):
            grid, flashes = takeStepAt(copy.deepcopy(grid), grid, row, column, flashes)
            
    for (row, column) in flashes:
        grid[row][column] = 0

    return grid, flashes

# Day 11, part 1: 1735 (1.769 secs)
def day11_1(data):
    #data = read_input(2021, "111")
    steps = 100
    
    grid = [list(map(int, i)) for i in data]
    #printGrid(grid)

    totalFlashes = 0
    for step in range(1, steps+1):
        #print("step",step)
        grid, flashes = takeStep(grid)
        totalFlashes += len(flashes)        
        #printGrid(grid)

    result = totalFlashes  
    AssertExpectedResult(1735, result)

    return result

# Day 11, part 2: 400 (6.123 secs)
def day11_2(data):
    #data = read_input(2021, "111")
    steps = 1000    
    grid = [list(map(int, i)) for i in data]

    syncedFlashesStep = 0
    for step in range(1, steps+1):
        grid, flashes = takeStep(grid)
        if len(flashes) == 100:
            syncedFlashesStep = step 
            break        

    result = syncedFlashesStep  
    AssertExpectedResult(400, result)

    return result



##### Day 12 #####

def createGraphForCaveSystem(data):
    graph = dict()
    for line in data:
        nodes = line.split("-")
        fromNode = nodes[0]
        toNode = nodes[1]
        if fromNode not in graph:
            graph[fromNode] = []
        if toNode not in graph:
            graph[toNode] = []
        graph[fromNode].append(toNode)
        graph[toNode].append(fromNode)
    return graph


@hashable_lru
def findAllPathsInCaveSysem(graph, start, end, path=[]):
        path = path + [start]
        
        if start == end:
            return [path]
        if graph[start] ==  None:
            return []
        paths = []

        for node in graph[start]:
            if node.islower() and node in path:
                continue
            newpaths = findAllPathsInCaveSysem(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
        return paths

@hashable_lru
def findAllPathsInCaveSysemV2(graph, lowers, start, end, path=[]):
        path = path + [start]
        
        if start == end:
            return [path]
        if graph[start] ==  None:
            return []
        paths = []        
        
        for node in graph[start]:
            
            counter = Counter(node for node in path if node.islower())
            if node == 'start' and node in path:
                continue
            if node == 'end' and node in path:
                continue

            if node.islower()and counter[node] > 1:
                continue   

            if sum(counter.values()) > len(counter)+1:
                continue               

            newpaths = findAllPathsInCaveSysemV2(graph, lowers,node, end, path)
            for newpath in newpaths:
                paths.append(newpath)

        return paths

def day12_1(data):
    #data = read_input(2021, "121")   

    graph = createGraphForCaveSystem(data)   
    paths = findAllPathsInCaveSysem(graph, 'start', 'end')
    result = len(paths)
    print(result)
    AssertExpectedResult(5756, result)

def day12_2(data):
    #data = read_input(2021, "121")   
    graph = createGraphForCaveSystem(data)   
    lowers = []
    for node in graph:
        if node.islower():
            lowers.append(node)

    paths = findAllPathsInCaveSysemV2(graph, lowers,'start', 'end')
   
    result = len(paths)
    print(result)              

    AssertExpectedResult(144603, result)


##### Day 13 #####

def day13_1(data):
    data = read_input(2021, "131")   

    for line in data:
        inputData = line.split(" ")
    
    result = 0
    AssertExpectedResult(5756, result)
    #data = read_input(2021, "121")   


    

if __name__ == "__main__":
    # override timeout
    main(sys.argv, globals(), AOC_EDITION_YEAR, 900)

