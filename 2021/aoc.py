# Based on template from https://github.com/scout719/adventOfCode/
# -*- coding: utf-8 -*
# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=wildcard-import
# pylint: disable=wrong-import-position
# pylint: disable=consider-using-enumerate-

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
#from typing_extensions import Literal 
import numpy as np
from functools import lru_cache
import operator
from itertools import takewhile
import itertools, collections
from turtle import Turtle, Screen, heading
from math import exp, pi, remainder, sqrt
from collections import namedtuple
from collections import Counter
from collections import defaultdict
from termcolor import colored
import termcolor

# UPDATE THIS VARIABLE
AOC_EDITION_YEAR = 2021

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
print(FILE_DIR)
sys.path.insert(0, FILE_DIR + "/")
sys.path.insert(0, FILE_DIR + "/../")
sys.path.insert(0, FILE_DIR + "/../../")

from common.utils import read_input, main, clear, AssertExpectedResult, ints, printGridsASCII  # NOQA: E402
from common.mapUtils import printMap, buildMapGrid, buildGraphFromMap
from common.graphUtils import dijsktra, printGraph, find_all_paths, find_path, find_shortest_path, find_shortest_pathOptimal, bfs, dfs, Graph, hashable_lru
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

def parseDots(data):
    Dot = namedtuple('Dot', 'x y')
    Fold = namedtuple('Fold', 'axis value')
    foldInstructions = []
    dots = []
    i = 0
    rows = 0
    columns = 0

    for line in data:
        i +=1
        dotPosition = line.split(",")
        if len(dotPosition) != 2:
            break
        x = int(dotPosition[0])
        y = int(dotPosition[1])
        dots.append(Dot(x, y))

        if x > columns:
            columns = x
        if y > rows:
            rows = y
        #print(i)

    
    for i in range(i, len(data)):
        line = data[i]
        input = line.split("fold along ")[1]
        instruction = input.split("=")
        foldInstructions.append(Fold(instruction[0], int(instruction[1])))
    
    return dots, foldInstructions, rows, columns
    #print(dots)
    #print(foldInstructions)

def printPapper(papper):
    for row in papper:
        print(row)

def foldUp(papper, value):
    columns = len(papper[value])
    i = value - 1
    foldedPapper = copy.deepcopy(papper)
    for row in range(value+1, len(papper)):
        for column in range(columns):
            if (papper[row][column] == '#'):
                foldedPapper[i][column] = papper[row][column]
        i -=1

    return foldedPapper[:value][:columns]

def foldLeft(papper, value):
    rows = len(papper)    
    i = value - 1
    foldedPapper = copy.deepcopy(papper)

    for column in range(value+1, len(papper[0])):
        for row in range(rows):
            if (papper[row][column] == '#'):
                foldedPapper[row][i] = papper[row][column]
        i -=1
    
    return [foldedPapper[i][:value] for i in range(0,rows)]

def followFoldInstructions(foldInstructions, papper, totalFolds):    
    for fold in foldInstructions:
        if totalFolds == 0:
            break
        if fold.axis == 'x':
            papper = foldLeft(papper, fold.value)
        elif fold.axis == 'y':
            papper = foldUp(papper, fold.value)
        totalFolds -=1
    return papper

def fillPapperAndFold(data, foldAll = False):
    dots, foldInstructions, rows, columns = parseDots(data)
    papper = [['.' for x in range(columns+1)] for y in range(rows+1)] 
    
    for dot in dots:
        papper[dot.y][dot.x] = '#'    
    
    if foldAll:
        return followFoldInstructions(foldInstructions, papper, 1000)
    else:
        return followFoldInstructions(foldInstructions, papper, 1)

def day13_1(data):
    #data = read_input(2021, "131")   
    foldedPapper = fillPapperAndFold(data)
    result = 0
    for column in range(len(foldedPapper)):
        for row in range(len(foldedPapper[0])):
            if foldedPapper[column][row] == '#':
                result += 1
    
    print(result)
    AssertExpectedResult(693, result)


def day13_2(data):
    #data = read_input(2021, "131")   

    foldedPapper = fillPapperAndFold(data, True)
    printGridsASCII(foldedPapper, '#')

    result = "UCLZRAZU"
    AssertExpectedResult("UCLZRAZU", result)


##### Day 14 #####

def parseRule(line):
    rule = line.split(" -> ")
    triggers = rule[0]
    result = rule[1]
    return (triggers[0], triggers[1], result)

def parseAllRules(data):
    rules = defaultdict()
    counter = defaultdict()
    for line in data:
        (first, second, res) = parseRule(line)
        if first not in rules:
            secondDict = defaultdict()
        else:
            secondDict = rules[first]
        secondDict[second]  = res
        rules[first] = secondDict
        counter[first+second] = 0
                
    return (rules, counter)


def printTimeTaken(start, end, msg = ''):
    print(msg,"Took ({0:.3f} secs)".format(end - start))

def applyAllRules(pair, rules, old, counter, cannotDelete):
    first = pair[0]
    second = pair[1]    
    val = rules[first][second]

    if val is not None: # found matching rule AB -> C
        
        times = old[pair]
        #print("Triggered rule",pair,"->",val, times,"times")

        # increments C occurences in polymer + times rule AB triggered
        counter[val] += times

        if pair not in cannotDelete:
            # delete AB for next iteration
            #print("Removing pair",pair)
            del counter[pair]
        else:
            counter[pair] -= times      

        # adds new pair AC for next iteration
        #print("Adding new pair",first+val)
        cannotDelete.append(first+val)
        counter[first+val] += times

        # adds new pair CB for next iteration
        #print("Adding new pair",val+second)
        cannotDelete.append(val+second)
        counter[val+second] += times

    return (counter, cannotDelete)


def applyRulesForEachPair(rules, counter):    
    c = copy.deepcopy(counter)
    #print("Initial state:", c)
    cannotDelete = []
    for (pair, _) in c.items():
        if len(pair) == 2:    
            (counter, cannotDelete) = applyAllRules(pair, rules, c, counter, cannotDelete)

    return counter


def getPolymerTemplateResult(data, steps):
    polymer_template = data[0]
    (rules, counter) = parseAllRules(data[2:])

    pairs = [polymer_template[i:i+2] for i in range(len(polymer_template)-1)]     
    counter = Counter(polymer_template)
    for pair in pairs:
        counter[pair] = 1    

    for step in range(1, steps+1):
        #print("After step",step,":")
        counter = applyRulesForEachPair(rules, counter)
        #print(counter)
        #print("-----")
        #print()
        
    minVal = sys.maxsize
    maxVal = 0
    for (k,v) in counter.items():
        if len(k) == 1:
            minVal = minVal if minVal < v else v
            maxVal = maxVal if maxVal > v else v
    return maxVal - minVal

# Day 14, part 1: 5656 (0.067 secs)
def day14_1(data):
    #data = read_input(2021, "141")   
    result = getPolymerTemplateResult(data, 10)
    print(result)
    AssertExpectedResult(5656, result)
    
def sort_dict(dict):
    return {k: v for k, v in sorted(dict.items(), key=lambda item: item[1])}

# Day 14, part 2: 12271437788530 (0.015 secs)
def day14_2(data):
    #data = read_input(2021, "141")   
    result = getPolymerTemplateResult(data, 40)
    print(result)
    AssertExpectedResult(12271437788530, result)


##### Day 15 #####


def createLink(map, fromNode, i, j, graph, rows, columns):
    (x,y,v) = fromNode
    
    if i >= rows or j >= columns or j < 0 or i < 0:
        return graph

    toNode = (i, j, map[i][j])

    graph.add_edge((x,y), (i,j), map[i][j])

    #graph[toNode].append(fromNode)
    return graph


def expandMap(map, rows, columns, boundX, boundY):
    fullMap = [ [ 0 for _ in range(columns) ] for _ in range(rows) ]   

    for i in range(rows):
        addRow = math.floor(i / boundX)
        for j in range(columns):            
            addColumn = math.floor(j / boundY)
            #print("val:",map[i%boundX][j%boundY], addRow, addColumn)
            risk = (map[i%boundX][j%boundY] + addColumn + addRow) #% 10
            #print("risk:",(map[i%boundX][j%boundY] + addColumn + addRow), risk)
            
            if risk >= 10:
                risk = (risk % 10) +1
            fullMap[i][j] = risk

    '''
    for l in fullMap:
        string_ints = [str(int) for int in l]
        str_of_ints = "".join(string_ints)
        print(str_of_ints)
    '''

    return fullMap

def getMap(data, expand = False):
    map = [ [ 0 for _ in range(len(data[0])) ] for _ in range(len(data)) ] 
    i = 0
    for line in data:
        positions = [int(n) for n in line]
        map[i] = positions
        i+=1
    
    if expand:
        return expandMap(map, len(data)*5, len(data[0])*5, len(data), len(data[0]))

    return map

def createGraphForCave(data, expand = False):
    #graph = dict()
    graph= Graph()     
    map = getMap(data, expand)   
    rows = len(map)
    columns = len(map[0])   
    #print(len(map))

    start = (0, 0) #, map[0][0])
    end = (rows-1, columns-1) #, map[rows-1][columns-1])
    for i in range(rows):
        for j in range(columns):
            fromNode = (i, j, map[i][j])
            graph = createLink(map, fromNode, i+1, j, graph, rows, columns)
            graph = createLink(map, fromNode, i, j+1, graph, rows, columns)
            graph = createLink(map, fromNode, i-1, j, graph, rows, columns)
            graph = createLink(map, fromNode, i, j-1, graph, rows, columns)    

    return (graph, start, end, map)


# Day 15, part 1: None (3.092 secs)
def day15_1(data):
    #data = read_input(2021, "151")   
    path = []
    (graph, start, end, map) = createGraphForCave(data)

    path = dijsktra(graph, start, end)
    #print(path)

    result = 0
    for (i,j) in path:
        #print("(",i,",",j,")",map[i][j])
        result += map[i][j]
    result-= map[0][0]    
    
    print(result)
    AssertExpectedResult(508, result)

# Day 15, part 2: None (3.942 secs)
def day15_2(data):
    #data = read_input(2021, "151")   
    (graph, start, end, _) = createGraphForCave(data, True)
    (_, cost) = graph.a_star_search(start, end)
    #print(path)    
    result = cost[end]    
    print(result)
    AssertExpectedResult(2872, result)


##### Day 16 #####


def HexToBin(data):
    bin = ''
    #print("string:",data)
    print("input read:",list(data))
    packetVersion = ''

    for c in list(data):
        if c == '0':
            bin += '0000'
        elif c == '1':
            bin += '0001'
        elif c == '2':
            bin += '0010'
        elif c == '3':
            bin += '0011'
        elif c == '4':
            bin += '0100'
        elif c == '5':
            bin += '0101'
        elif c == '6':
            bin += '0110'
        elif c == '7':
            bin += '0111'
        elif c == '8':
            bin += '1000'
        elif c == '9':
            bin += '1001'
        elif c == 'A':
            bin += '1010'
        elif c == 'B':
            bin += '1011'
        elif c == 'C':
            bin += '1100'
        elif c == 'D':
            bin += '1101'
        elif c == 'E':
            bin += '1110'
        elif c == 'F':
            bin += '1111'
    
    return bin

def BinToDec(bin):
    return int(bin, 2)

# Literal Value packet
def processLiteralValuePacket(packet, ops, depth):
    Instruction = namedtuple('Instruction', 'type val')
    print("***Literal value packet:",packet,"***")
    safeGuard = 100
    literal = ''
    while safeGuard > 0:
        safeGuard -= 1
        num = packet[:5]
        lastGroup = True if num[0] == '0' else False
        packet = packet[5:]
        literal += num[1:]

        print("Is last group ?",lastGroup )
        print("Literal bin", literal)

        if lastGroup:
            if packet.count('0') == len(packet):
                packet = ''
            print("Remaining transmission:", packet)
            print("Literal:", BinToDec(literal))
            #ops.append(Instruction('lit', BinToDec(literal)))
            return packet, BinToDec(literal), ops

        print("Next literal packet:", packet)

    return packet, literal, ops

# Lenght ID = 0
def processOperatorPacketByFixedLength(packet, ops, type, depth):
    print("Processing ID 0 (by total lenght)")
    lenght = BinToDec(packet[:15])
    packet = packet[15:]
    print("Length:", lenght)

    packetsLeft = packet#[:lenght]
    packet = packetsLeft #'' #packet[lenght:]
    print("Packets to process:", packetsLeft)
    t, tt, o = processTransmission(packetsLeft, ops, depth)

    Instruction = namedtuple('Instruction', 'type val')    
    #o.append( Instruction('op', type) )
    #ops.insert(1,o)
    
    return t,tt,o

# Length ID = 1
def processOperatorPacketByNumberPackets(packet, ops, type, depth):
    print("Processing ID 1 (fixed number of packets)")
    packetsLeft = BinToDec(packet[:11])
    print("Packets to process", packet[:11])
    packet = packet[11:]

    print("Number of packets:", packetsLeft)

    total = 0
    subops = []
    #ddepth = depth+1
    for i in range(packetsLeft):
        print("Processing subpacket", i+1,":", packet)
        packet, t, ops = processTransmission(packet, ops, depth)
        #subops.append(ops)
        total += t
    
    Instruction = namedtuple('Instruction', 'type val')
    #subops.append( Instruction('op', type) )
    #ops.insert(1,subops)
    
    return packet, total, ops

# Operator packet
def processOperatorPacket(packet, ops, type, depth):
    total = 0
    print("***Operator packet***")
    packetLengthId = packet[:1]
    packet = packet[1:]

    print("Packet Length ID:",packetLengthId)
    if BinToDec(packetLengthId) == 0:
        packet, t, ops = processOperatorPacketByFixedLength(packet, ops, type, depth+1)
    elif BinToDec(packetLengthId) == 1:
        packet, t, ops = processOperatorPacketByNumberPackets(packet, ops, type, depth+1)
    total += t 

    return packet, total, ops


# Process operations
def processOperation(acc, args, packetType, result, ops):
    Instruction = namedtuple('Instruction', 'type val')

    print("Process operation type", packetType,"with operands",args)
    if len(args) == 0:
        print("empty args, using acc instead:", acc)
        args = acc 

    if packetType == 0: # sum
        result = sum(args)
        operation = 'sum (+)'
    elif packetType == 1: # product
        result = 1#functools.reduce(operator.mul, [arg for arg in args], 1)
        for arg in args:
            result *= arg
        operation = 'product (*)'
    elif packetType == 2: # min
        result = min(args)
        operation = 'min'
        acc = []
    elif packetType == 3: # max
        result = max(args)
        operation = 'max'
        acc = []
    elif packetType == 5: # gt
        result = 1 if args[0] < args[1] else 0
        operation = 'gt (>)'
    elif packetType == 6: # lt
        result = 1 if args[0] > args[1] else 0
        operation = 'lt (<)'
    elif packetType == 7: # gt
        result = 1 if args[0] == args[1] else 0
        operation = 'equal (=)'

    acc.insert(0,result)   

    print("Result is", result,"for operation", operation,"with operands",args,"leftover ops", ops)
    return result, acc


# Process operations
def processAllOperations(ops, result):
    #ops.reverse()
    print("Process all",len(ops),"operations",ops)

    result = 0
    args = []
    acc = []
    opDepth = 1
    while len(ops) > 0:
        
        packetType = None
        instruction, depth = ops.pop()
        print("Instruction", instruction,"depth:", depth)        

        if instruction.type == 'lit':
            print("depth:",opDepth, depth)
            if opDepth == depth or opDepth +1 == depth:
                args.append(instruction.val)  

        if instruction.type == 'op':
            opDepth+=1
            packetType = instruction.val
            res, acc = processOperation(acc, args, packetType, result, ops)
            args = []
            result = res


    print("Result is", result)
    return result, ops

# Packet processing main procedure
def processTransmission(transmission, ops = [], depth = 1):   
    if len(transmission) == 0:
        return '', 0, ops
    
    total = 0
    print()
    print("Processing packet ", transmission)
    packetVersion = BinToDec(transmission[:3]) # take first 3 bits
    packeType = BinToDec(transmission[3:6]) # take first 3 bits
    transmission = transmission[6:]

    print("Packet Version", packetVersion)
    print("Packet Type", packeType)

    total += packetVersion
    result = 0
    Instruction = namedtuple('Instruction', 'type val')

    if packeType == 4:
        transmission, literal, ops = processLiteralValuePacket(transmission, ops, depth)    
        ops.append( (Instruction('lit', literal), depth) )
        
        print("end with", transmission)
    else:        
        ops.append( (Instruction('op', packeType), depth) )
        transmission, t, ops = processOperatorPacket(transmission, ops, packeType, depth)

    transmission, t, ops = processTransmission(transmission, ops)
    
    total += t

    return transmission, total, ops

def runTestsDay16():
    data = read_input(2021, "16Tests")  
    expected = [3,54,7,9,1,0,0,1,10,5000000000,1495959086337]

    i = 0
    for line in data:
        t = line
        converted = HexToBin(t)    
        result = 0
        _, result, ops = processTransmission(converted, [])
        result, _ = processAllOperations(ops, result)
        print("Operation result is:",result)
        if expected[i] == result:
            print("Example", t,"passed!")
        else:
            print("Example", t,"FAILED! Should be", expected[i])
        i+=1

def day16_1(data):
    data = read_input(2021, "161")   
    
    transmission = data[0]
    converted = HexToBin(transmission)
    
    result = 0
    _, result, ops = processTransmission(converted)


    print("Sum packet versions:",result)
    AssertExpectedResult(951, result)


# 6364040 too low
# 46326779219 too low
# 1495959086337
def day16_2(data):
    runTestsDay16()
    
    data = read_input(2021, "161")   
    
    transmission = data[0]
    converted = HexToBin(transmission)
    
    result = 0
    _, result, ops = processTransmission(converted, [])
    result, _ = processAllOperations(ops, result)


    print("Operation result is:",result)
    AssertExpectedResult(0, result)



##### Day 17 #####

def checkIfProbeInTargetArea(probe, minX, maxX, minY, maxY):
    probeX, probeY = probe

    #print("probe X",probeX,">=", minX,probeX >= minX)
    #print("probe X",probeY,"<=", maxX,probeX <= maxX)
    #print("probe Y",probeY,">=", minY,probeY >= minY)
    #print("probe Y",probeY,"<=", maxY, probeY <= maxY)
    return probeX >= minX and probeX <= maxX and probeY >= minY and probeY <= maxY

def checkIfProbeWillNoLongerReachArea(probe, minX, maxX, minY, maxY):
    probeX, probeY = probe
    return probeX >= maxX #or probeY <= minY

def fireProbe(probe, velocity, minX, maxX, minY, maxY):
    step = 300
    maxYPostition = 0
    probeX, probeY = probe
    velocityX, velocityY = velocity

    count = 0
    for i in range(step):
        #print("step",i,"probe position:", probeX, probeY)
        probeX += velocityX
        probeY += velocityY
        if velocityX != 0:
            velocityX += -1 if velocityX > 0 else 1
        velocityY -= 1

        if probeY > maxYPostition:
            maxYPostition = probeY

        if checkIfProbeInTargetArea((probeX, probeY), minX, maxX, minY, maxY):
            count +=1
            #print("Within target area! Max pos:", maxYPostition)
            return maxYPostition, velocityX, velocityY, i, count

        if checkIfProbeWillNoLongerReachArea((probeX, probeY), minX, maxX, minY, maxY):
            #print("Will not reach target area!", (probeX, probeY), (velocityX, velocityY ) )
            return None,None,None,None, 0
    return 0,0,0,0,0
        

# Day 17, part 1: None (42.221 secs)
def day17_1(data):
    #data = read_input(2021, "171")   
    
    targetArea = data[0].split(" ")
    minX = int(targetArea[2].split("..")[0][2:])
    maxX = int(targetArea[2].split("..")[1][:-1])
    minY = int(targetArea[3].split("..")[0][2:])
    maxY = int(targetArea[3].split("..")[1])

    
    maxPosY = 0
    foundVel = (0,0)
    stop = False
    for x in range(10000):
        for y in range(10000):
            yy, velocityX, velocityY, step,_ = fireProbe((0,0), (x,y), minX, maxX, minY, maxY)
            if yy == None:
                stop = True
                break
            if yy > maxPosY:
                maxPosY = yy
                foundVel = (x,y)
                vel = (velocityX, velocityY)
                sstep = step 
        if stop:
            break
    
    print("Found initial velocity", foundVel,"with highest Y", maxPosY,"velocity", vel,"at step",sstep)    
    result = maxPosY
    print("result is:",result)
    AssertExpectedResult(3160, result)


# Day 17, part 2: None (38.974 secs)
def day17_2(data):
    #data = read_input(2021, "171")   
    
    targetArea = data[0].split(" ")
    minX = int(targetArea[2].split("..")[0][2:])
    maxX = int(targetArea[2].split("..")[1][:-1])
    minY = int(targetArea[3].split("..")[0][2:])
    maxY = int(targetArea[3].split("..")[1])

    print(minX,maxX,minY, maxY)

    count = 0
    for x in range(-300,500):
        for y in range(-300,500):
            yy, _, _, _,c = fireProbe((0,0), (x,y), minX, maxX, minY, maxY)
            if yy == None:
                continue
            if c != 0:  
                count +=1
    
    result = count
    print("result is:",result)
    AssertExpectedResult(1928, result)


if __name__ == "__main__":
    # override timeout
    main(sys.argv, globals(), AOC_EDITION_YEAR, 900)

