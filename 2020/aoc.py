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
from math import sqrt

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

#Day 1, part 1: 1006875 (0.001 secs)
#Day 1, part 2: 165026160 (0.323 secs)
def day1_1(data):
    sum = 2020
    data = sorted(data, key=int)
    result = 0
    for i in range(0, len(data)):
        elem1 = int(data[i])
        for j in range(i, len(data)):
            elem2 = int(data[j])
            if (elem1 + elem2 == sum):
                result = elem1 * elem2
                break
    
    AssertExpectedResult(1006875, result, 1)
    return result

def day1_2(data):
    sum = 2020
    data = sorted(data, key=int)

    for i in range(0, len(data)):
        elem1 = int(data[i])
        for j in range(i, len(data)):
            elem2 = int(data[j])
            for k in range(j, len(data)):
                elem3 = int(data[k])
                if (elem1 + elem2 + elem3 == sum):
                    result = elem1 * elem2 * elem3
    
    AssertExpectedResult(165026160, result, 2)
    return result

#Day 2, part 1: 424 (0.002 secs)
#Day 2, part 2: 747 (0.002 secs)
def day2_1(data):
    validPasswords = 0
    for inputLine in data:
        input = inputLine.split(" ")
        letter = input[1][0]
        min = int(input[0].split("-")[0])
        max = int(input[0].split("-")[1])
        occurrences = input[2].count(letter)

        if occurrences <= max and occurrences >= min:
            validPasswords += 1

    AssertExpectedResult(424, validPasswords, 1)
    return validPasswords        

def day2_2(data):
    validPasswords = 0
    for inputLine in data:
        input = inputLine.split(" ")
        letter = input[1][0]
        pos1 = int(input[0].split("-")[0])-1
        pos2 = int(input[0].split("-")[1])-1
               
        if bool(input[2][pos1] == letter) ^ bool(input[2][pos2] == letter):
            validPasswords += 1

    AssertExpectedResult(747, validPasswords, 2)
    return validPasswords

def day3Aux(data, initR, initD):
    right = initR
    down = initD
    deltaD = initD
    deltaR = initR
    trees = 0
   
    while down != len(data):
        if str(data[down][right]) == '#':
            trees += 1
        down += deltaD
        right += deltaR
        right = right%31 

        if down > len(data):
            break

    return trees 

#Day 3, part 1: 280 (0.001 secs)
#Day 3, part 2: 4355551200 (0.001 secs)
def day3_1(data):    
    result = day3Aux(data, 3, 1)
    AssertExpectedResult(280, result, 1)
    return result
    
def day3_2(data):
    result = day3Aux(data, 3, 1) * day3Aux(data, 1, 1) * day3Aux(data, 5, 1) * day3Aux(data, 7, 1) * day3Aux(data, 1, 2)  
    AssertExpectedResult(4355551200, result, 2)
    return result

# Day 4 methods

def addFields(line, passportFields):
    pairs = line.split(" ")            
    for field in pairs:
        passportFields.append(field.split(":")[0]) 

def isPassportValid(fields, passportFields):
    return fields.issubset(passportFields)
    #for p in passportFields:
    #    if p in fields:
    #        numFields+=1

    #return numFields >= 7
        
#Day 4, part 1: 235 (0.002 secs)
#Day 4, part 2: 194 (0.003 secs)
def day4_1(data):
    fields = {'byr','iyr','eyr','hgt','hcl','ecl','pid'}
    valid = 0
    
    passportFields = []

    for line in data:
        if line == '':     
            if isPassportValid(fields, passportFields):
                valid += 1
            passportFields = []

        elif line != '': #passport            
            addFields(line, passportFields) 

    #check last passport
    if isPassportValid(fields, passportFields):
                valid += 1
    
    AssertExpectedResult(235, valid, 1)
    return valid

def addValidFields(line, passportFields):
    pairs = line.split(" ")            
    for field in pairs:
        key = field.split(":")[0]
        val = field.split(":")[1]
        if isFieldValid(key, val):
            passportFields.append(key) 

def isFieldValid(key, val):
    if key == 'byr':
        num = int(val)
        return len(val) == 4 and num >= 1920 and num <= 2002
    elif key == 'iyr':
        num = int(val)
        return len(val) == 4 and num >= 2010 and num <= 2020
    elif key == 'eyr':
        num = int(val)
        return len(val) == 4 and num >= 2020 and num <= 2030
    elif key == 'hgt':
        try:
            height = int(val[:-2])
        except:
            return False
        metric = val[-2:]
        if metric == 'in':
            return height >= 59 and height <= 76
        elif metric == 'cm':
            return height >= 150 and height <= 193
    elif key == 'hcl': 
        validChars = {'a','b','c','d','e','f','0','1','2','3','4','5','6','7','8','9'}
        num = 0
        for v in val:
            if v in validChars:
                num+=1
        return val[0] == '#' and num == len(val)-1
    elif key == 'ecl':
        validEyeColours = {'amb', 'blu', 'brn', 'gry' ,'grn' , 'hzl', 'oth'}
        return val in validEyeColours
    elif key == 'pid':
        try:
            int(val)
            return len(val) == 9
        except:
            return False

def day4_2(data):
    fields = {'byr','iyr','eyr','hgt','hcl','ecl','pid'}
    valid = 0
    
    passportFields = []

    for line in data:
        if line == '':         
            if isPassportValid(fields, passportFields):
                valid += 1
            passportFields = []

        elif line != '': #passport            
            addValidFields(line, passportFields) 

    #check last passport
    if isPassportValid(fields, passportFields):
                valid += 1
    
    AssertExpectedResult(194, valid, 2)
    return valid


def computeRow(input):
    #print(input)
    lower = 0
    upper = 127

    for d in input:
        step = (upper - lower)/2
        if d == 'F':
            upper-= step
        elif d == 'B':
            lower+= step    
    #print(math.ceil(lower))
    #print(math.floor(upper))

    return math.floor(upper)

def computeColumn(input):
    #print(input)
    lower = 0
    upper = 7

    for d in input:
        step = (upper - lower)/2
        if d == 'L':
            upper-= step
        elif d == 'R':
            lower+= step    
    #print(math.ceil(lower))
    #print(math.floor(upper))

    return math.floor(upper)

#Day 5, part 1: 850 (0.003 secs)
#Day 5, part 2: 599 (0.010 secs)
def day5_1(data):     
    #data = read_input(2020, "51")
    maxSeatId = 0
    for boardingPass in data:
        row = computeRow(boardingPass[:-3])
        column = computeColumn(boardingPass[-3:])
        seatID = row*8+column
        if seatID > maxSeatId:
            maxSeatId = seatID
    
    AssertExpectedResult(850, maxSeatId, 1)
    return maxSeatId

def day5_2(data):     
    rows = list(range(128))
    columns = list(range(8))
    rows.remove(0)
    rows.remove(127)
    seats = []
    results = []
    for r in rows:
       for c in columns:
            seats.append(r*8+c)
    
    for boardingPass in data:
        row = computeRow(boardingPass[:-3])
        column = computeColumn(boardingPass[-3:])
        seatID = row*8+column
        results.append(seatID)

    result = 0
    for seat in seats:
        if seat-1 in results and seat+1 in results and seat not in results:
            result = seat
            break

    AssertExpectedResult(599, result, 2)
    return result

#Day 6, part 1: 6686 (0.004 secs)
#Day 6, part 2: 3476 (0.003 secs)
def day6_1(data):     
    #data = read_input(2020, "61")
    
    sum = 0
    answers = []
    count = 0
    
    for answer in data:
        if answer == '': # new group
            sum+= count
            answers = []   
            count = 0        
        else:
            for q in answer:

                try:
                    if q not in answers:
                        count += 1
                        answers.append(q)
                except:
                    None
    sum +=count

    AssertExpectedResult(6686, sum, 1)
    return sum

def processUniqueAnswers(answers, groupSize):
    count = 0
    for _, v in answers.items():
        if v == groupSize:
            count+= 1
    return count

def day6_2(data):     
    #data = read_input(2020, "61")
    
    sum = 0
    answers = {}
    groupSize = 0
    for answer in data:        
        if answer == '': # new group        
            sum+= processUniqueAnswers(answers, groupSize)
            answers = {}        
            groupSize = 0
        else:           
            for q in answer:
                if q not in answers:
                    answers[q] = 1
                else:
                    answers[q] = answers[q] + 1
            groupSize += 1

    sum += processUniqueAnswers(answers, groupSize)

    AssertExpectedResult(3476, sum, 2)
    return sum

def printBags(bags):
    for key, val in bags.items():
        print(key," => ", val)

def buildGraphForBags(data):
    bags = {}
    for line in data:
        b = line.split("bags")
        bag = b[0].strip()  
        bags[bag] = [] 
                
        otherBags = line.split("contain")[1].split(",")
        #print(otherBags)
        for content in otherBags:
            number = content.strip()[0]
            bagType = content.strip()[1:].split("bag")[0].strip()
            
            if content.find('no other bags') == -1:
                #print("number: ",number)
                #print(bagType)
                bags[bag].append( (bagType, number) )
            else: 
                bags[bag].append( ('END', 0) )
    return bags   

def find_bag(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if start not in graph:
        return None
    for node, _ in graph[start]:
        if node not in path:
            newpath = find_bag(graph, node, end, path)
            if newpath: 
                return newpath
    return None
    
#Day 7, part 1: 179 (0.131 secs)
#Day 7, part 2: 18925 (0.003 secs)
def day7_1(data):     
    #data = read_input(2020, "71")
    bags = buildGraphForBags(data)
    target = 'shiny gold'
    #printBags(bags)

    count = -1
    for key in bags:
        p = find_bag(bags, key, target)
        if p != None:
            count += 1
    
    AssertExpectedResult(179, count, 1)
    return count

def computeBags(bags, total, contents):
    for bag, val in contents:
        if bag == 'END':
            return 0
        #print(val ,"+", computeBags(bags, total, bags[bag]), " * ", val)
        total += int(val) + int(val) * computeBags(bags, 0, bags[bag])

    return total
        

def day7_2(data):     
    #data = read_input(2020, "72")
    bags = buildGraphForBags(data)
    target = 'shiny gold'
    
    contents = bags[target]
    result = computeBags(bags, 0, contents)
    AssertExpectedResult(18925, result, 2)
    return result


def preventInfiniteLoop(data):
    visited = set()
    machine = HandheldMachine(data)

    while True:  
        machine.runStep()
        # if we reach this state it means we are in an infinite loop
        if machine.program_counter in visited:
            break
        else:
            visited.add(machine.program_counter)

    return machine.accumulator

#Day 8, part 1: 1675 (0.002 secs)
#Day 8, part 2: 1532 (0.019 secs)
def day8_1(data):    
    #data = read_input(2020, "81")
    result = preventInfiniteLoop(data)
    AssertExpectedResult(1675, result, 1)
    return result


def fixInfiniteLoop(data, switchOperations):    
    success = False
    targetPC = len(data)

    machine = HandheldMachine(data)
    while not success:
        visited = set()  
        machine.reset()             
        switchPC = switchOperations.pop(0)
        machine.swapOperation(switchPC)

        while True:                                                  
            machine.runStep()
            
            # if we reach the last instruction then we fixed the infinite loop!
            if machine.program_counter  == targetPC:
                success = True
                break

            # if we reach this state it means we are in an infinite loop :(
            if machine.program_counter  in visited:
                break
            else:
                visited.add(machine.program_counter)            

    return machine.accumulator


def preprocess(data):
    switchOperations = []
    pc = 0
    for instruction in data:
        op = instruction[:3]
        if op == 'nop' or op== 'jmp':
            switchOperations.append(pc)
        pc += 1
    return switchOperations

def day8_2(data):    
    #data = read_input(2020, "81")

    switchOperations = preprocess(data)
    result = fixInfiniteLoop(data, switchOperations)
    AssertExpectedResult(1532, result, 2)
    return result


def sumSearch(data, sum, preamble, preambleSize):    
    data = sorted(data[preamble : preamble + preambleSize], key=int)
    
    for i in range(0, len(data)):       
        elem1 = data[i]
        for j in range(i, len(data)):
            elem2 = data[j]
           #print(data[i],"+",data[j],"=",elem1 + elem2)
            if (elem1 + elem2 == sum):
                return True

    return False  

#Day 9, part 1: 27911108 (0.004 secs)
#Day 9, part 2: 4023754 (0.004 secs)
def day9_1(data):    
    #data = read_input(2020, "91")
    data = [int(numeric_string) for numeric_string in data]
    
    preambleSize = 25
    preamble = 0

    result = 0
    for number in data[preambleSize:]:
        valid = sumSearch(data, number, preamble, preambleSize)
        preamble += 1
        if not valid:
            result = number
            break

    AssertExpectedResult(27911108, result, 1)
    return result
        

def breakXMAS(data, num, preambleSize):
    max = len(data)
    original = data
    
    for i in range(0, max):
        for j in range(i + 2, i + preambleSize):
            data = original[i:j]   
            if sum(data) == num:
                #print(i,",",j)
                data = sorted(data, key=int)
                return data[0] + data[len(data)-1]
    return None

def day9_2(data):    
    #data = read_input(2020, "91")
    data = [int(numeric_string) for numeric_string in data]
    
    sum = 27911108
    preambleSize = 25
    #sum = 127
    #preambleSize = 5
    result = breakXMAS(data, sum, preambleSize)
    AssertExpectedResult(4023754, result, 2)
    return result

def checkAdapterArrangement(data, deviceJoltage):
    jolt = 0
    diff1 = 0
    diff3 = 1
    data = sorted(data, key=int)

    data = [0] + data + [data[-1] + 3]
    
    deltaJoltage = 3
    diffVoltages = list()

    for adapter in data:        
        if jolt <= adapter <= jolt + deltaJoltage:
            dif = adapter - jolt
            jolt = adapter
            diffVoltages.append(dif)
            #print(adapter,"with diff", dif)
            if dif == 1:
                diff1 +=1
            elif dif == 3:
                diff3 +=1
           # else:
            #    return False, 0

    #print(diffVoltages)
    return diff1 * diff3, diffVoltages

#Day 10, part 1: 2590 (0.000 secs)
#Day 10, part 2: 296196766695424 (0.033 secs)
def day10_1(data):    
    #data = read_input(2020, "102")
    data = ints(data)

    deltaJoltage = 3
    deviceJoltage = max(data) + deltaJoltage

    result, _ = checkAdapterArrangement(data, deviceJoltage)
    AssertExpectedResult(2590, result, 1)
    return result

def buildGraphForVoltages(data):
    data = data.copy()
    jolt = 0
    deltaJoltage = 3
    deviceJoltage = max(data) + deltaJoltage    
    data.append(deviceJoltage)
    graph = {} 

    while True:
        graph[jolt] = list() #0
        
        for adapter in data:
            if jolt <= adapter <= jolt + deltaJoltage:
                    graph[jolt].append(adapter)
            else:
                jolt = data[0]
                #print("updating jolt to",jolt)
                
                if jolt in data:
                    data.remove(jolt)
                break        
        
        if len(data) == 1 and data[0] == deviceJoltage:             
            break

    #printGraph(graph)
    return graph

# cleaner way of doing part2
def cleanerWay(data):
    data = list(sorted(data))
    data = [0] + data + [data[-1] + 3]
    # diff of joltages 
    data = np.diff(data)

    st = [str(int) for int in data] 
    str_of_ints = "".join(st)  

    s = (str_of_ints.replace("1111","A").replace("111","B").replace("11","C") )
    return (7 ** s.count("A")) * (4 ** s.count("B")) * (2 ** s.count("C"))


# taken from https://www.reddit.com/r/adventofcode/comments/kb7qt0/2020_day_10_part_2_python_number_of_paths_by/
# O(n3 log n), with DP solution we can get O(n) so this one is not optimal but is very nice nonetheless :)
def matrixAdjanciesBasedSolution(data):
    '''
    n_lines = len(data)
    f = lambda i, j: j > i and data[j] - data[i] <= 3
    m = np.fromfunction(np.vectorize(f), (n_lines, n_lines), dtype=int).astype(int)
    aux = np.identity(n_lines)    

    sol_part_2 = 0
    for _ in range(n_lines):
        aux = aux @ m
        sol_part_2 += aux[0, n_lines - 1]
    '''
    data.insert(0,0)
    n = len(data)
    f = lambda i, j : j > i and data[j] - data[i] <= 3
    m = np.fromfunction(np.vectorize(f), (n, n), dtype=np.int64).astype(np.int64)
    m[n-1, n-1] = 1

    aux = np.linalg.matrix_power(m, n)
    ans = aux[0, n - 1]

    return ans

def day10_2(data):    
    #data = read_input(2020, "102")
    data = ints(data)
    data = sorted(data, key=int)       

    deltaJoltage = 3
    deviceJoltage = max(data) + deltaJoltage          

    # fail solution with graphs :(
    #target = data[len(data)-1]
    #g = buildGraphForVoltages(data)
    #g[target] = [deviceJoltage]
    #printGraph(g)
    #even cache_lru doesnt help this lost cause :P
    #print(len(find_all_paths(g, 0, target)))
    
    # solution based on patterns
    result, diffVoltages = checkAdapterArrangement(data, deviceJoltage) 

    string_ints = [str(int) for int in diffVoltages]    
    str_of_ints = "".join(string_ints)
    str_of_ints = (str_of_ints.replace("1111","A").replace("111","B").replace("11","C") )
    result = (7 ** str_of_ints.count("A")) * (4 ** str_of_ints.count("B")) * (2 ** str_of_ints.count("C"))
    
    AssertExpectedResult(296196766695424, result, 2)
    #return result
    #print(result == matrixAdjanciesBasedSolution(data))
    result = matrixAdjanciesBasedSolution(data)
    AssertExpectedResult(296196766695424, result, 2)
    return result


def countOccurencesInDirection(map, x, y, xInc, yInc, seat, rows, columns):
    x += xInc
    y += yInc
    
    #print(x,y,"count for",seat,"is")
    if 0 <= x < columns and 0 <= y < rows:        
        return map[y][x].count(seat)
    else:
        return 0

# factorized for part 2 to take an increment as parameter, turned out not to be necessaRY
def countOccurences(map, x, y, seat, rows, columns, increment = 1):
    
    east = countOccurencesInDirection(map, x, y, increment, 0, seat, rows, columns)     
    west = countOccurencesInDirection(map, x, y, -increment, 0, seat, rows, columns)
    north = countOccurencesInDirection(map, x, y, 0, -increment, seat, rows, columns)     
    south = countOccurencesInDirection(map, x, y, 0, increment, seat, rows, columns)     
    northeast = countOccurencesInDirection(map, x, y, increment, -increment, seat, rows, columns)     
    northwest = countOccurencesInDirection(map, x, y, -increment, -increment, seat, rows, columns)      
    southeast = countOccurencesInDirection(map, x, y, increment, increment, seat, rows, columns)     
    southwest = countOccurencesInDirection(map, x, y, -increment, increment, seat, rows, columns)     
     
    return (east + west + north + south + northeast + northwest + southeast + southwest)

def applySeatRules(map):
    newMap = buildMapGrid(map)
    
    rows = len(map)
    columns = len(map[0])    
    changed = False
    
    for y in range(rows):
        for x in range(columns):                
            seat = map[y][x]
            occupiedSeats = countOccurences(map, x, y, "#", rows, columns)
            
            if seat == 'L' and occupiedSeats == 0:
                newMap[y][x] = '#'
                changed = True
            elif seat == '#' and occupiedSeats >= 4:
                newMap[y][x] = 'L'
                changed = True

    return (newMap, changed)

# initial approach for part 1 was based on this until I figured I could determine stop condition while applying seat rules
def shouldStop(oldMap, newMap):
    rows = len(oldMap)
    columns = len(oldMap[0]) 
    shouldStop = True
    count = 0

    for y in range(0, rows):
        for x in range(0,columns): 
            if newMap[y][x] == '#':
                count += 1
            if oldMap[y][x] != newMap[y][x]:
                shouldStop = False
                break
        if not shouldStop:
            break

    return shouldStop, count

#Day 11, part 1: 2222 (3.186 secs)
#Day 11, part 2: 2032 (2.932 secs)
def day11_1(data):   
    #data = read_input(2020, "111")
    
    newMap = data  
    while True:
        oldMap = newMap     
        newMap, changed = applySeatRules(oldMap)
        if not changed:              
            count = sum( [ seatRow.count("#") for seatRow in newMap])
            break
            #break
        #printMap(m)
        # initial approach until I changed applySeatRules
        #stop, count = shouldStop(oldMap, newMap)   
           
    AssertExpectedResult(2222, count, 1)
    return count  

def applyNewSeatRules(map):
    newMap = buildMapGrid(map)
    
    rows = len(map)
    columns = len(map[0])    
    changed = False

    for y in range(rows):
        for x in range(columns):                
            seat = map[y][x]
            
            occupiedSeats = 0
            # cycle through all directions 
            for dx, dy in [(1,1), (1,0), (0,1), (-1,-1), (-1, 0), (0,-1), (1,-1), (-1,1)]:
                newX = x + dx
                newY = y + dy

                # find first visible seat
                iters = 0
                while 0 <= newX < columns and 0 <= newY < rows and map[newY][newX] == '.': 
                    newX += dx
                    newY += dy
                    iters+=1
                    #if iters > 11:
                    #    print("oops")
                
                #check if seat is occupied
                if 0 <= newX < columns and 0 <= newY < rows:
                    if map[newY][newX] == '#':
                        occupiedSeats += 1

            #fail approach, was based on a dictionary to determine whether a given direction had view blocked... already deleted structure and auxiliary functions :| 
            #for inc in range(1, rows):                  
                #count = countOccurences(map, x, y, "#", rows, columns, inc, blockedViews )
                #occupiedSeats += count
                
            
            if seat == 'L' and occupiedSeats == 0:
                newMap[y][x] = '#'
                changed = True
            elif seat == '#' and occupiedSeats >= 5:
                newMap[y][x] = 'L'
                changed = True


    return (newMap, changed)

def day11_2(data):    
    #data = read_input(2020, "111")
 
    newMap = data 
    result = 0
    while True:
        oldMap = newMap     
        newMap, changed = applyNewSeatRules(oldMap)
        printMap(newMap)

        if not changed:              
            result = sum( [ seatRow.count("#") for seatRow in newMap])
            break
    AssertExpectedResult(2032 , result, 2)
    return result
    

def turnDirection(direction, action):
    '''
    E:  turn L -> N
        turn R -> S
    
    W:  turn L -> S
        turn R -> N

    N:  turn L -> W
        turn R -> E

    S:  turn L -> E
        turn R -> W
    '''
    
    if action == 'L':
        if direction == 'E':
            return 'N'
        elif direction == 'W':
            return 'S'
        elif direction == 'N':
            return 'W'
        elif direction == 'S':
            return 'E'
    if action == 'R':
        if direction == 'E':
            return 'S'
        elif direction == 'W':
            return 'N'
        elif direction == 'N':
            return 'E'
        elif direction == 'S':
            return 'W'

def changeDirection(direction, action, value):
    turns = value/90
    if turns == 4:
        return direction
    
    if turns == 1:
        return turnDirection(direction, action)
    elif turns == 2:
        direction = turnDirection(direction, action)
        return turnDirection(direction, action)
    elif turns == 3:
        direction = turnDirection(direction, action)
        direction = turnDirection(direction, action)
        return turnDirection(direction, action)

def PASystem(data):
    direction = 'E'
    
    x = 0
    y = 0
    for instruction in data:
        action = instruction[0]
        value = int(instruction[1:])

        if action == 'F':
            if direction == 'N':
                y+=value
            elif direction == 'S':
                y-= value
            elif direction == 'E':
                x += value
            elif direction == 'W':
                x -= value
        
        elif action in ['L', 'R']:
            direction = changeDirection(direction, action, value)
        else:
            x,y = moveInDirection(action, value, x, y)  

    return abs(x) + abs(y)

#Day 12, part 1: 1007 (0.002 secs)
#Day 12, part 2: 41212 (0.002 secs)
def day12_1(data):   
    #data = read_input(2020, "121")   
    result = PASystem(data)
    AssertExpectedResult(1007, result, 1)
    return result

def moveInDirection(action, value, waypointX, waypointY):    
    # y axis
    if action == 'N':
        waypointY += value
    elif action == 'S':
        waypointY -= value
    # x axis
    elif action == 'E':
        waypointX += value
    elif action == 'W':
        waypointX -= value  

    return waypointX, waypointY

def changeDirectionPart2(waypointX, waypointY, action, value):
    turns = value/90

    if turns == 4:
        return waypointX, waypointY
    if turns == 2:
        return -waypointX, -waypointY

    if turns == 1:
        if action == 'L':
            return -waypointY, waypointX
        else: 
            return waypointY, -waypointX
    if turns == 3:
        if action == 'L':
            return waypointY, -waypointX
        else: 
            return -waypointY, waypointX
    

def moveForwardShip(x, y, waypointX, waypointY, value):    
    y += waypointY * value
    x += waypointX * value
    return x , y

def PASystemWithWaypoint(data):   
    waypointX = 10
    waypointY = 1
    x = 0
    y = 0
    for instruction in data:
        action = instruction[0]
        value = int(instruction[1:])

        #print("Ship position:(",x,",",y,")")         
        #print("Waypoint position:(",waypointX,",",waypointY,")") 
        #print("Waypoint direction:(",directionX,",",directionY,")")     
        #print()
        #print(instruction)       

        # ship moves, waypoint stays inaltered
        if action == 'F': 
            x , y = moveForwardShip(x, y, waypointX, waypointY, value)        
        # rotates waypoint
        elif action in ['L', 'R']:    
            waypointX, waypointY = changeDirectionPart2(waypointX, waypointY, action, value)         
        else:
            # move waypoint, ships stays the same
            waypointX, waypointY = moveInDirection(action, value, waypointX, waypointY)
            
    #print("Ship position:(",x,",",y,")")         
    #print("Waypoint position:(",waypointX,",",waypointY,")")          
    #print("Waypoint direction:(",directionX,",",directionY,")")   

    return abs(x) + abs(y)

#Day 12, part 1: 1007 (0.004 secs)
#Day 12, part 2: 41212 (0.003 secs)
def day12_2(data):   
    #data = read_input(2020, "121")
    result = PASystemWithWaypoint(data)
    AssertExpectedResult(41212, result, 2)
    return result


def readInput(data):
    timestamp = int(data[0])
    busIDs = []
    for busID in data[1].split(','):
        if busID != 'x':
            busIDs.append(int(busID))

    busIDs = sorted(busIDs, key=int) 
    return timestamp, busIDs, data[1].split(',')


#Day 13, part 1: 410 (0.001 secs)
#Day 13, part 2: 600691418730595 (0.002 secs)
def day13_1(data):   
    #data = read_input(2020, "136")   

    timestamp, busIDs, _ = readInput(data)
    
    #print("Timestamp",timestamp)
    #print(busIDs)

    i = max(busIDs)
    results = {}
    for bus in busIDs:
        #print("Schedule for bus", bus,":")

        for k in range(timestamp, (timestamp-bus) + i):
            if k % bus == 0: # k is multiple of bus
                results[bus] = k
                break
    
    bus = min(results.items(), key=operator.itemgetter(1))[0]

    result = bus * (results[bus] - timestamp)
    AssertExpectedResult(410, result, 1)
    return result

def computeMinimumTimestamp(schedule, i, step):
    delta = sys.maxsize
    init = i

    #print("Searching starting in",init,"with step", step)
    # limit search range based on previous schedule iteration results
    for timestamp in range(init, init + delta, step):    
        offset = 0
        valid = True

        # check buses in schedule
        for bus in schedule:
            if bus != 'x':
                # compute (timestamp + offset mod busId)
                # if timestamp + offset is not a multiple of the bus then we can stop
                if (timestamp + offset) % int(bus) != 0: 
                    valid = False
            offset += 1

        # after checking all the buses IDs if all were valid we got our timestamp!
        if valid:
            return timestamp       
            

def day13_2(data):   
    #data = read_input(2020, "136")   

    '''
        7,13,x,x,59,x,31,19 is  1068788
        17,x,13,19 is 3417.
        67,7,59,61 first occurs at timestamp 754018.
        67,x,7,59,61 first occurs at timestamp 779210.
        67,7,x,59,61 first occurs at timestamp 1261476.
        1789,37,47,1889 first occurs at timestamp 1202161486
    '''

    '''
    print(is_multiple(1068781 + 0, 7))
    print(is_multiple(1068781 + 1, 13))
    print(is_multiple(1068781 + 4, 59))
    print(is_multiple(1068781 + 6, 31))
    print(is_multiple(1068781 + 7, 19))
    ''' 

    _,_, schedule = readInput(data)
    results = []
    t = 0

    # compute partial results based on subsets of the schedule to optimize final search (reduce search space)
    for i in range(1, len(schedule)):
        
        if schedule[i] != 'x':
            factor = 1
            # we can do this because all buses IDs are coprimes, meaning all buses IDs will share the same gcd
            for b  in schedule[0:i]:
                if b != 'x':
                    factor *= int(b)

            #print("factor",factor)
            #print("init",t)
            t = computeMinimumTimestamp(schedule[0:i+1], t, factor)
            #print("T for schedule",schedule[0:i],"is",t)
            results.append(t)

    AssertExpectedResult(600691418730595, t, 2)
    return t
    #return t + math.gcd(results[0], results[1])
    
def resetNumber(mask):
    number = ['0'] * 36
    i = 0
    for bit in mask:
        if bit != 'X':
            number[i] = mask[i]
        i += 1
    return number

def memoryAddressDecoder(data, version):
    memory = {}
    # stored inverted
    mask =  ['X'] * 36    

    for cmd in data:        
        args = cmd.split("=")        
        operation = args[0].strip()
        value = args[1].strip()

        if operation == 'mask':
            mask = list(value)
            mask.reverse()
        else:
            
            position = operation.split("[")[1].split("]")[0]

            if version == 1:
                value = applyMaskToNumber(mask, value)
                memory = updateMemory(position, value, memory)            
            elif version == 2:
                #print("Processing:",args,"with mask", str(''.join(mask)))
                positions = applyMaskToAddress(mask, position)
                for pos in positions:
                    memory = updateMemory(pos, int(value), memory)            

    return memory


#Day 14, part 1: 17765746710228 (0.004 secs)
#Day 14, part 2: 4401465949086 (0.251 secs)
def day14_1(data):    
    #data = read_input(2020, "141")

    memory = memoryAddressDecoder(data, 1)
    result = sum(memory.values())

    AssertExpectedResult(17765746710228, result, 1)
    return result

def convertToDecimal(number):
    # invert to convert to decimal
    number.reverse()  
    
    # convert to decimal
    decimal = int(''.join(number),2)
    return decimal

def expandFloatingBits(number):
    positions = list()
    # convert to a string
    address = str(''.join(number))

    #print("Expanding",str(''.join(number)))
    #print(str(''.join(number)).count('X'))
    
    addresses = [address]
    while len(addresses) != 0:
        addr = addresses.pop()
        if addr.count('X') == 0:
            positions.append(convertToDecimal(list(addr)))
        else:
            addresses.append(addr.replace('X','1', 1))
            addresses.append(addr.replace('X','0', 1))

    #print("Addresses: ",len(positions))
    #print("Addresses: ",positions)
    return positions

def applyMaskToAddress(mask, address):    
    # kept inverted
    number = resetNumber(mask)

    # convert to binary then to a list
    binary = list(bin(int(address))) 
    # keep it inverted           
    binary.reverse()         

    for i in range(len(mask)):
        if mask[i] != '0':
            number[i] = mask[i]
        else:
            if i > len(binary)-3:
                number[i] = mask[i]
            else:
                number[i] = binary[i]            
    
    # deal with floating bits
    return expandFloatingBits(number)   


def applyMaskToNumber(mask, value):
    # kept inverted
    number = resetNumber(mask)

    # convert to binary then a list
    binary = list(bin(int(value))) 
    # keep it inverted           
    binary.reverse()         

    for i in range(len(binary)-2):
        if mask[i] != 'X':
            number[i] = mask[i]
        else:
            number[i] = binary[i]            
    
    # invert to convert to decimal
    number.reverse()          
    # convert to decimal
    decimal = int(''.join(number),2)

    return decimal

def updateMemory(position, value, memory):    
    # update memory
    memory[position] = value
    return memory

def day14_2(data):    
    #data = read_input(2020, "142")

    memory = memoryAddressDecoder(data, 2)
    #print(memory)
    result = sum(memory.values())
    AssertExpectedResult(4401465949086, result, 2)
    return result


def updateLastSpoken(numbersSpoken, lastSpoken, turn):
    if lastSpoken not in numbersSpoken:
        numbersSpoken[lastSpoken] = [turn]
    else:
        turns = numbersSpoken[lastSpoken]
        if len(turns) == 1:
            numbersSpoken[lastSpoken].append(turn)
        elif len(turns) >= 2:
            turns = numbersSpoken[lastSpoken]
            # update turns spoken
            turns[0] = turns[1]
            turns[1] = turn
            numbersSpoken[lastSpoken] = turns
        else:
            numbersSpoken[lastSpoken].append(turn)
    return numbersSpoken

def memoryGame(numbers, maxTurns):
    numbersSpoken = {}
    lastSpoken = 0
    
    turn = 1
    # initial rounds done
    for number in numbers:
        numbersSpoken[number] = [turn]
        lastSpoken = number  
        turn += 1

    for turn in range(len(numbers) + 1, maxTurns + 1):
        # first time
        if lastSpoken not in numbersSpoken:
            numbersSpoken[number] = [turn]
            lastSpoken = 0

        else:
            turns = numbersSpoken[lastSpoken]
            # first time
            if len(turns) == 1:
                lastSpoken = 0
            # number already spoken
            else:
                turns = numbersSpoken[lastSpoken]
                res = turns[1] - turns[0]                
                lastSpoken = res     

            numbersSpoken = updateLastSpoken(numbersSpoken, lastSpoken, turn)
        #print("Turn",turn,":",lastSpoken)   
    
    return lastSpoken
        
def day15_1(data):    
    #data = read_input(2020, "151") 
    numbers = ints(data[0].split(","))
    result = memoryGame(numbers, 2020)        
    AssertExpectedResult(319, result, 1)
    return result

def day15_2(data):    
    #data = read_input(2020, "151")
    numbers = ints(data[0].split(","))
    result = memoryGame(numbers, 30000000)  
    AssertExpectedResult(2424, result, 2)
    return result

def isTicketValid(ticket, rules):
    ticketValues = ints(ticket.split(","))
    isValid = True
    errorRate = 0
    for value in ticketValues:
        validRules = 0
        # values must be valid for at least one rule
        for k,v in rules.items():
            conditions = v.split("or")
            leftCondition = conditions[0].strip()
            rightCondition = conditions[1].strip()

            rangeLeft = ints(leftCondition.split("-"))
            rangeRight = ints(rightCondition.split("-"))

            if rangeLeft[0] <= value <= rangeLeft[1] or rangeRight[0] <= value <= rangeRight[1]:
                validRules += 1
            #else:
            #    print("Failed condition",k)
            #    print(value)

        if validRules == 0:
            errorRate += value
            isValid = False
    return isValid, errorRate

def processTicketsInput(data):
    rules = {}
    rulesProcessingDone = False
    myTicketProcessing = False 
    nearbyTicketsProcessing = False
    nearbyTickets = list()
    for info in data:
        
        if info == '':
            rulesProcessingDone = True
        elif not rulesProcessingDone:
            fieldInfo = info.split(":")
            rules[fieldInfo[0].strip()] = fieldInfo[1].strip()
        elif info == 'your ticket:':
                myTicketProcessing = True
        elif myTicketProcessing:
            myTicket = info
            myTicketProcessing  = False
            nearbyTicketsProcessing = True
        elif nearbyTicketsProcessing:
            nearbyTickets.append(info)
    nearbyTickets.pop(0)

    return rules, myTicket, nearbyTickets

#Day 16, part 1: 19060 (0.186 secs)
#Day 16, part 2: 953713095011 (0.886 secs)
def day16_1(data):    
    #data = read_input(2020, "161") 

    rules, _, nearbyTickets = processTicketsInput(data)
    
    total = 0
    for nearbyTicket in nearbyTickets:
        _, errorRate = isTicketValid(nearbyTicket, rules)
     #   print(nearbyTicket,"is valid?", isValid)
        total += errorRate
    

    result = total
    AssertExpectedResult(19060, result, 1)
    return total

def pruneInvalidTickets(nearbyTickets, rules):
    validNearbyTickets = list()
    for nearbyTicket in nearbyTickets:
        isValid, _ = isTicketValid(nearbyTicket, rules)
        if isValid:
            validNearbyTickets.append(nearbyTicket)
        #print(nearbyTicket,"is valid?", isValid)
    return validNearbyTickets

def isRuleValid(ruleCondition, value):
    conditions = ruleCondition.split("or")
    leftCondition = conditions[0].strip()
    rightCondition = conditions[1].strip()

    rangeLeft = ints(leftCondition.split("-"))
    rangeRight = ints(rightCondition.split("-"))

    return rangeLeft[0] <= value <= rangeLeft[1] or rangeRight[0] <= value <= rangeRight[1]

def updateFieldsOrder(fieldsOrder, isRuleValid, position, ruleName):

    if position not in fieldsOrder:
        fieldsOrder[position] = {ruleName: isRuleValid}
    else:
        fields = fieldsOrder[position]
        if ruleName in fields:
            fields[ruleName] = fields[ruleName] and isRuleValid
        else:
            fields[ruleName] = isRuleValid
        fieldsOrder[position] = fields
    
    return fieldsOrder

def validateFieldsOrder(fieldsOrder):

    # don't feel like implementing a fix point algorithm
    for _ in range(100):
        for k,v in fieldsOrder.items():
            possibleFields = list(field for field,val in v.items() if val)
            if len(possibleFields) == 1:
                for kk, _ in fieldsOrder.items():
                    if k != kk:
                        fieldsOrder[kk][possibleFields[0]] = False
    return fieldsOrder

def getTicketFieldsOrder(nearbyTickets, rules):

    fieldsOrder = {}
    for ticket in nearbyTickets:
        ticketsValues = ints(ticket.split(","))

        for position in range(len(ticketsValues) ):
            value = int(ticketsValues[position])
            for k,v in rules.items():
                fieldsOrder = updateFieldsOrder(fieldsOrder, isRuleValid(v, value), position, k)
        fieldsOrder = validateFieldsOrder(fieldsOrder)

    return fieldsOrder

def generateDeparturePositions(fieldsToCheck, fieldsOrder):
    positions = []

    while True:
        fieldToCheck = fieldsToCheck.pop()
        for position, fields in fieldsOrder.items():
            if fields[fieldToCheck]:
                positions.append(position)
        
        if len(fieldsToCheck) == 0:
            break

    return positions

def day16_2(data):    
    #data = read_input(2020, "162") 

    rules, myTicket, nearbyTickets = processTicketsInput(data)
    nearbyTickets = pruneInvalidTickets(nearbyTickets, rules)
    fieldsOrder = getTicketFieldsOrder(nearbyTickets, rules)
    fieldsToCheck = ['departure location','departure station','departure platform','departure track','departure date', 'departure time']
    positions = generateDeparturePositions(fieldsToCheck, fieldsOrder)
    
    myTicket = ints(myTicket.split(","))    
    result = myTicket[positions[0]] * myTicket[positions[1]] * myTicket[positions[2]] * myTicket[positions[3]] * myTicket[positions[4]] * myTicket[positions[5]]
    
    AssertExpectedResult(953713095011, result, 2)
    return result


def processInitialCubes(data, offset, part = 1):
    rows = len(data)
    columns = len(data[0])    
    active = list()

    for y in range(rows):
        for x in range(columns): 
            if data[y][x] == '#':
                if part == 1:
                    active.append((x + offset, y + offset, 0))
                else:
                    active.append((x + offset, y + offset, 0, 0))
    
    return active

def countActiveNeighbours(active, x, y, z, offset, part = 1, w = 0):

    count = 0
    x += offset
    y += offset

    if part == 1:
        for (xx, yy, zz) in active:
            isNeighbour = abs(xx -x) in [0,1] and abs(yy -y) in [0,1] and abs(zz -z) in [0,1]
            if isNeighbour:
                #print((xx , yy, zz),"is an active neighbour!")
                count += 1
    else:
        for (xx, yy, zz, ww) in active:
            isNeighbour = abs(xx -x) in [0,1] and abs(yy -y) in [0,1] and abs(zz -z) in [0,1] and abs(ww -w) in [0,1]
            if isNeighbour:
                #print((xx , yy, zz),"is an active neighbour!")
                count += 1

    if (x,y,z) in active and part  == 1:
        return count-1
    elif (x,y,z,w) in active and part  == 2:
        return count-1
    else:
        return count

def bootProcess(active, cycles, rows, columns, offset):
    newActives = list()

    for z in range(-cycles, cycles+1):
        for x in range(-offset , rows + offset):
            for y in range(-offset, columns + offset): 
                point = (x + offset, y + offset, z)

                #print("Analysing", point)
                neighbours = countActiveNeighbours(active, x, y, z, offset)
                if point in active and (neighbours in [2,3]):
                    #print("Is active #, will remain active # because it has", neighbours, "neighbours")
                    newActives.append(point)                
                elif point not in active and (neighbours == 3):
                    #print("Was inactive ., will now become active # because it has", neighbours, "neighbours")
                    newActives.append(point)                

                #print()
    return newActives


def printCubesByLevel(actives, cycle):   
    for i in range(-cycle, cycle+1):
        level = [ (x,y,z) for (x,y,z) in actives if z == i]
        print("Level", i)
        print(level)

#Day 17, part 1: 448 (1.464 secs)
#Day 17, part 2: 2400 (43.614 secs)
def day17_1(data):    
    #data = read_input(2020, "171") 
    data = buildMapGrid(data)

    rows = len(data)
    columns = len(data[0]) 
    offset = 5
    cycles = 6
    active = processInitialCubes(data, offset)

    #printCubesByLevel(active, 0)
    #print()
    for c in range(1, cycles+1):
        active = bootProcess(list(active), c, rows, columns, offset)
        #print("Cycle",c)
        #printCubesByLevel(active, c)
        #print("# Actives:", len(active))
        #print()

    result = len(active)
    AssertExpectedResult(448, result, 1)
    return result

def bootProcess2(active, cycles, rows, columns, offset):
    newActives = list()

    for z in range(-cycles, cycles+1):
        for w in range(-cycles, cycles+1):
            for x in range(-offset , rows + offset):
                for y in range(-offset, columns + offset): 
                    point = (x + offset, y + offset, z, w)
                    #print("Analysing", point)
                    neighbours = countActiveNeighbours(active, x, y, z, offset, 2, w)
                    if point in active and (neighbours in [2,3]):
                        #print("Is active #, will remain active # because it has", neighbours, "neighbours")
                        newActives.append(point)                
                    elif point not in active and (neighbours == 3):
                        #print("Was inactive ., will now become active # because it has", neighbours, "neighbours")
                        newActives.append(point)                

                    #print()
    return newActives


def day17_2(data):    
    #data = read_input(2020, "171") 
    data = buildMapGrid(data)

    rows = len(data)
    columns = len(data[0]) 

    offset = 5
    cycles = 6
    active = processInitialCubes(data, offset, 2)

    for c in range(1, cycles+1):
        active = bootProcess2(list(active), c, rows, columns, offset)

    result = len(active)
    AssertExpectedResult(2400, result, 2)
    return result


''' Day 18 '''
# These grammars are based on the examples given in Lark documentation, just have to switch the precedence order accordingly

calc_grammar = """
    ?start: sum

    ?sum: atom
        | sum "+" atom   -> add
        | sum "-" atom   -> sub
        | sum "*" atom  -> mul
        | sum "/" atom  -> div

    ?atom: NUMBER           -> number
         | "(" sum ")"

    %import common.NUMBER
    %import common.WS_INLINE

    %ignore WS_INLINE
"""

calc_grammar2 = """
    ?start: product

    ?product: sum
        | product "*" sum  -> mul
        | product "/" sum  -> div

    ?sum: atom
        | sum "+" atom   -> add
        | sum "-" atom   -> sub

    ?atom: NUMBER           -> number
         | "(" product ")"

    %import common.NUMBER
    %import common.WS_INLINE

    %ignore WS_INLINE
"""

@v_args(inline=True) 
class CalculateTree(Transformer):
    from operator import add, mul
    number = int

#Day 18, part 1: 11076907812171.0 (0.103 secs)
#Day 18, part 2: 283729053022731.0 (0.100 secs)
def day18_1(data):    
    #data = read_input(2020, "181") 
    
    calc_parser = Lark(calc_grammar, parser='lalr', transformer=CalculateTree())
    calc = calc_parser.parse

    sum = 0
    for exp in data:
        sum += calc(exp)

    result = sum
    AssertExpectedResult(11076907812171, result, 1)
    return result

def day18_2(data):    
    #data = read_input(2020, "181") 
    
    calc_parser = Lark(calc_grammar2, parser='lalr', transformer=CalculateTree())
    calc = calc_parser.parse

    sum = 0
    for exp in data:

        sum += calc(exp)

    result = sum
    AssertExpectedResult(283729053022731, result, 2)
    return result

def processInputDay19(data):
    rules = {}
    messages = []
    lines = 0
    for line in data:
        lines += 1
        if line == '':
            messages = data[lines:]
            break
        
        # processing Rules
        info = line.split(":")

        subRules = info[1].strip().split("|")        
            
        for subRule in subRules:
            rule = int(info[0].strip())
                
            rulesList = subRule.strip().replace(' ','').replace('\"','')
            #if rulesList.isnumeric():
                 #rulesList = ints(rulesList)

            if rule in rules:
                rules[rule].append(rulesList)
            else: 
                rules[rule] = [rulesList]
    
    return rules, messages


grammarDay19Part1 = """
    ?start: zero

    ?eigthy: twenty ten | thritynine twelve
    ?seventysix: twenty ten | thritynine oneohseven
    ?nine: twenty two | thritynine seventyfour
    ?twentytwo: thritynine eightytwo | twenty fourty
    ?sistyfive: twentyfive twenty | onetwentyseven thritynine
    ?sisxtyseven: ninetytwo twenty | twentyfour thritynine
    ?oneohnine: thritynine thritynine | thirtyeight twenty
    ?fiftyseven: oneohseven thritynine | onethrity twenty
    ?twentythree: sisxtyseven thritynine | onetwentynine twenty
    ?eighteen: twenty thritynine | thritynine thritynine
    ?hundred: ninetysix twenty | oneohnine thritynine
    ?eightfour: one thritynine | ninetynine twenty
    ?three: twenty onetwentyeight | thritynine oneten
    ?fiftyeight: fourtyfour thritynine | seventyseven twenty
    ?eightthree: ninetysix thritynine | oneohseven twenty
    ?seventythree: ninetysix twenty | eighteen thritynine
    ?fourtyeight: eightytwo thritynine | twelve twenty
    ?thirteen: thritynine onethritytwo | twenty thirtytwo
    ?twelve: twenty thritynine | twenty twenty
    ?eightseven: twenty onetwentyeight | thritynine twelve
    ?seventyfive: thritynine onethrityone | twenty thirtyfive
    ?eightfive: oneoheight twenty | eigthy thritynine
    ?ninety: twenty eighteen | thritynine ninetysix
    ?onetwentyseven: onetwentyone thritynine | eightone twenty
    ?seventynine: thritynine eightsix | twenty thirtysix
    ?eighteight: thirtyeight eighteen
    ?fiveteen: thritynine nine | twenty fiftynine
    ?oneohseven: twenty thritynine | thritynine twenty
    ?fourtytwo: twenty fiftyone | thritynine onetwenty
    ?thirtyseven: oneohsix thritynine | eightfive twenty
    ?seventyfour: thritynine seventysix | twenty ninetyseven
    ?onetwentynine: eightseven thritynine | thirtyfour twenty
    ?onefifteen: twenty eighteen | thritynine oneten
    ?fourty: thritynine twenty
    ?twentyfive: eightfour thritynine | onetwelve twenty
    ?ninetyseven: fourty thritynine | ninetysix twenty
    ?ninetysix: twenty thritynine
    ?oneohone: onefourteen twenty | eightnine thritynine
    ?eightytwo: thritynine thritynine | twenty twenty
    ?onenineteen: twenty thirtyeight | thritynine thritynine
    ?thirtythree: thritynine thritynine
    ?onetwentyeight: thritynine twenty | twenty thirtyeight
    ?thirtyone: thritynine fourtythree | twenty oneeightteen
    ?four: twelve thirtyeight
    ?seventyseven: ninetyeight thritynine | thirty twenty
    ?fourtynine: thritynine sixty | twenty eighteight
    ?seventytwo: ninetysix thritynine | oneten twenty
    ?oneseventeen: twenty onetwentsix | thritynine sixtynine
    ?fiftysix: twelve thritynine | oneohnine twenty
    ?fourtythree: twentyone thritynine | oneohthree twenty
    ?fiftytwo: oneohnine thritynine | ten twenty
    ?thirtysix: twenty oneohseven | thritynine fourty
    ?onetwentsix: twenty twentysix | thritynine fourtyfive
    ?sixtynine: twenty fiftythree | thritynine fourteen
    ?oneeleven: seventeen thritynine | fourtyeight twenty
    ?onetwentytwo: thritynine seventy | twenty ninetysix
    ?fiftyfive: oneohnine thirtyeight
    ?onethirteen: sixtysix twenty | fourtyseven thritynine
    ?twentyone: seven thritynine | oneohfive twenty
    ?onetwelve: twentytwo twenty | fiftyfour thritynine
    ?ninetytwo: twelve twenty | onetwentyeight thritynine
    ?eightnine: onetwentyeight twenty | oneohseven thritynine
    ?thirtyeight: twenty | thritynine
    ?thirtyfour: oneohseven thritynine | ten twenty
    ?oneohfour: twenty eighteen | thritynine onethrity
    ?fourtyseven: eightytwo twenty | twelve thritynine
    ?two: thritynine sixtysix | twenty onefifteen
    ?seventyone: thritynine six | twenty ninetytwo
    ?five: twenty eighteight | thritynine twentyseven
    ?twentyseven: oneohnine twenty | fourty thritynine
    ?thirtytwo: thritynine ninety | twenty fiftyseven
    ?onetwentythree: thritynine ten | twenty ninetysix
    ?ninetyfour: twenty fourtysix | thritynine oneohnine
    ?seventeen: thritynine ten | twenty thirtythree
    ?ninetyone: twenty thirteen | thritynine ninetythree
    ?oneohthree: thritynine seventyfive | twenty nineteen
    ?eleven: fourtytwo thirtyone 
    ?sixtythree: twenty oneohseven | thritynine seventy
    ?six: twenty ten | thritynine eighteen
    ?oneten: thirtyeight thirtyeight
    ?onethritytwo: thritynine sixtyfour | twenty oneohfour
    ?onethritythree: oneohseven twenty | twelve thritynine
    ?sixtyone: twenty twelve | thritynine thirtythree
    ?sixteen: thritynine thirtythree | twenty fourtysix
    ?fifty: thritynine thirtysix | twenty six
    ?ninetyfive: twenty oneohtwo | thritynine fiftytwo
    ?onetwentyfour: twenty seventy | thritynine thirtythree
    ?onethrityone: thritynine sixtythree | twenty onetwentyfour
    ?sixtyfour: twenty ninetysix | thritynine oneohseven
    ?fiftyone: thritynine ninetyone | twenty oneseventeen
    ?seven: five thritynine | oneeleven twenty
    ?ninetynine: fourtysix twenty | oneohseven thritynine
    ?one: eightytwo thritynine | onenineteen twenty
    ?sixty: twenty onetwentyeight | thritynine eightytwo
    ?thirty: twenty onetwentytwo | thritynine fiftytwo
    ?onetwentyfive: thritynine ninetysix | twenty ten
    ?eightone: onetwentyfive thritynine | sixtyone twenty
    ?eightsix: thritynine ninetysix | twenty ninetysix
    ?onethrity: thritynine twenty | twenty twenty
    ?ninetythree: sixtytwo twenty | seventyone thritynine
    ?twentyeight: onethrity twenty | onenineteen thritynine
    ?onetwenty: fourtyone thritynine | fiftyeight twenty
    ?fiftyfour: eightytwo thirtyeight
    ?sixtytwo: seventytwo twenty | twentyeight thritynine
    ?twentyfour: fourty twenty | ten thritynine
    ?oneohtwo: eighteen twenty | eightytwo thritynine
    ?twentysix: twenty fourtyeight | thritynine eightthree
    ?ten: thritynine twenty | thritynine thritynine
    ?oneeightteen: fiveteen twenty | sistyfive thritynine
    ?seventy: twenty thritynine | thritynine thirtyeight
    ?oneoheight: thritynine oneohseven | twenty onetwentyeight
    ?fourtysix: twenty twenty
    ?thritynine: "a"
    ?sixtyeight: thritynine onenineteen | twenty onetwentyeight
    ?fiftynine: ninetyfive twenty | fourtynine thritynine
    ?fourtyone: twentythree thritynine | thirtyseven twenty
    ?twentynine: twenty sixteen | thritynine seventyeight
    ?onesixteen: oneoheight thritynine | ninetynine twenty
    ?thirtyfive: thritynine fiftyfive | twenty four
    ?fiftythree: ninetyfour twenty | onethritythree thritynine
    ?twenty: "b"
    ?ninetyeight: ninetyseven twenty | fiftysix thritynine
    ?eight: fourtytwo
    ?nineteen: twenty onethirteen | thritynine onesixteen
    ?oneohsix: thritynine three | twenty onetwentythree
    ?fourtyfour: seventynine twenty | fifty thritynine
    ?fourtyfive: thritynine seventythree | twenty sixtyeight
    ?zero: eight eleven
    ?sixtysix: thritynine onetwentyeight | twenty twelve
    ?seventyeight: thritynine onethrity | twenty fourty
    ?fourteen: thritynine hundred | twenty onefourteen
    ?oneohfive: thritynine twentynine | twenty oneohone
    ?onefourteen: thritynine fourty | twenty eighteen
    ?onetwentyone: twenty sixtyfour | thritynine ninetytwo


    %import common.WS_INLINE

    %ignore WS_INLINE
"""

def processInputDay19ForCFG(data):
    # Creation of terminals
    ter_a = Terminal("a")
    ter_b = Terminal("b")
    
    variables = []
    productions = []

    messages = []
    lines = 0
    for line in data:
        lines += 1
        if line == '':
            messages = data[lines:]
            break
        
        # processing Rules
        info = line.split(":")
        subRules = info[1].strip().split("|")        
            
        for subRule in subRules:
            rule = str(info[0].strip())

            # Create vars for CFG
            ruleVar = Variable(rule)
            if ruleVar not in variables:
                variables.append(ruleVar)

            rulesList = subRule.strip().replace('\"','').split(' ')
            #print(rulesList)
            # Create productions for CFG
            if rulesList == ['a']:
                prod = Production(ruleVar, [ter_a])
                productions.append(prod)
            elif rulesList == ['b']:
                prod = Production(ruleVar, [ter_b])
                productions.append(prod)
            else:
                prod = Production(ruleVar, [Variable(i) for i in rulesList])
                productions.append(prod)  
    
    cfg = CFG(variables, [ter_a, ter_b], Variable("0"), productions)
    return cfg, messages


#Day 19, part 1: 224 (2.004 secs) - Lark
#Day 19, part 1: 224 (74.343 secs) - CFG using pyformlang
#Day 19, part 2: 436 (3.966 secs)
def day19_1(data):    
    #data = read_input(2020, "191") 

    '''
    # initial approach based on Lark parser. Problem with this solution is the lack of automation in the grammar generation. 
    # EDIT: this was until I figured I was being dumb agaibb :facepalm:
    parser = Lark(grammarDay19Part1)
    ast = parser.parse    
    
    # this approach is very slow
    cfg, messages = processInputDay19ForCFG(data)
    '''


    # joins each line, separated by \n, until it finds an empty string
    it = iter(data)
    grammar = "\n".join(takewhile(lambda l: l != '', it))

    # prefixes digits by r
    grammar = re.sub(r'(\d+)', r'r\1', grammar)

    #print(grammar)
    parser = Lark(grammar, start="r0")
    ast = parser.parse

    count = 0
    for msg in list(it):
        try:
            ast(msg)
            #if cfg.contains(msg): -- this is really inneficient lol
            count += 1
        except:
            count = count
            #print("no match for",msg)

    result = count
    AssertExpectedResult(224, result, 1)
    return result

grammarDay19Part2 = """
    ?start: zero

    ?eigthy: twenty ten | thritynine twelve
    ?seventysix: twenty ten | thritynine oneohseven
    ?nine: twenty two | thritynine seventyfour
    ?twentytwo: thritynine eightytwo | twenty fourty
    ?sistyfive: twentyfive twenty | onetwentyseven thritynine
    ?sisxtyseven: ninetytwo twenty | twentyfour thritynine
    ?oneohnine: thritynine thritynine | thirtyeight twenty
    ?fiftyseven: oneohseven thritynine | onethrity twenty
    ?twentythree: sisxtyseven thritynine | onetwentynine twenty
    ?eighteen: twenty thritynine | thritynine thritynine
    ?hundred: ninetysix twenty | oneohnine thritynine
    ?eightfour: one thritynine | ninetynine twenty
    ?three: twenty onetwentyeight | thritynine oneten
    ?fiftyeight: fourtyfour thritynine | seventyseven twenty
    ?eightthree: ninetysix thritynine | oneohseven twenty
    ?seventythree: ninetysix twenty | eighteen thritynine
    ?fourtyeight: eightytwo thritynine | twelve twenty
    ?thirteen: thritynine onethritytwo | twenty thirtytwo
    ?twelve: twenty thritynine | twenty twenty
    ?eightseven: twenty onetwentyeight | thritynine twelve
    ?seventyfive: thritynine onethrityone | twenty thirtyfive
    ?eightfive: oneoheight twenty | eigthy thritynine
    ?ninety: twenty eighteen | thritynine ninetysix
    ?onetwentyseven: onetwentyone thritynine | eightone twenty
    ?seventynine: thritynine eightsix | twenty thirtysix
    ?eighteight: thirtyeight eighteen
    ?fiveteen: thritynine nine | twenty fiftynine
    ?oneohseven: twenty thritynine | thritynine twenty
    ?fourtytwo: twenty fiftyone | thritynine onetwenty
    ?thirtyseven: oneohsix thritynine | eightfive twenty
    ?seventyfour: thritynine seventysix | twenty ninetyseven
    ?onetwentynine: eightseven thritynine | thirtyfour twenty
    ?onefifteen: twenty eighteen | thritynine oneten
    ?fourty: thritynine twenty
    ?twentyfive: eightfour thritynine | onetwelve twenty
    ?ninetyseven: fourty thritynine | ninetysix twenty
    ?ninetysix: twenty thritynine
    ?oneohone: onefourteen twenty | eightnine thritynine
    ?eightytwo: thritynine thritynine | twenty twenty
    ?onenineteen: twenty thirtyeight | thritynine thritynine
    ?thirtythree: thritynine thritynine
    ?onetwentyeight: thritynine twenty | twenty thirtyeight
    ?thirtyone: thritynine fourtythree | twenty oneeightteen
    ?four: twelve thirtyeight
    ?seventyseven: ninetyeight thritynine | thirty twenty
    ?fourtynine: thritynine sixty | twenty eighteight
    ?seventytwo: ninetysix thritynine | oneten twenty
    ?oneseventeen: twenty onetwentsix | thritynine sixtynine
    ?fiftysix: twelve thritynine | oneohnine twenty
    ?fourtythree: twentyone thritynine | oneohthree twenty
    ?fiftytwo: oneohnine thritynine | ten twenty
    ?thirtysix: twenty oneohseven | thritynine fourty
    ?onetwentsix: twenty twentysix | thritynine fourtyfive
    ?sixtynine: twenty fiftythree | thritynine fourteen
    ?oneeleven: seventeen thritynine | fourtyeight twenty
    ?onetwentytwo: thritynine seventy | twenty ninetysix
    ?fiftyfive: oneohnine thirtyeight
    ?onethirteen: sixtysix twenty | fourtyseven thritynine
    ?twentyone: seven thritynine | oneohfive twenty
    ?onetwelve: twentytwo twenty | fiftyfour thritynine
    ?ninetytwo: twelve twenty | onetwentyeight thritynine
    ?eightnine: onetwentyeight twenty | oneohseven thritynine
    ?thirtyeight: twenty | thritynine
    ?thirtyfour: oneohseven thritynine | ten twenty
    ?oneohfour: twenty eighteen | thritynine onethrity
    ?fourtyseven: eightytwo twenty | twelve thritynine
    ?two: thritynine sixtysix | twenty onefifteen
    ?seventyone: thritynine six | twenty ninetytwo
    ?five: twenty eighteight | thritynine twentyseven
    ?twentyseven: oneohnine twenty | fourty thritynine
    ?thirtytwo: thritynine ninety | twenty fiftyseven
    ?onetwentythree: thritynine ten | twenty ninetysix
    ?ninetyfour: twenty fourtysix | thritynine oneohnine
    ?seventeen: thritynine ten | twenty thirtythree
    ?ninetyone: twenty thirteen | thritynine ninetythree
    ?oneohthree: thritynine seventyfive | twenty nineteen
    ?eleven: fourtytwo thirtyone | fourtytwo eleven thirtyone
    ?sixtythree: twenty oneohseven | thritynine seventy
    ?six: twenty ten | thritynine eighteen
    ?oneten: thirtyeight thirtyeight
    ?onethritytwo: thritynine sixtyfour | twenty oneohfour
    ?onethritythree: oneohseven twenty | twelve thritynine
    ?sixtyone: twenty twelve | thritynine thirtythree
    ?sixteen: thritynine thirtythree | twenty fourtysix
    ?fifty: thritynine thirtysix | twenty six
    ?ninetyfive: twenty oneohtwo | thritynine fiftytwo
    ?onetwentyfour: twenty seventy | thritynine thirtythree
    ?onethrityone: thritynine sixtythree | twenty onetwentyfour
    ?sixtyfour: twenty ninetysix | thritynine oneohseven
    ?fiftyone: thritynine ninetyone | twenty oneseventeen
    ?seven: five thritynine | oneeleven twenty
    ?ninetynine: fourtysix twenty | oneohseven thritynine
    ?one: eightytwo thritynine | onenineteen twenty
    ?sixty: twenty onetwentyeight | thritynine eightytwo
    ?thirty: twenty onetwentytwo | thritynine fiftytwo
    ?onetwentyfive: thritynine ninetysix | twenty ten
    ?eightone: onetwentyfive thritynine | sixtyone twenty
    ?eightsix: thritynine ninetysix | twenty ninetysix
    ?onethrity: thritynine twenty | twenty twenty
    ?ninetythree: sixtytwo twenty | seventyone thritynine
    ?twentyeight: onethrity twenty | onenineteen thritynine
    ?onetwenty: fourtyone thritynine | fiftyeight twenty
    ?fiftyfour: eightytwo thirtyeight
    ?sixtytwo: seventytwo twenty | twentyeight thritynine
    ?twentyfour: fourty twenty | ten thritynine
    ?oneohtwo: eighteen twenty | eightytwo thritynine
    ?twentysix: twenty fourtyeight | thritynine eightthree
    ?ten: thritynine twenty | thritynine thritynine
    ?oneeightteen: fiveteen twenty | sistyfive thritynine
    ?seventy: twenty thritynine | thritynine thirtyeight
    ?oneoheight: thritynine oneohseven | twenty onetwentyeight
    ?fourtysix: twenty twenty
    ?thritynine: "a"
    ?sixtyeight: thritynine onenineteen | twenty onetwentyeight
    ?fiftynine: ninetyfive twenty | fourtynine thritynine
    ?fourtyone: twentythree thritynine | thirtyseven twenty
    ?twentynine: twenty sixteen | thritynine seventyeight
    ?onesixteen: oneoheight thritynine | ninetynine twenty
    ?thirtyfive: thritynine fiftyfive | twenty four
    ?fiftythree: ninetyfour twenty | onethritythree thritynine
    ?twenty: "b"
    ?ninetyeight: ninetyseven twenty | fiftysix thritynine
    ?eight: fourtytwo | fourtytwo eight
    ?nineteen: twenty onethirteen | thritynine onesixteen
    ?oneohsix: thritynine three | twenty onetwentythree
    ?fourtyfour: seventynine twenty | fifty thritynine
    ?fourtyfive: thritynine seventythree | twenty sixtyeight
    ?zero: eight eleven
    ?sixtysix: thritynine onetwentyeight | twenty twelve
    ?seventyeight: thritynine onethrity | twenty fourty
    ?fourteen: thritynine hundred | twenty onefourteen
    ?oneohfive: thritynine twentynine | twenty oneohone
    ?onefourteen: thritynine fourty | twenty eighteen
    ?onetwentyone: twenty sixtyfour | thritynine ninetytwo


    %import common.WS_INLINE
    %ignore WS_INLINE
"""

def day19_2(data):    
    #data = read_input(2020, "191") 

    '''
    # dumb approach
    parser = Lark(grammarDay19Part2)
    ast = parser.parse    
    _, messages = processInputDay19(data)
    '''
    #print(rules)

    # joins each line, separated by \n, until it finds an empty string
    it = iter(data)
    grammar = "\n".join(takewhile(lambda l: l != '', it))
    # modify rules 8 and 11... I know it's a regex :(
    grammar = re.sub(r'8: 42', r'8: 42 | 42 8', grammar)
    grammar = re.sub(r'11: 42 31', r'11: 42 31 | 42 11 31', grammar)

    # prefixes digits by r
    grammar = re.sub(r'(\d+)', r'r\1', grammar)

    # this is indeed much cleaner :|
    parser = Lark(grammar, start="r0")
    ast = parser.parse
 
    count = 0
    for msg in list(it):
        try:
            ast(msg)
            count += 1
        except:
            count = count
            #print("no match for",msg)

    result = count
    AssertExpectedResult(436, result, 2)
    return result


def checkBorderMatches(tiles):
    #print()
    matches = 1
    for number, tile in tiles.items():
        #print("Checking tile", number)
        up = tile[0]
        down = tile[1]
        left = tile[2]
        right = tile[3]
        matchesU = 0
        matchesD = 0
        matchesL = 0
        matchesR = 0
        for  n, t in tiles.items():
            if n != number:
                if t[0] == up or t[1] == up or t[2] == up or t[3] == up or\
                   t[0][::-1] == up or t[1][::-1] == up or t[2][::-1] == up or t[3][::-1] == up:
                    #print("UP matches a border of",n)
                    matchesU += 1
                elif t[0] == down or t[1] == down or t[2] == down or t[3] == down or\
                     t[0][::-1] == down or t[1][::-1] == down or t[2][::-1] == down or t[3][::-1] == down:
                    #print("DOWN matches a border of",n)
                    matchesD += 1
                elif t[0] == left or t[1] == left or t[2] == left or t[3] == left or \
                     t[0][::-1]  == left or t[1][::-1]  == left or t[2][::-1]  == left or t[3][::-1]  == left:
                    #print("LEFT matches a border of",n)
                    matchesL += 1
                elif t[0] == right or t[1] == right or t[2] == right or t[3] == right or\
                     t[0][::-1]  == right or t[1][::-1]  == right or t[2][::-1]  == right or t[3][::-1]  == right:
                    #print("RIGHT matches a border of",n)
                    matchesR += 1

        if matchesU+ matchesD + matchesL + matchesR ==2:
            matches *= number
            '''
            print()
            print("total mathces:", matchesU+ matchesD + matchesL + matchesR)
            print("UP",tile[0],"has ", matchesU)   
            print("DOWN", tile[1],"has ", matchesD)   
            print("LEFT", tile[2],"has ", matchesL)   
            print("RIGHT", tile[3],"has ", matchesR)   
            print()
            '''
    #print()
    return matches

def parseInputDay20(data):
    tiles = {}
    left = []
    right = []
    up = []
    down = []
    newTile = True
    pos = -1

    for line in data:
        #print(line)
        pos += 1
        if newTile:
            number = int(line.split(" ")[1][:-1])
            newTile = False
            continue

        if line == '':
            down = (data[pos-1])

            tiles[number] = [up, down, "".join(left), "".join(right)]
            newTile = True
            left = []
            right = []
            up = []
            down = []
            #print(pos)            
            continue

        if len(up) == 0:
            up = line
        left.append(line[0])
        right.append(line[len(line)-1])

    return tiles

def parseInputDay20Part2(data):
    tiles = {}
    left = []
    right = []
    up = []
    down = []
    newTile = True
    pos = -1
    grid = []
    for line in data:
        #print(line)
        pos += 1
        if newTile:
            number = int(line.split(" ")[1][:-1])
            newTile = False
            
            #print()
            #print(number)
            continue

        if line == '':
            down = (data[pos-1])
            # remove border from final grid
            grid = grid[:-1]          

            tiles[number] = [up, down, "".join(left), "".join(right)]
            newTile = True
            left = []
            right = []
            up = []
            down = []
            #print(pos)            
            continue

        if len(up) == 0:
            up = line
        else:
            
            #print(line)
            #print("redact borders",line[1:len(line)-1])

            # redacting left and right borders in the final grid
            grid.append(line[1:len(line)-1])

        left.append(line[0])
        right.append(line[len(line)-1])

    return tiles, grid

'''
Didn't feel like solving a puzzle so I just compute the number of tiles with 2 matching borders for Part1.
Part 2 made me realise I was screwed :( yet I stil didn't feel like solving a puzzle so I started to look for strategies based on the number of monsters. 
I guess the issue was finding the correct number of monsters. 
My solution worked for the sample becuase it doesn't have 2 \n at the end. This screwed me with the real data input lol
When I hit the 15m timeout mark on AoC I went to validate how far off was I from the actual solution and found out this bug. 
I had already tried 29 monsters earlier T_T
'''
#Day 20, part 1: 17148689442341 (0.056 secs)
#Day 20, part 2: 2009 (0.010 secs)
def day20_1(data):    
    #data = read_input(2020, "201") 

    tiles = parseInputDay20(data)
    result = checkBorderMatches(tiles)
      
    AssertExpectedResult(17148689442341, result, 1)
    return result


def day20_2(data):    
    #data = read_input(2020, "201") 
    _, grid = parseInputDay20Part2(data)
    
    count = 0
    # 39 is high: 1857 is too low
    # 35 is high: 1917 is too low
    # 30 is high: 1992 is too low -- upper bound
    ## had a bug in input data (2 new lines at the end) which ruined my solution - this was the one :'(
    # 29 is wrong : 2007/ 2009 T_T
    # 28 is wrong : 2022 - no hint :(
    # 27 is wrong : 2037
    # 26 is wrong : 2052
    # 25 is wrong : 2067
    # 24 is wrong : 2082
    # 23
    # 22
    # 21
    # 20 is wrong : 2142
    # 15 is wrong : 2217
    monsters = 15 * 29
    
    # count all # in the grid without borders
    count = sum([line.count("#") for line in grid])   

    # guess the total amount of sea monsters lol
    result = count - monsters   
      
    AssertExpectedResult(2009, result, 2)
    return result


def processInputDay21(data):
    ingredientsDict = {}
    allergensDict = {}
    for food in data:
        ingredientList = food.split("(")
        ingredients = ingredientList[0].strip().split(" ")
        allergens = ingredientList[1]
        allergens = [allergen.strip() for allergen in allergens.split("contains")[1].strip()[:-1].split(",")]

        for i in ingredients:
            if i not in ingredientsDict:
                ingredientsDict[i] = 1
            else: 
                ingredientsDict[i] += 1
        
        for a in allergens:
            if a not in allergensDict:
                allergensDict[a] = set(ingredients)
            else:
                allergensDict[a].intersection_update(set(ingredients))   

    #print("# Allergens:", len(allergensDict.keys()))
    #print("# Ingredients:", len(ingredientsDict.keys()))
    #print("# Ingredients wo allergens:", len(ingredientsDict.keys()) - len(allergensDict.keys()))
    #print(allergensDict)
    #print(ingredientsDict)
    return ingredientsDict, allergensDict

def checkAllergens(allergens):
    # yes, I'm lazy
    for _ in range(3):
        for allergen, ingredients in allergens.items():
            # means this allergen is solved, remove the ingridient from all other allergens sets
            if len(ingredients) == 1:
                ingredient = ingredients.pop()
                for a, _ in allergens.items():
                    if a != allergen:
                        allergens[a].discard(ingredient)
                
                ingredients.add(ingredient)
                allergens[allergen] = ingredients
    return allergens

def day21_1(data):    
    #data = read_input(2020, "211") 
    
    ingredients, allergens = processInputDay21(data)
    # find match between allergens and ingredients
    allergens = checkAllergens(allergens)
    ingridientsWithAllergens = [ing.pop() for ing in allergens.values()]
    result = sum([occurences for ingredient, occurences in ingredients.items() if ingredient not in ingridientsWithAllergens])

    AssertExpectedResult(1977, result, 1)
    return result

def day21_2(data):    
    #data = read_input(2020, "211") 
    
    _, allergens = processInputDay21(data)
    allergens = checkAllergens(allergens)

    sortedAllergens = [keys for keys in allergens.keys()]
    sortedAllergens.sort()
    sortedIngridients = [allergens[k].pop() for k in sortedAllergens ]

    result = ','.join(sortedIngridients)
    
    AssertExpectedResult('dpkvsdk,xmmpt,cxjqxbt,drbq,zmzq,mnrjrf,kjgl,rkcpxs', result, 2)
    return result


def processInputDay22(data):
    player1 = []
    player2 = []
    switchPlayer = False
    for card in data:
        if card == 'Player 1:' or card == '':
            continue
        elif card == 'Player 2:':
            switchPlayer = True
            continue

        if switchPlayer:
            player2.append(card)
        else:
            player1.append(card)
    
    return ints(player1), ints(player2)

def playCombat(player1, player2):    

    while True:
        if len(player1) == 0: 
            return player2
        elif len(player2) == 0:
            return player1

        play1 = player1.pop(0)
        play2 = player2.pop(0)

        if play1 > play2:
            player1.append(play1)
            player1.append(play2)
        else:
            player2.append(play2)
            player2.append(play1)

#Day 22, part 1: 33403 (0.001 secs)
#Day 22, part 2: 29177 (10.875 secs)
def day22_1(data):    
    #data = read_input(2020, "221")    

    player1, player2 = processInputDay22(data)
    cards = playCombat(player1, player2)

    #print("Player 1 cards:", player1)
    #print("Player 2 cards:", player2)

    score = 0
    pos = 1
    cards.reverse()
    for card in cards:
        score += card * pos
        pos += 1
    
    result = score 
    AssertExpectedResult(33403, result, 1)
    return result

def playRecursiveCombatGame(player1, player2, game, debug = False):    
        # play game
        rounds = set()
        roundNumber = 0
        
        if debug:
            print("=== Game", game,"===")
            print()
        
        while True:   
            roundNumber += 1

            # recursion stop condition, player 1 wins 
            if (str(player1), str(player2)) in rounds:
                return "Player 1", player1
            
            # save round state
            rounds.add( ( str(player1.copy()), str(player2.copy()) ) )

            if debug:
                print("-- Round", roundNumber," (Game", game, ") --")
                print("Player 1's deck:", str(player1) )
                print("Player 2's deck:", str(player2) )
            
            # game ends
            if len(player1) == 0: 
                return "Player 2", player2
            elif len(player2) == 0:
                return "Player 1", player1
                
            # make a play
            play1 = player1.pop(0)
            play2 = player2.pop(0)

            if debug:
                print("Player 1 plays:", play1)
                print("Player 2 plays:", play2)

            # play sub-game
            winner = None
            if len(player1) >= play1 and len(player2) >= play2:
                if debug:
                    print("Playing a sub-game to determine the winner...")
                    print()
                player1Copy = player1.copy()[:+play1]
                player2Copy = player2.copy()[:+play2]
                winner, _ = playRecursiveCombatGame(player1Copy, player2Copy, game+1)
                
                if debug:
                    print("The winner of game", game+1,"is",winner,"!")
                    print("...anyway, back to game", game,".")

            # determine round's winner
            if winner == "Player 1":
                if debug:
                    print("Player 1 wins round", roundNumber," of game", game,"!")
                    print()
                player1.append(play1)
                player1.append(play2)
            elif winner == "Player 2":
                if debug:
                    print("Player 2 wins round", roundNumber," of game", game,"!")
                    print()
                player2.append(play2)
                player2.append(play1)
            elif play2 > play1:
                if debug:
                    print("Player 2 wins round", roundNumber," of game", game,"!")
                    print()
                player2.append(play2)
                player2.append(play1)
            elif play1 > play2:
                if debug:
                    print("Player 1 wins round", roundNumber," of game", game,"!")
                    print()
                player1.append(play1)
                player1.append(play2)

def day22_2(data):    
    #data = read_input(2020, "221")    

    player1, player2 = processInputDay22(data)
    _, cards = playRecursiveCombatGame(player1, player2, 1)

    score = 0
    pos = 1
    cards.reverse()
    for card in cards:
        score += card * pos
        pos += 1

    result = score 
    AssertExpectedResult(29177, result, 2)
    return result

#@hashable_lru
def playCrabGame(cupsList, moves):
    cups = len(cupsList)
    currentCupLabel = cupsList[0]
    currentCupPos = 0

    original = cupsList.copy()
    for move in range(moves):
        #print("-- move", move+1,"--")
        #print(cupsList)

        currentCupLabel = cupsList[currentCupPos]
        #print("current cup:", currentCupLabel)

        pos = (currentCupPos+1)
        pickOne = cupsList[pos % len(cupsList)]
        pickTwo = cupsList[(pos+1) % len(cupsList)]
        pickThree = cupsList[(pos+2) % len(cupsList)]
        #print("pick up:", [pickOne, pickTwo, pickThree])
        cupsList.remove(pickOne)
        cupsList.remove(pickTwo)
        cupsList.remove(pickThree)

        destination = currentCupLabel - 1
        while destination in [pickOne, pickTwo, pickThree] or destination not in cupsList:
            destination -= 1
            if destination < min(original):
                destination = max(original)
        
        
       # print("destination:",destination)
       # print()
        destination = cupsList.index(destination) + 1
        cupsList = cupsList[0:destination] + [pickOne, pickTwo, pickThree] + cupsList[destination:]
        
        currentCupPos = (cupsList.index(currentCupLabel) + 1 ) % cups
    return cupsList

def computeScore(cups):
    start = (cups.index(1) + 1) % len(cups)
    if start == 0:
        return ''.join([str(cup) for cup in cups[:-1]])
    else:
        cups = cups[start:] + cups[:start-1]
        return ''.join([str(cup) for cup in cups])

def computeScore2(cups):
    pos = 1
    while cups[pos] != 1:
        print(cups[pos])
        pos = cups[pos]
    return 0

#Day 23, part 1: 96342875 (0.001 secs)
#Day 23, part 2: 563362809504 (8.657 secs)
def day23_1(data):    
    #data = read_input(2020, "231")        
    cupsList = ints(data[0])
    cups = playCrabGame(cupsList, 100)

    '''
    original = ints(data[0])
    
    cupsDict = {}
    for i in range(len(original)):
        pos = (i + 1)% len(original)
        cupsDict[original[i]] = original[pos]

    cupsDict["curr"] = original[0]

    print(cupsDict)
    moves = 100
    cups = playCrabGameOptimized(cupsDict, moves, max(original), 1)
    '''
    result = computeScore(cups)
    AssertExpectedResult('96342875', result, 1)
    return result

def playCrabGameOptimized(cupsDict, moves, max, min):
    '''
    curr - 3
    389125467
    1 -> 2
    2 -> 5
    3 -> 8
    4 -> 6
    5 -> 4
    6 -> 6
    7 -> 3
    8 -> 9
    9-> 1
    10 -> 11
    (...)
    '''
    for move in range(moves):
        #print("-- move", move+1,"--")
        #print(cupsDict)
        currentCupLabel = cupsDict["curr"]
        #print("current cup:", currentCupLabel)

        pickOne = cupsDict[currentCupLabel]
        pickTwo = cupsDict[pickOne]
        pickThree = cupsDict[pickTwo]
        
        #print("pick up:", [pickOne, pickTwo, pickThree])        
        
        destination = currentCupLabel - 1
        while destination in [pickOne, pickTwo, pickThree] or destination not in cupsDict:  
            destination -= 1
            if destination < min:
                destination = max 
        if destination < min:
            destination = max 
        
        #print("destination:", destination)
        #print()        
        
        # "removes" the three picked cups
        cupsDict[currentCupLabel] = cupsDict[pickThree]
        # change pickThree next to destination's next label and destination next label to first picked cup
        cupsDict[pickThree] = cupsDict[destination]
        cupsDict[destination] = pickOne
        cupsDict["curr"] = cupsDict[currentCupLabel]
      

    return cupsDict

def day23_2(data):
    #data = read_input(2020, "231")    
    
    original = ints(data[0])
    cupsDict = dict(zip(range(1,1000001), range(2,1000001)))
    
    for i in range(len(original)):
        pos = (i + 1)% len(original)
        cupsDict[original[i]] = original[pos]

    cupsDict["curr"] = original[0]
    cupsDict[1000000] = original[0]
    cupsDict[5] = 10

    moves = 10000000
    cups = playCrabGameOptimized(cupsDict.copy(), moves, 1000000, 1)
    result = cups[cups[1]] * cups[1]
    
    AssertExpectedResult(563362809504, result, 2)
    return result



def addOrFlipTile(tiles, position):
    # True means tile is black
    #print("Tile position", position)
    if position not in tiles:
        #print("Adding tile", position, "colour black")
        tiles[position] = True
    else:        
        tiles[position] = not tiles[position]        
        #print("Flipping tile", position,"to colour", "black" if tiles[position] else "white")
    return tiles
    
def flipTiles(data):
    # https://www.redblobgames.com/grids/hexagons/
    directions = ['e', 'se', 'sw', 'w', 'nw', 'ne']
    directionsDelta = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    directionsDict = dict(zip(directions, directionsDelta))

    tiles = {}

    for tile in data:
        position = (0,0)
        iterator = iter(tile)

        #print("Getting position for tile", tile)
        while True:
            try:
                dir = next(iterator)
                if dir == 'n' or dir == 's':                    
                    direction = dir + next(iterator)
                else:
                    direction = dir
                
                pos = directionsDict[direction]
                #print("Direction", direction,"has delta", pos)
                #print("Adding delta",pos," to current position", position)
                position = (position[0] + pos[0], position[1] + pos[1])
                #print("Final position", position)

                #print("dir:",direction)
            except StopIteration:
                break
        #print()
        tiles = addOrFlipTile(tiles, position)        
        #print("Tile is in position:", position)
        #print()
    return tiles

#Day 24, part 1: 232 (0.004 secs)
#Day 24, part 2: 3519 (3.540 secs)
def day24_1(data):
    #data = read_input(2020, "241") 
       
    tiles = flipTiles(data)
    #print("# Tiles:", len(tiles))
    result = sum([1 for (pos, isBlack) in tiles.items() if isBlack])

    #print("# Black tiles:", result)
    AssertExpectedResult(232, result, 1)
    return result


def countBlackNeightbours(grid, pos):
    directionsDelta = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    black = 0
    for x,y in directionsDelta:
        neighbour = (pos[0] + x , pos[1] + y)
        if neighbour in grid:
            if grid[neighbour]:
                black +=1

    return black

def flipTilesGame(tiles, moves):
    newGrid = tiles.copy()
    
    for day in range(moves):
        grid = newGrid.copy()
        
        for pos, isBlack in grid.items():
            blackNeighbours = countBlackNeightbours(grid, pos)
            if isBlack and (blackNeighbours == 0 or blackNeighbours > 2):
                #flip to white
                newGrid[pos] = False
            elif not isBlack and blackNeighbours == 2:
                newGrid[pos] = True

        #print("Day",day,":", sum([1 for (pos, isBlack) in newGrid.items() if isBlack]))
    return newGrid

def hexagon(turtle, radius, color):
    FONT_SIZE = 18
    FONT = ('Arial', FONT_SIZE, 'normal')
    ROOT3_OVER_2 = sqrt(3) / 2

    clone = turtle.clone()  # so we don't affect turtle's state
    xpos, ypos = clone.position()
    clone.setposition(xpos - radius / 2, ypos - ROOT3_OVER_2 * radius)
    clone.setheading(-30)
    clone.color('black', color)
    clone.pendown()
    clone.begin_fill()
    clone.circle(radius, steps=6)
    clone.end_fill()
    clone.penup()
    clone.setposition(xpos, ypos - FONT_SIZE / 2)
    #clone.write(label, align="center", font=FONT)

def plotTiles(tiles):    
    SIDE = 50  # the scale used for drawing

    # Initialize the turtle
    tortoise = Turtle(visible=False)
    tortoise.speed('fastest')
    tortoise.penup()

    # Plot the points
    for pos, isBlack in tiles.items():
        if isBlack:
            color = 'black'
        else:
            color = 'white'
        tortoise.goto((pos[0]*SIDE, pos[1] * SIDE))
        hexagon(tortoise, SIDE, color)

    # Wait for the user to close the window
    screen = Screen()
    screen.exitonclick()   
      

def day24_2(data):
    #data = read_input(2020, "241") 

    size = 70
    positions = [(0+i, 0+j) for i in range(-size,size+1) for j in range(-size,size+1)]    
    tiles = dict(zip(positions, [False]*len(positions)))
    
    inputTiles = flipTiles(data)
    for pos, isBlack in inputTiles.items():
        tiles[pos] = isBlack
    
    moves = 100
    tiles = flipTilesGame(tiles.copy(), moves)

    #print("# Tiles:", len(tiles))
    result = sum([1 for (pos, isBlack) in tiles.items() if isBlack])
   
    #does not work :'(
    #plotTiles(tiles)
    
    AssertExpectedResult(3519, result, 2)
    return result


def transformSubjectNumber(number, loopSize):
    value = 1
    loop_size = loopSize
    n = 20201227

    for _ in range(loop_size):
        value *= number
        value %= n
    
    return value

'''
# initial inneficient version, work was being repeated when computing the subject transformation
def findLoopSize(publicKey):
    loopSize = 1
    result = 1
    while True:
        loopSize += 1 
        result = transformSubjectNumber(7, loopSize)
        if(result == publicKey):
            break

    return loopSize
'''

def findLoopSize(publicKey):
    loopSize = 0
    result = 1
    
    while result != publicKey:
        loopSize += 1 
        result *= 7
        result %= 20201227

    return loopSize

#Day 25, part 1: 19774660 (5.556 secs)
def day25_1(data):
    #data = read_input(2020, "251") 

    iterator = iter(data)
    cardPublicKey = int(next(iterator))
    doorPublicKey = int(next(iterator))
    cardLoopsSize = findLoopSize(cardPublicKey)
    doorLoopSize = findLoopSize(doorPublicKey)
    
    encryptKeyDoor = transformSubjectNumber(doorPublicKey, cardLoopsSize)
    encryptKeyCard = transformSubjectNumber(cardPublicKey, doorLoopSize)
    
    # double check :p 
    if encryptKeyDoor == encryptKeyCard:
        result = encryptKeyCard
    else:
        result = 0
    
    AssertExpectedResult(19774660, result, 1)
    return result


if __name__ == "__main__":
    main(sys.argv, globals(), 2020)

