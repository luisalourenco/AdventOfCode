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
from math import sqrt
import hashlib
from itertools import groupby
import codecs
from tsp_solver.greedy import solve_tsp
from tsp_solver.util import path_cost
#from python_tsp.exact import solve_tsp_dynamic_programming



FILE_DIR = os.path.dirname(os.path.realpath(__file__))
print(FILE_DIR)
sys.path.insert(0, FILE_DIR + "/")
sys.path.insert(0, FILE_DIR + "/../")
sys.path.insert(0, FILE_DIR + "/../../")

from common.test import solve_tsp2
from common.utils import read_input, main, clear, AssertExpectedResult, ints  # NOQA: E402
from common.mapUtils import printMap, buildMapGrid, buildGraphFromMap
from common.graphUtils import printGraph, find_all_paths, find_path, find_shortest_path, find_shortest_pathOptimal, bfs, dfs, Graph, hashable_lru
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

'''
Day 1 - Not Quite Lisp
'''

def followDirections(directions, findBasement = False):
    floor = 0
    basement = 1
    for direction in directions:
        if direction == '(':
            floor += 1
        elif direction == ')':
            floor -= 1
        
        if findBasement:
            if floor == -1:
                return floor, basement
        basement += 1
    
    return floor, basement

#Day 1, part 1: 280 (0.003 secs)
#Day 1, part 2: 1797 (0.001 secs)
def day1_1(data):
    #data = read_input(2015, "251") 
    result, _ = followDirections(data[0])
    AssertExpectedResult(280, result, 1)
    return result

def day1_2(data):
    #data = read_input(2015, "251") 
    _, result = followDirections(data[0], True)
    AssertExpectedResult(1797, result, 2)
    return result


'''
Day 2 - I Was Told There Would Be No Math
'''

def processInputDay2(data):
    boxes = []
    for box in data:
        dimensions = box.split("x")
        boxes.append( (int(dimensions[0]), int(dimensions[1]), int(dimensions[2])) )
    
    return boxes

def computeBoxSurfaceArea(l, w, h):
    return 2*l*w + 2*w*h + 2*h*l

def computeSmallestSideArea(l, w, h):
    sides = [l,w,h]
    sideA = min(sides)
    sides.remove(sideA)
    sideB = min(sides)
    return sideA * sideB

def determineWrappingPaperOrder(boxes):
    total = 0
    for box in boxes: 
        total += computeBoxSurfaceArea(box[0], box[1], box[2]) + computeSmallestSideArea(box[0], box[1], box[2])
    return total

#Day 2, part 1: 1598415 (0.004 secs)
#Day 2, part 2: 3812909 (0.004 secs)
def day2_1(data):
    #data = read_input(2015, "201") 
    boxes = processInputDay2(data)
    result = determineWrappingPaperOrder(boxes)
    AssertExpectedResult(1598415, result, 1)
    return result

def computeSmallestSidePerimeter(l, w, h):
    sides = [l,w,h]
    sideA = min(sides)
    sides.remove(sideA)
    sideB = min(sides)
    return 2 * sideA + 2 *sideB

def computeBoxVolume(l, w, h):
    return l * w * h

def determineRibbonOrder(boxes):
    total = 0
    for box in boxes: 
        total += computeBoxVolume(box[0], box[1], box[2]) + computeSmallestSidePerimeter(box[0], box[1], box[2])
    return total

def day2_2(data):
    #data = read_input(2015, "201") 
    boxes = processInputDay2(data)
    result = determineRibbonOrder(boxes)
    AssertExpectedResult(3812909, result, 1)
    return result

 
'''
Day 3 - Perfectly Spherical Houses in a Vacuum
'''

def findHowManyUniqueDeliveries(directions):
    position = (0, 0)
    uniqueHouses = {position}
    directionsDelta = { "<": (-1, 0), ">": (1,0), "v": (0, -1), "^": (0,1)}

    for direction in directions:
        position = tuple(map(operator.add, position, directionsDelta[direction]))
        uniqueHouses.add(position)

    return len(uniqueHouses)


def day3_1(data):
    #data = read_input(2015, "301") 

    result = findHowManyUniqueDeliveries(data[0])
    AssertExpectedResult(2565, result, 1)
    return result

def findHowManyUniqueDeliveriesWithRoboSanta(directions):
    positionSanta = (0, 0)
    positionRoboSanta = (0, 0)
    uniqueHouses = {positionSanta}
    directionsDelta = { "<": (-1, 0), ">": (1,0), "v": (0, -1), "^": (0,1)}

    roboSantaTurn = False
    for direction in directions:
        if roboSantaTurn:
            positionRoboSanta = tuple(map(operator.add, positionRoboSanta, directionsDelta[direction]))
            uniqueHouses.add(positionRoboSanta)
        else:
            positionSanta = tuple(map(operator.add, positionSanta, directionsDelta[direction]))
            uniqueHouses.add(positionSanta)
        roboSantaTurn = not roboSantaTurn

    return len(uniqueHouses)

def day3_2(data):
    #data = read_input(2015, "301") 

    result = findHowManyUniqueDeliveriesWithRoboSanta(data[0])
    AssertExpectedResult(2639, result, 2)
    return result


'''
Day 4 - The Ideal Stocking Stuffer
'''

def findLowestPositiveNumber(secretKey, part = 1):
    if part == 2:
        number = 1000000
    else:
        number = 1
    while True:
        numberStr = str(number)
        sample = secretKey + numberStr
        md5Sum = hashlib.md5(sample.encode('utf-8')).hexdigest()
        if part == 1:
            if md5Sum.startswith('00000'):
                return number
        elif part == 2:
            if md5Sum.startswith('000000'):
                return number
        number+= 1
    return None

def day4_1(data):
    #data = read_input(2015, "301") 
    result = findLowestPositiveNumber(data[0])
    AssertExpectedResult(254575, result, 1)
    return result

def day4_2(data):
    #data = read_input(2015, "301") 
    result = findLowestPositiveNumber(data[0], 2)
    #result = findLowestPositiveNumber('pqrstuv', 2)
    AssertExpectedResult(1038736, result, 2)
    return result


'''
Day 5 - Doesn't He Have Intern-Elves For This?
'''

def isNiceString(text):
    badWords = ['ab', 'cd', 'pq', 'xy']
    vowels = ['a', 'e', 'i', 'o', 'u']

    if sum([1 for badWord in badWords if badWord in text]) != 0:
        return False
    if sum([text.count(vowel) for vowel in vowels if vowel in text]) < 3:
        return False
    
    groups = groupby(text)    
    
    if sum([1 for label, group in groups if sum(1 for _ in group) >= 2]) == 0:
        return False

    return True

def countNiceStrings(textFile, part = 1):
    if part == 1:
        return sum([1 for text in textFile if isNiceString(text)])
    elif part == 2:
        return sum([1 for text in textFile if isNiceStringNewRules(text)])

def day5_1(data):
    #data = read_input(2015, "301") 
    result = countNiceStrings(data)
    AssertExpectedResult(255, result, 1)
    return result

# yes, I'm starting to use regexs :(
def isNiceStringNewRules(text):    
    if len(re.findall(r"(\w{2}).*?(\1)", text, re.IGNORECASE)) != 1:
        return False
    
    if len(re.findall(r"(\w{1})\w{1}(\1)", text, re.IGNORECASE)) == 0:
        return False    

    return True


def day5_2(data):
    #data = ['qjhvhtzxzqqjkmpb', 'xxyxx' , 'uurcxstgmygtbstg', 'ieodomkazucvgmuy']
    result = countNiceStrings(data, 2)
    AssertExpectedResult(55, result, 2)
    return result


'''
Day 6 - Probably a Fire Hazard
'''

def performInstruction(grid, command, rangeInstructions):
    # (lower, upper), (lower, upper)
    rows, columns = rangeInstructions
    
    #print(rows)
    #print(columns)
    for y in range(rows[0], rows[1]+1):
        for x in range(columns[0], columns[1]+1):
            if command == 'turn on':
                grid[y][x] = 1
            elif command == 'turn off':
                grid[y][x] = 0
            elif command == 'toggle':
                grid[y][x] = not int(grid[y][x])
    return grid

def executeInstructions(grid, instructions, part = 1):
    #toggle 461,550 through 564,900
    #turn off 370,39 through 425,839
    for instruction in instructions:
        split = instruction.split(' ')
        
        if split[0] == 'toggle':
            command = 'toggle'
            rangeLower = split[1].split(',')
            rangeUpper = split[3].split(',')
        else:
            command = split[0] + ' ' + split[1]
            rangeLower = split[2].split(',')
            rangeUpper = split[4].split(',')
        
        # (x,y) throgh (x', y') => (x,x') , (y, y')
        rowsRange = (int(rangeLower[0]), int(rangeUpper[0]))
        columnsRange = (int(rangeLower[1]), int(rangeUpper[1]))

        if part == 1:
            grid = performInstruction(grid, command, ( rowsRange , columnsRange) )  
        elif part == 2:
            grid = performInstructionCorrect(grid, command, ( rowsRange , columnsRange) )  

    return grid


def day6_1(data):
    #data = read_input(2015, "601") 

    rows, columns = 1000, 1000   
    grid = [ [ 0 for i in range(columns) ] for j in range(rows) ]    

    grid = executeInstructions(grid, data)

    result = sum( [ grid[j][i] for i in range(columns) for j in range(rows) ] ) 

    AssertExpectedResult(543903, result, 1)
    return result


def performInstructionCorrect(grid, command, rangeInstructions):
    # (lower, upper), (lower, upper)
    rows, columns = rangeInstructions
    
    for y in range(rows[0], rows[1]+1):
        for x in range(columns[0], columns[1]+1):
            if command == 'turn on':
                grid[y][x] += 1
            elif command == 'turn off':
                grid[y][x] -= 1
                if grid[y][x] < 0:
                    grid[y][x] = 0
            elif command == 'toggle':
                grid[y][x] += 2
    return grid


def day6_2(data):
    #data = read_input(2015, "601") 

    rows, columns = 1000, 1000   
    grid = [ [ False for i in range(columns) ] for j in range(rows) ]    

    grid = executeInstructions(grid, data, 2)
    result = sum( [ grid[j][i] for i in range(columns) for j in range(rows) ] ) 

    AssertExpectedResult(14687245, result, 2)
    return result


'''
Day 7 - Some Assembly Required
'''
def processInstructionsBooklet(data):

    wires = {}
    commands = []
    for instruction in data:
        pieces = instruction.split('->')
        wire = pieces[1].strip()

        operation = pieces[0].strip().split(' ')
        size = len(operation)
        if size == 1:
            try:
                wires[wire] = int(operation[0])
            except ValueError:
                wires[wire] = operation[0]
        elif size == 2:
            command = operation[0].strip()
            args = [operation[1].strip()]
            commands.append((command, args, wire))
        elif size == 3:
            command = operation[1].strip()
            args = [operation[0].strip(), operation[2].strip()]
            commands.append((command, args, wire))
        else:
            raise ValueError

    #print(commands)
    #print(wires)
    
    return wires, commands

def emulateCircuit(wires, circuit):

            pending = []
            while len(circuit) > 0:
                operation, args, wire = circuit.pop()
                #print(operation, args, wire)
                try:
                    if len(args) == 2:
                        left = args[0]
                        left = int(left) if left.isnumeric() else wires[left]
                        right = args[1]
                        right = int(right) if right.isnumeric() else wires[right]
                    elif len(args) == 1:
                        arg = args[0]
                        arg = int(arg) if arg.isnumeric() else wires[arg]
                except KeyError:
                    circuit.insert(0, (operation, args, wire))
                    continue
                
                
                if operation == 'AND':
                    #print(wire, "<-", left,operation,right,"[",args[0],",",args[1],"]")
                    wires[wire] = left & right
                elif operation == 'OR':
                    wires[wire] = left | right
                    #print(wire, "<-", left,operation,right,"[",args[0],",",args[1],"]")
                elif operation == 'LSHIFT':
                    wires[wire] = left << right
                    #print(wire, "<-", left,operation,right,"[",args[0],",",args[1],"]")
                elif operation == 'RSHIFT':
                    wires[wire] = left >> right
                    #print(wire, "<-", left,operation,right,"[",args[0],",",args[1],"]")
                elif operation == 'NOT':
                    wires[wire] = abs(~arg)
                    #print(wire, "<-", operation,arg,"[",args[0],"]")
                else:
                    #circuit.append((operation, args, wire))
                    raise ValueError           
                    
            #print("pending:",pending)
            return wires

def day7_1(data):
    #data = read_input(2015, "701")

    wires, circuit = processInstructionsBooklet(data)
    #print(wires)
    wires = emulateCircuit(wires, circuit)
    #print(wires)
    result = wires[wires['a']]
    AssertExpectedResult(956, result, 1)
    return result

def day7_2(data):
    #data = read_input(2015, "701")

    wires, circuit = processInstructionsBooklet(data)
    wires['b'] = 956
    wires = emulateCircuit(wires, circuit)
    result = wires[wires['a']]
    AssertExpectedResult(40149, result, 2)
    return result


'''
Day 8 - Matchsticks
'''

def rawString(string):
    return '%r'%string

def processStringsAndComputeResult(data):

    inMemoryCount = 0
    inCodeCount = 0
    encodedCount = 0
    for string in data:
        string = string.strip()
        inCodeCount += len(string)

        ss = codecs.decode(string[1: len(string)-1], 'unicode_escape')
        inMemoryCount += len(ss)

        encoded = 6
        for s in string[1:-1]:
            #print("char:",s)
            if s in ['\\', '"']:
                encoded += 2
            else:
                encoded += 1
        encodedCount += encoded
        #print("encoded :", string, encoded)
        #print("in memory:", ss, len(ss))
        #print("in code:", string, len(string))
        #print()

    return inCodeCount - inMemoryCount, encodedCount - inCodeCount

def day8_1(data):
    #data = read_input(2015, "801")
    result, _ = processStringsAndComputeResult(data)
    AssertExpectedResult(1342, result, 1)
    return result

def day8_2(data):
    #data = read_input(2015, "801")
    _, result = processStringsAndComputeResult(data)
    AssertExpectedResult(2074, result, 2)
    return result


'''
Day 9 - All in a Single Night
'''

def processInputAndBuildGraph(data):   
    
    # Faerun to Norrath = 129
    distances = []
    cities = {}
    pos = 0
    for route in data:
        routes = route.split("to")
        orig = routes[0].strip()
        routes2 = routes[1].split("=")
        dest = routes2[0].strip()
        distance = int(routes2[1].strip())

        if orig not in cities:
            origN = pos
            cities[orig] = pos
            pos += 1
        else:
            origN = cities[orig]
        
        if dest not in cities:
            destN = pos
            cities[dest] = pos
            pos += 1
        else:
            destN = cities[dest]

        # Add route as an edge to the graph
        #graph[origN][destN] = distance
        
        distances.append((origN,destN,distance))
        #graph.add_edge(orig, dest, distance=(distance))
    
    graph = [ [ sys.maxsize for i in range(pos) ] for j in range(pos) ]  
    for orig, dest, distance in distances:
        graph[orig][dest] = distance
        graph[dest][orig] = distance
        graph[orig][orig] = 0
        graph[dest][dest] = 0
    
    return graph, cities


def day9_1(data):
    #data = read_input(2015, "901")
    graph, cities = processInputAndBuildGraph(data)
    
    #print(cities)
    #for ds in graph:
    #    print(ds)
        
    path = solve_tsp(graph)
    print(path)
    result = path_cost(graph, path)
    AssertExpectedResult(207, result, 1)
    return result

def day9_2(data):
    #data = read_input(2015, "901")
    graph, cities = processInputAndBuildGraph(data)    
    cost = 0
    for i in itertools.permutations(list(cities.values())):
        r = path_cost(graph, i)
        if r > cost:
            cost = r
    result = cost
    AssertExpectedResult(804, result, 2)
    return result

if __name__ == "__main__":
    main(sys.argv, globals(), 2015)

