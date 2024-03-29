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
from itertools import combinations, takewhile
from itertools import permutations
from math import sqrt
import hashlib
from itertools import groupby
import codecs
from tsp_solver.greedy import solve_tsp
from tsp_solver.util import path_cost
from collections import namedtuple, Counter




FILE_DIR = os.path.dirname(os.path.realpath(__file__))
print(FILE_DIR)
sys.path.insert(0, FILE_DIR + "/")
sys.path.insert(0, FILE_DIR + "/../")
sys.path.insert(0, FILE_DIR + "/../../")

from common.test import solve_tsp2
from common.utils import read_input, main, clear, AssertExpectedResult, ints  # NOQA: E402
from common.mapUtils import printMap, buildMapGrid, buildGraphFromMap
from common.graphUtils import printGraph, find_all_paths, find_path, find_shortest_path, find_shortest_pathOptimal, bfs, dfs, Graph, hashable_lru
from lark import Lark, Transformer, Visitor, v_args, Token
from lark.visitors import Interpreter, visit_children_decor
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

'''
Day 10 - Elves Look, Elves Say
'''

'''
3113322113
[('3', 1), ('1', 2), ('3', 2), ('2', 2), ('1', 2), ('3', 1)]
'''
def playLookAndSay(number, rounds):  
    groups = groupby(number)
    groupResults = ''.join( [ (str(sum(1 for _ in group)) + label ) for label, group in groups] )

    return groupResults

#Day 10, part 1: 329356 (0.983 secs)
#Day 10, part 2: 4666278 (13.142 secs)
def day10_1(data):
    #data = read_input(2015, "101")
    
    rounds = 40
    result = data[0]
    for _ in range(rounds):
        result = playLookAndSay(result, rounds)
    
    result = len(result)
    AssertExpectedResult(329356, result, 1)
    return result

def day10_2(data):
    #data = read_input(2015, "101")
  
    rounds = 50
    result = data[0]
    for _ in range(rounds):
        result = playLookAndSay(result, rounds)
    
    result = len(result)
    AssertExpectedResult(4666278, result, 2)
    return result


'''
Day 11 - Corporate Policy
'''

def incrementPassword(password):
    newPassword = ''
    stopIncrements = False
    for letter in password[::-1]:
        if stopIncrements:
             newPassword = newPassword + letter
        else:
            # ensures x is between [ord('a'), ord('z')]
            x = ( ord(letter) + 1 ) % ( ord('z')+1 ) 
            if x == 0:                
                x = ord('a')
            else:
                stopIncrements = True  
            newPassword = newPassword + chr(x)

    return newPassword[::-1]

def containsIncreasingStraight(password):
    for i in range(2, 8):
        if ord(password[i]) - ord(password[i-1]) == 1 and ord(password[i-1]) - ord(password[i-2]) == 1:
            return True 
    return False

#abcdefkd
def isPasswordValid(password):
    forbiddenCharacters = ['i', 'o', 'l']

    # rule #1
    if not containsIncreasingStraight(password):
        #print("Rule #1")
        return False

    # rule #2
    if sum([1 for letter in password if letter in forbiddenCharacters]) > 0:
        #print("Rule #3: contains forbidden characters")
        return False

    # rule #3
    uniqueGroups = []  
    for char, group in groupby(password):
        if sum(1 for _ in group) >= 2:
            if char not in uniqueGroups:
                uniqueGroups.append(char)
            else:
                #print("Rule #3: does not contain two DISTINCT pairs of letters")
                return False
    # rule #3
    if len(uniqueGroups) < 2:
        #print("Rule #3: does not contain at least two pairs of letters")
        return False

    return True

def nextPassword(password):
    preventInifniteLoops = 0
    newPassword = password
    while True:
        preventInifniteLoops += 1
        newPassword = incrementPassword(newPassword)
        if isPasswordValid(newPassword):# or preventInifniteLoops == 1000000:
            return newPassword

    return ''

#Day 11, part 1: hepxxyzz (2.015 secs)
#Day 11, part 2: heqaabcc (5.394 secs)
def day11_1(data):
    #data = read_input(2015, "111")
    oldPassword = data[0]
    #newPassword = nextPassword(newPassword)
    
    #print("next:",nextPassword('abcdefgh'))
    #print("next:",nextPassword('ghijklmn'))
    result = nextPassword(oldPassword)
    AssertExpectedResult('hepxxyzz', result, 1)
    return result

def day11_2(data):
    #data = read_input(2015, "111")
    oldPassword = 'hepxxyzz'
 
    result = nextPassword(oldPassword)
    AssertExpectedResult('heqaabcc', result, 2)
    return result


'''
Day 12 - JSAbacusFramework.io
'''

json_grammar = r"""
    start: value

    value:  object
            | list
            | string
            | SIGNED_NUMBER -> number
            | "true" -> true
            | "false" -> false
            | "null" -> null
    
    pvalue:  object
            | plist
            | pstring
            | SIGNED_NUMBER -> pnumber
            | "true" -> true
            | "false" -> false
            | "null" -> null

    list : "[" [value ("," value)*] "]"

    plist : "[" [value ("," value)*] "]"

    object : "{" [pair ("," pair)*] "}"

    pair : string ":" pvalue

    string : ESCAPED_STRING
    pstring.1 : ESCAPED_STRING

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS

    %ignore WS

"""

class TreeToJson(Visitor):
    count = 0

    #def __init__(self):
    #    self.count = 0

    def sum(self):
        return self.count

    @v_args(inline=True)
    def string(self, s):
        return s[1:-1].replace('\\"', '"')
    
    @v_args(inline=True)
    def pstring(self, s):
        return s[1:-1].replace('\\"', '"')

    list = list
    plist = list
    pair = tuple
    object = dict
    
    #number = v_args(inline=True)(int)

    @v_args(inline=True)
    def number(self, n):
        self.count += int(n)
        #print(self.count)
        return int(n)
    
    @v_args(inline=True)
    def pnumber(self, n):
        self.count += int(n)
        #print(self.count)
        return int(n)

    null = lambda self, _: None
    true = lambda self, _: True
    false = lambda self, _: False


def day12_1(data):
    #data = read_input(2015, "121")   

    tree2Json = TreeToJson()
    json_parser = Lark(json_grammar, parser='lalr', transformer=tree2Json)
    parse = json_parser.parse
    parsedTree = parse(data[0])
    
    tree2Json.visit_topdown(tree=parsedTree)
    print("sum:", tree2Json.sum())
    
    #print("sum:", parsedTree)
    
    result = tree2Json.sum()
    AssertExpectedResult(111754, result, 1)
    return result



class TreeToJson2(Visitor):
    count = 0
    pcount = 0
    total = 0
    hasRed = False
    DEBUG_MODE = False

    def _updateCounts(self):
        if self.DEBUG_MODE:
            print("hasRead:",self.hasRed)

        if self.hasRed:            
            self.total += 0#self.count   
            self.hasRed = False                   
        else:
            self.total += self.count + self.pcount
            self.count = 0
        self.pcount = 0

    def sum(self):
        return self.total

    @v_args(inline=True)
    def string(self, s):
        return s[1:-1].replace('\\"', '"')

    @v_args(inline=True)
    def pstring(self, s):        
        ss = s[1:-1].replace('\\"', '"')
        if self.DEBUG_MODE:
            print("eval pstring", ss)

        if ss == 'red':
            self.hasRed = True

        return ss

    def list(self, items):
        self._updateCounts()        
        return list(items)
    
    def plist(self, items):
        if self.DEBUG_MODE:
            print("eval plist", items)
            print("eval plist count", self.count)
            print("eval plist pcount", self.pcount)
        if self.hasRed:
            self.count = 0
        #self._updateCounts()        
        return list(items)

    def pair(self, key_value):
        k, v = key_value
        return k, v

    def object(self, items):
        if self.DEBUG_MODE:
            print("val obj", items)

        self._updateCounts()
        if self.hasRed:
            self.hasRed = False
        return dict(items)

    @v_args(inline=True)
    def number(self, n):
        self.count += int(n)
        return int(n)
    
    @v_args(inline=True)
    def pnumber(self, n):
        self.pcount += int(n)
        return int(n)

    null = lambda self, _: None
    true = lambda self, _: True
    false = lambda self, _: False

def parseAndGetResult(input):
    treeToJson = TreeToJson2()
    json_parser = Lark(json_grammar, parser='lalr', transformer=treeToJson)
    parse = json_parser.parse
    parsedTree = parse(input)    
    treeToJson.visit(tree=parsedTree)
    #print( parsedTree.pretty() )
    
    return treeToJson.sum()


def day12(input):
    import re
    return sum(int(n) for n in re.findall(r"-?\d+", input))

def troubleshoot2(input):
    def eval_obj_without_red(match):
        obj = match.group()
        if ':"red"' not in obj:
            return str(day12(obj))
        else:
            return str(0)
    
    import re
    while ':"red"' in input:
        input = re.sub(r"{[^{}]*}", eval_obj_without_red, input)
    return day12(input)

import json

def testSamples():
    sample1 = '[1,2,3]'
    sample2 = '[1,{"c":"red","b":2},3]'
    sample3 = '{"d":"red","e":[1,2,3,4],"f":5}'
    sample4 = '[1,"red",5]'
    sample5 = '{"c": ["red", 1], "c":1}'
    sample6 = '{"c": ["red", {"c": ["red", 1], "c":1}], "c":1}'
    sample7 = '{"c": ["red", {"c": ["red", {"d":"red","e":[1,2,3,4],"f":5}, 2], "c":1}], "c":1}'
    sample8 = '{"red":"blue","e":[1,2,3,4],"f":5}'
    sample9 = '{"e": 86,"c": 23,"a": {"a": [120,169,"green","red","orange"],"b": "red"},"g": "yellow","b": ["yellow"],"d": "red","f": -19}'
    sample10 = '{"a": [120,169,"green","red","orange"],"b": "red"}'


    ex1 = troubleshoot(sample1)
    ex2 = troubleshoot(sample2)
    ex3 = troubleshoot(sample3)
    ex4 = troubleshoot(sample4)
    ex5 = troubleshoot2(sample5)
    ex6 = troubleshoot2(sample6)
    ex7 = troubleshoot2(sample7)
    ex8 = troubleshoot(sample8)
    ex9 = troubleshoot(sample9)
    ex10 = troubleshoot(sample10)

    samples = [sample1, sample2, sample3, sample4, sample5, sample6, sample7, sample8, sample9, sample10]
    expected = [ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex9, ex10]       

    for sample, expected in zip(samples, expected):  
        res = parseAndGetResult(sample)
        if res != expected:
            print("Got", res, "but expected", expected, "for sample", sample)
        else:
            print("Correct result for sample ", sample,":", expected)

def troubleshoot(input):
    def sum_numbers(obj):
        if type(obj) == type(dict()):
            if "red" in obj.values():
                return 0
            return sum(map(sum_numbers, obj.values()))

        if type(obj) == type(list()):
            return sum(map(sum_numbers, obj))

        if type(obj) == type(0):
            return obj

        return 0

    data = json.loads(input)
    return sum_numbers(data)

# incomplete :(
def day12_2(data):    
    #data = read_input(2015, "121")   
    testSamples()

    result = 0
    result = parseAndGetResult(data[0]) 
    
    # 54040 too low
    # 89528 too high
    # should be around 65k :\
    AssertExpectedResult(111754, result, 2)
    return result

'''
Day 13: Knights of the Dinner Table  
'''

# 0: person, 2: signal, 3: value
def parseConstraints(data, includeMyself = False):
    people = dict()
    me = 'Myself'
    if includeMyself:
        people[me] = dict()

    for line in data:
        allData = line.split(" ")
        person = allData[0]
        neighbour = allData[len(allData)-1][:-1]
        value = int(allData[3])
        if allData[2] == 'lose':
            value = -value

        if person not in people.keys():
            people[person] = {neighbour : value}
            if includeMyself:
                people[me][person] = 0
                people[person][me] = 0
        else:
            people[person][neighbour] = value
        
    
    #print(len(people))
    return people

def computeChangeInHappines(p, people):
    happines = 0
    for i in range(0, len(p)):
        person = p[i]
        neighbourOne = p[(i-1) % len(p)]
        neighbourTwo = p[(i+1) % len(p)]
        preferences = people[person]
        happines += preferences[neighbourOne] + preferences[neighbourTwo]

    return happines

#Day 13, part 1: 733 (0.235 secs)
def day13_1(data):
    people = parseConstraints(data)
 
    perm = permutations(people.keys())
    maxHappines = 0
    for p in perm:
        happines = computeChangeInHappines(p, people)
        if happines > maxHappines:
            maxHappines = happines

    result = maxHappines
    return result

# Day 13, part 2: 725 (2.359 secs)
def day13_2(data):
    people = parseConstraints(data, True)
 
    perm = permutations(people.keys())
    maxHappines = 0
    for p in perm:
        happines = computeChangeInHappines(p, people)
        if happines > maxHappines:
            maxHappines = happines

    result = maxHappines
    return result

'''
Day 14: Reindeer Olympics
'''


# 0: reindeer, 3: speed, 6: duration, :-2: rest
# Rudolph can fly 22 km/s for 8 seconds, but then must rest for 165 seconds.
def parseReindeerData(data):
    reindeers = dict()
    
    for line in data:
        allData = line.split(" ")
        reindeer = allData[0]       
        speed = int(allData[3])
        duration = int(allData[6])
        rest = int(allData[13])

        if reindeer not in reindeers.keys():
            reindeers[reindeer] = (speed, duration, rest)

    #print(len(people))
    return reindeers

def computeDistanceAtTime(time, reindeers, distances):
    winningDistance = 0
    for reindeer in reindeers.keys():       

        (speed, duration, rest) = reindeers.get(reindeer)

        timesFlew = math.floor(time/(duration+rest))
        lastTakeOff = timesFlew * (duration + rest) 

        if lastTakeOff + duration < time:
            timesFlew +=  1      
            distance = timesFlew * speed * duration
        else:
            distance = timesFlew * speed * duration + ((time - lastTakeOff) * speed)
        
        distances[reindeer] = distance

        if distance > winningDistance:
            winningDistance = distance

    return (distances, winningDistance)

# 2660 too low
# 2720 too high
#Day 14, part 1: 2696 (0.002 secs)
def day14_1(data):    
    #data = read_input(2015, "141")  
    reindeers = parseReindeerData(data)
    scoreboard = {x : 0 for x in reindeers}
    #print(reindeers)
    (_, result) = computeDistanceAtTime(2503, reindeers, scoreboard)
    #result = computeDistanceAtTime(1000, reindeers)
    return result

def computeScoreboard(winningDistance, distances, scoreboard):
    for reindeer in distances:
        distance = distances.get(reindeer)
        if distance == winningDistance:
            scoreboard[reindeer] += 1

    return scoreboard

#1085 too high
#Day 14, part 2: 1085 (0.028 secs)
def day14_2(data):
    #data = read_input(2015, "141")  
    reindeers = parseReindeerData(data)
    distances = {x : 0 for x in reindeers}
    scoreboard = {x : 0 for x in reindeers}
    result = 0
    time = 2503

    for i in range(time):
        (distances, winningDistance) = computeDistanceAtTime(i, reindeers, distances)
        scoreboard = computeScoreboard(winningDistance, distances, scoreboard)
    
    result = max(scoreboard.values())
    
    return result

'''
Day 15: Science for Hungry People
'''

def parseIngredients(data):
    ingredients = dict()
    Ingredient = namedtuple('Ingredient', 'name capacity durability flavor texture calories')

    # Sprinkles: capacity 2, durability 0, flavor -2, texture 0, calories 3
    for line in data:
        ingredientsData = line.split(" ")
        name = ingredientsData[0][:-1]
        capacity = ingredientsData[2][:-1]
        durability = ingredientsData[4][:-1]
        flavor = ingredientsData[6][:-1]
        texture = ingredientsData[8][:-1]
        calories = ingredientsData[10]
        ingredient = Ingredient(name, int(capacity), int(durability), int(flavor), int(texture), int(calories))
        ingredients[name] = ingredient

    return ingredients

def computeCookieRecipeScore(ingredients, recipe):
    #recipe =  name: teaspoons
    capacity = 0
    durability = 0
    flavor = 0
    texture = 0
    calories = 1

    for (name, teaspoon) in recipe:
        capacity += (teaspoon * ingredients[name].capacity)
        durability += (teaspoon * ingredients[name].durability)
        flavor += (teaspoon * ingredients[name].flavor)
        texture += (teaspoon * ingredients[name].texture)

    if (capacity < 0 or durability < 0 or flavor < 0 or texture <0):
        return 0

    return capacity * durability * flavor * texture * calories

#failed to describe this using LP :(
def solveWithLinearProgramming():
    from scipy.optimize import linprog

    # Set up values relating to both minimum and maximum values of y

    # x y z w
    coefficients_inequalities = [[2,0,-2,0], [0,5,-3,0], [0,0,5,-1], [0,-1,0,5]]  # require -1*x + -1*y <= -180
    constants_inequalities = [1,1,1,1]

    coefficients_equalities = [[2,4,0,-6]]  # require 2x + 4y + 0z - 6w = 100
    constants_equalities = [100]
    bounds_x = (0, 100)  # require 30 <= x <= 160
    bounds_y = (0, 100)  # require 10 <= y <= 60
    bounds_z = (0, 100)
    bounds_w = (0, 100)

    # Find and print the maximal value of y = minimal value of -y
    coefficients_max_y = [1,1,1,1]  # minimize 0*x + -1*y
    res = linprog(coefficients_max_y,
                A_ub=coefficients_inequalities,
                b_ub=constants_inequalities,
                A_eq=coefficients_equalities,
                b_eq=constants_equalities,
                bounds=(bounds_x, bounds_y, bounds_z, bounds_w))
    print('Maximum value of y =', res.fun)  # opposite of value of -y

# 91352448 too high
# Day 15, part 1: 21367368 (28.625 secs)
# Day 15, part 2: 1766400 (27.154 secs)
def day15_1(data):
    #data = read_input(2015, "151")  

    ingredients = parseIngredients(data)       
    
    combinations = [combination for combination in permutations(range(1,100), len(ingredients)) if sum(combination) == 100 ]
    bestScore = 0
    for recipe in combinations:
        score = computeCookieRecipeScore(ingredients, (zip(ingredients.keys(), recipe)))
        if score > bestScore:
            bestScore = score
   
    result = bestScore
    
    result = 0
    AssertExpectedResult(21367368, result)    
    return result

def day15_2(data):
    #data = read_input(2015, "151")  
    ingredients = parseIngredients(data)      

    combinations = [combination for combination in permutations(range(1,100), len(ingredients)) 
                        if sum(combination) == 100 and 
                        sum(np.multiply(combination, [i.calories for i in ingredients.values()])) == 500
                   ]
    bestScore = 0
    for recipe in combinations:
        score = computeCookieRecipeScore(ingredients, (zip(ingredients.keys(), recipe)))
        if score > bestScore:
            bestScore = score
   
    result = bestScore
    AssertExpectedResult(1766400, result)    
    return result



'''
    Day 16: Aunt Sue
'''
'''
Clues
    children: 3
    cats: 7
    samoyeds: 2
    pomeranians: 3
    akitas: 0
    vizslas: 0
    goldfish: 5
    trees: 3
    cars: 2
    perfumes: 1
'''
def validateProperty(prop, value):  
    if(prop == 'children'):
        if value != 3: return False
    if(prop == 'cats'):
        if value != 7: return False
    if(prop == 'samoyeds'):
        if value != 2: return False
    if(prop == 'pomeranians'):
        if value != 3: return False
    if(prop == 'akitas'):
        if value != 0: return False
    if(prop == 'vizslas'):
        if value != 0: return False
    if(prop == 'goldfish'):
        if value != 5: return False
    if(prop == 'trees'):
        if value != 3: return False
    if(prop == 'cars'):
        if value != 2: return False
    if(prop == 'perfumes'):
        if value != 1: return False

    #print("validating property",prop,"with value",value)
    return True

def parseAunstSue(data, validatePropertiesFunc):
    #Sue 1: goldfish: 6, trees: 9, akitas: 0
    candidates = dict()
    for line in data:
        properties = dict()
        inputData = line.split(" ")
        number = inputData[1][:-1]

        # parse and validate properties
        for i in range(2, len(inputData)-1, 2):            
            prop = inputData[i][:-1]
            value = inputData[i+1]
            if value.find(',') != -1:
                value = int(value[:-1])
            
            if validatePropertiesFunc(prop, int(value)):
                properties[prop] = value
            else:
                break

        if properties:
            # all properties are valid, add as candidate
            candidates[number] = dict(properties)
    return candidates        

# 473, too high
# Day 16, part 1: 103 (0.114 secs)
def day16_1(data):
    #data = read_input(2015, "151")  
    candidates = parseAunstSue(data, validateProperty)     

    mostConditionsSatisfied = 0
    for aunt in candidates.keys():
        property = candidates[aunt]

        if len(list(property))  > int(mostConditionsSatisfied):
            mostConditionsSatisfied = len(list(property)) 
            auntSue = int(aunt)
   
    result = auntSue
    AssertExpectedResult(103, result)    
    return result

def validatePropertyV2(prop, value):  
    if(prop == 'children'):
        if value != 3: return False
    if(prop == 'cats'):
        if value <= 7: return False
    if(prop == 'samoyeds'):
        if value != 2: return False
    if(prop == 'pomeranians'):
        if value >= 3: return False
    if(prop == 'akitas'):
        if value != 0: return False
    if(prop == 'vizslas'):
        if value != 0: return False
    if(prop == 'goldfish'):
        if value >= 5: return False
    if(prop == 'trees'):
        if value <= 3: return False
    if(prop == 'cars'):
        if value != 2: return False
    if(prop == 'perfumes'):
        if value != 1: return False

    #print("validating property",prop,"with value",value)
    return True

# Day 16, part 2: 405 (0.005 secs)
def day16_2(data):
    #data = read_input(2015, "151")  
    candidates = parseAunstSue(data, validatePropertyV2)     

    mostConditionsSatisfied = 0
    for aunt in candidates.keys():
        property = candidates[aunt]

        if len(list(property))  > int(mostConditionsSatisfied):
            mostConditionsSatisfied = len(list(property)) 
            auntSue = int(aunt)
   
    result = auntSue
    AssertExpectedResult(405, result)    
    return result

'''
    Day 17: No Such Thing as Too Much
'''

def fillContainers(containers, eggnogLiters, part2 = False):
    containers.sort()
    combinations = []
    minContainers = 0
    i = 2
    while i < len(containers)-1:
        for l in itertools.combinations(containers,i):      
            if sum(list(l)) > eggnogLiters:
                continue
            if sum(list(l)) == eggnogLiters:                                  
                if part2 and minContainers == 0:
                    minContainers = i
                if l not in combinations:
                    if part2 and i == minContainers:
                        combinations.append(list(l)) 
                    elif not part2:
                        combinations.append(list(l)) 
        i += 1
    return len(combinations)

#Day 17, part 1: 4372 (0.267 secs)
def day17_1(data):
    data = read_input(2015, "17")
    eggnogLiters = 150    
    #eggnogLiters = 25
    #containers = [20, 15, 10, 5, 5]
    containers = []
    for line in data:
        containers.append(int(line))

    result = fillContainers(containers, eggnogLiters)
    AssertExpectedResult(4372, result)   
    return result

#Day 17, part 2: 4 (0.186 secs)
def day17_2(data):
    data = read_input(2015, "17")
    eggnogLiters = 150    
    containers = []
    for line in data:
        containers.append(int(line))

    result = fillContainers(containers, eggnogLiters, part2 = True)
    AssertExpectedResult(4, result)   
    return result

#### Day 18

def countOnNeighbours(grid, posX, posY, part2 = False):
    directionsDelta = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 1),(-1, -1)]
    corners = [ (0,0), (0, len(grid[0])-1), (len(grid)-1, 0), (len(grid)-1, len(grid[0])-1) ]

    if posX == 0:
        if (-1,0) in directionsDelta:
            directionsDelta.remove((-1,0))
        if (-1,-1) in directionsDelta:   
            directionsDelta.remove((-1,-1))
        if (-1,1) in directionsDelta:
            directionsDelta.remove((-1,1))
    if posX == len(grid)-1:
        if (1,0) in directionsDelta:
            directionsDelta.remove((1,0))
        if (1,-1) in directionsDelta:
            directionsDelta.remove((1,-1))
        if (1,1) in directionsDelta:
            directionsDelta.remove((1,1))
    if posY == 0:
        if (0,-1) in directionsDelta:
            directionsDelta.remove((0,-1))
        if (-1,-1) in directionsDelta:
            directionsDelta.remove((-1,-1))
        if (1,-1) in directionsDelta:
            directionsDelta.remove((1,-1))
    if posY == len(grid[0])-1:
        if (0,1) in directionsDelta:
            directionsDelta.remove((0,1))
        if (1,1) in directionsDelta:
            directionsDelta.remove((1,1))
        if (-1,1) in directionsDelta:
            directionsDelta.remove((-1,1))

    #print(directionsDelta)

    on = 0
    for x,y in directionsDelta:
        xx = posX+x
        yy = posY+y
        if grid[posY+y][posX+x] == '#':
            on +=1
        elif part2:
            if (xx,yy) in corners:
                on += 1

    return on

def compute_new_map(map, cycles, part2 = False):
    rows = len(map)
    columns = len(map[0])   
    newGrid = copy.deepcopy(map)
    corners = [ (0,0), (0, columns-1), (rows-1, 0), (rows-1, columns-1) ]

    cycle = 0
    while cycle < cycles:
        for y in range(rows):
            for x in range(columns): 
                if part2 and (x,y) in corners:
                    continue

                onNeighbours = countOnNeighbours(map,x,y,part2)
                if map[y][x] == '#':
                    if onNeighbours == 2 or onNeighbours == 3:
                        newGrid[y][x] = '#'
                    else:
                        newGrid[y][x] = '.'
                else:
                    if onNeighbours == 3:
                        newGrid[y][x] = '#'
                    
        map = copy.deepcopy(newGrid)
        #printMap(map)
        cycle += 1
    return map

# 3896 too high
# 2438 too high
#Day 18, part 1: 821 (0.951 secs)
def day18_1(data):
    #data = read_input(2015, "181")
    result = 0
    
    map = buildMapGrid(data, initValue='.')
    rows = len(data)
    columns = len(data[0])    

    cycles = 100
    #printMap(map)
    map = compute_new_map(map, cycles)

    result = sum( [map[i].count('#') for i in range(rows) ] )
    AssertExpectedResult(821, result)   
    return result

# 885 too low
#Day 18, part 2: 886 (1.836 secs)
def day18_2(data):
    #data = read_input(2015, "181")
    result = 0
    map = buildMapGrid(data, initValue='.')
    rows = len(data)
    columns = len(data[0])    
    corners = [ (0,0), (0, columns-1), (rows-1, 0), (rows-1, columns-1) ]

    for x,y in corners:
        map[y][x] = '#'

    cycles = 100
    #printMap(map)
    
    map = compute_new_map(map, cycles, part2=True)
    result = sum( [map[i].count('#') for i in range(rows) ] )
    AssertExpectedResult(886, result)   
    return result

#### Day 19

def get_new_molecule(molecule, i, rep, double_rep):
    if i == 0:
        if double_rep:
            new_molecule = rep + molecule[i+2:]
        else:
            new_molecule = rep + molecule[i+1:]
    elif i == len(molecule)-1:
        new_molecule = molecule[:i-1] + rep
    else:
        if double_rep:
            before = molecule[:i]
            after = molecule[i+2:]
        else:
            before = molecule[:i]
            after = molecule[i+1:]       
        new_molecule = before + rep + after
    return new_molecule

def count_distinct_molecules(molecule, replacements):
    molecules = set()
    
    for key in replacements.keys():
        if len(key) == 2:
            double_rep = True
        else:
            double_rep = False
        reps = replacements[key]
        indices = [index for index in range(len(molecule)) if molecule.startswith(key, index)]

        for rep in reps:
            for i in indices:
                molecules.add(get_new_molecule(molecule, i, rep, double_rep))
        
        #print("key:",key, indices)
    return len(molecules)
    
#603 too high
#567 too high
#Day 19, part 1: 509 (0.021 secs)
def day19_1(data):
    #data = read_input(2015, "191")
    result = 0
    molecule = data[-1]

    replacements = {}
    for line in data[:-2]:
        replacement = line.split(' => ')
        if replacement[0] not in replacements:
            replacements[replacement[0]] = []
        replacements[replacement[0]].append(replacement[1])
    
    #print(replacements)
    #print(molecule)
    result = count_distinct_molecules(molecule, replacements)
    AssertExpectedResult(509, result)   
    return result


def make_distinct_molecules(medicine_molecule, replacements, molecule, molecules):
    
    print(molecules)
    print(molecule)

    if len(molecule) > len(medicine_molecule):
        return 0
    
    if molecule == medicine_molecule:
        print("BINGO")
        return 1
        
    for key in replacements.keys():
        if len(key) == 2:
            double_rep = True
        else:
            double_rep = False
        reps = replacements[key]
        indices = [index for index in range(len(molecule)) if molecule.startswith(key, index)]

        min_steps = sys.maxsize
        for rep in reps:
            for i in indices:
                new_molecule = get_new_molecule(molecule, i, rep, double_rep)
                if new_molecule in molecules:
                    return 0
                molecules.add(new_molecule)
                
                steps = 1 + make_distinct_molecules(medicine_molecule, replacements, new_molecule, molecules)
                if steps < min_steps:
                    min_steps = steps
        
        #print("key:",key, indices)
    return min_steps



def day19_2(data):
    data = read_input(2015, "191")
    result = 0
    medicine_molecule = data[-1]
    molecule = 'e'
    molecules = set()

    replacements = {}
    for line in data[:-2]:
        replacement = line.split(' => ')
        if replacement[0] not in replacements:
            replacements[replacement[0]] = []
        replacements[replacement[0]].append(replacement[1])
    

    result = make_distinct_molecules(medicine_molecule, replacements, molecule, molecules)
    AssertExpectedResult(509, result)   
    return result


if __name__ == "__main__":
    main(sys.argv, globals(), 2015)

