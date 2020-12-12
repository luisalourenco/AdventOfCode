# Based on template from https://github.com/scout719/adventOfCode/
# -*- coding: utf-8 -*-
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

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
print(FILE_DIR)
sys.path.insert(0, FILE_DIR + "/")
sys.path.insert(0, FILE_DIR + "/../")
sys.path.insert(0, FILE_DIR + "/../../")
from common.utils import read_input, main, clear  # NOQA: E402
from common.mapUtils import printMap, buildMapGrid, buildGraphFromMap
from common.graphUtils import printGraph, find_all_paths, find_path, find_shortest_path, find_shortest_pathOptimal, bfs, dfs, Graph

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

def day1_1(data):
    sum = 2020
    data = sorted(data, key=int)

    for i in range(0, len(data)):
        elem1 = int(data[i])
        for j in range(i, len(data)):
            elem2 = int(data[j])
            if (elem1 + elem2 == sum):
                 return elem1 * elem2

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
    return result

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

def day3_1(data):
    return day3Aux(data, 3, 1)
    
def day3_2(data):
    return day3Aux(data, 3, 1) * day3Aux(data, 1, 1) * day3Aux(data, 5, 1) * day3Aux(data, 7, 1) * day3Aux(data, 1, 2)  

# Day 4 methods

def addFields(line, passportFields):
    pairs = line.split(" ")            
    for field in pairs:
        passportFields.append(field.split(":")[0]) 

def isPassportValid(fields, passportFields):
    numFields = 0
    return fields.issubset(passportFields)
    #for p in passportFields:
    #    if p in fields:
    #        numFields+=1

    #return numFields >= 7
        

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

def day5_1(data):     
    #data = read_input(2020, "51")
    maxSeatId = 0
    for boardingPass in data:
        row = computeRow(boardingPass[:-3])
        column = computeColumn(boardingPass[-3:])
        seatID = row*8+column
        if seatID > maxSeatId:
            maxSeatId = seatID
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

    for seat in seats:
        if seat-1 in results and seat+1 in results and seat not in results:
            return seat


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

    return sum

def processUniqueAnswers(answers, groupSize):
    count = 0
    for k, v in answers.items():
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
    return computeBags(bags, 0, contents)


def handheldMachine(program, pc, acumulator):
    instruction = program[pc] 
    op = instruction[:3]
    arg = int(instruction[3:])   
               
    if op == 'acc':
        acumulator += arg
        pc += 1
    elif op == 'jmp':
        pc += arg
    elif op == 'nop':
        pc += 1

    return (acumulator, pc)

def preventInfiniteLoop(data):
    visited = set()
    pc = 0
    acumulator = 0

    while True:  
        acumulator, pc = handheldMachine(data, pc, acumulator) 

        # if we reach this state it means we are in an infinite loop
        if pc in visited:
            break
        else:
            visited.add(pc)

    return acumulator

def day8_1(data):    
    #data = read_input(2020, "81")
    return preventInfiniteLoop(data)

def replaceOperation(program, pc):
    instruction = program[pc]
    op = instruction[:3]    
    
    if op == 'jmp':
       program[pc] = program[pc].replace('jmp','nop')
    elif op == 'nop':
       program[pc] = program[pc].replace('nop','jmp')
    
    return program

def fixInfiniteLoop(data, switchOperations):    
    success = False
    targetPC = len(data)

    while not success:
        pc = 0
        visited = set()        
        acumulator = 0
      
        switchPC = switchOperations.pop(0)
        program = replaceOperation(list(data), switchPC)

        while True:                                         
            acumulator, pc = handheldMachine(program, pc, acumulator)
            
            # if we reach the last intruction then we fixed the infinite loop!
            if pc == targetPC:
                success = True
                break

            # if we reach this state it means we are in an infinite loop :(
            if pc in visited:
                break
            else:
                visited.add(pc)            

    return acumulator

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
    return fixInfiniteLoop(data, switchOperations)


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

def day9_1(data):    
    #data = read_input(2020, "91")
    data = [int(numeric_string) for numeric_string in data]
    
    preambleSize = 25
    preamble = 0

    for number in data[preambleSize:]:
        valid = sumSearch(data, number, preamble, preambleSize)
        preamble += 1
        if not valid:
            return number

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
    return breakXMAS(data, sum, preambleSize)

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
    return jolt + deltaJoltage >= deviceJoltage, diff1 * diff3, diffVoltages

def day10_1(data):    
    #data = read_input(2020, "102")
    data = [int(numeric_string) for numeric_string in data]

    deltaJoltage = 3
    deviceJoltage = max(data) + deltaJoltage

    isValid, result, _ = checkAdapterArrangement(data, deviceJoltage)
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

    n = len(data)
    f = lambda i, j : j > i and data[j] - data[i] <= 3
    m = np.fromfunction(np.vectorize(f), (n, n), dtype=np.int64).astype(np.int64)
    m[n-1, n-1] = 1

    aux = np.linalg.matrix_power(m, n)
    ans = aux[0, n - 1]

    return ans*2

def day10_2(data):    
    #data = read_input(2020, "102")
    data = [int(numeric_string) for numeric_string in data]   
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
    isValid, result, diffVoltages = checkAdapterArrangement(data, deviceJoltage) 

    string_ints = [str(int) for int in diffVoltages]    
    str_of_ints = "".join(string_ints)
    str_of_ints = (str_of_ints.replace("1111","A").replace("111","B").replace("11","C") )
    result = (7 ** str_of_ints.count("A")) * (4 ** str_of_ints.count("B")) * (2 ** str_of_ints.count("C"))
    
    #return result
    print(result == matrixAdjanciesBasedSolution(data))

    return matrixAdjanciesBasedSolution(data)



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

def day11_1(data):   
    #data = read_input(2020, "111")
    
    newMap = data  
    while True:
        oldMap = newMap     
        newMap, changed = applySeatRules(oldMap)
        if not changed:              
            count = sum( [ seatRow.count("#") for seatRow in newMap])
            return count
            #break
        #printMap(m)
        # initial approach until I changed applySeatRules
        #stop, count = shouldStop(oldMap, newMap)        



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
    count = 0
    while True:
        oldMap = newMap     
        newMap, changed = applyNewSeatRules(oldMap)
        printMap(newMap)

        if not changed:              
            return sum( [ seatRow.count("#") for seatRow in newMap])



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


def day12_1(data):   
    #data = read_input(2020, "121")   
    result = PASystem(data)
    print(result == 1007)
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

    # too low 15076
    # too high 123430
    # too high 57574
    # high 462890
    # wrong 28766
    # wrong 34064
    result = PASystemWithWaypoint(data)

    print(result == 41212)
    return result
    


if __name__ == "__main__":
    main(sys.argv, globals(), 2020)

