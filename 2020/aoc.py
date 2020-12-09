# Based on template from https://github.com/scout719/adventOfCode/
# -*- coding: utf-8 -*-
import functools
import math
import os
import sys
import time
import copy
import re

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


if __name__ == "__main__":
    main(sys.argv, globals(), 2020)

