# Based on template from https://github.com/scout719/adventOfCode/
# -*- coding: utf-8 -*-
import functools
import math
import os
import sys
import time
import copy

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
print(FILE_DIR)
sys.path.insert(0, FILE_DIR + "/")
sys.path.insert(0, FILE_DIR + "/../")
sys.path.insert(0, FILE_DIR + "/../../")
from common.utils import read_input, main, clear  # NOQA: E402
from common.graphUtils import find_all_paths, find_path, find_shortest_path, find_shortest_pathOptimal, bfs, dfs, Graph

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
    return data


if __name__ == "__main__":
    main(sys.argv, globals(), 2020)

