from threading import Thread
import functools
from timeit import default_timer as timer
import sys
import os
import copy

# might need some tweaks depending on the map
def buildGraphFromMap_v2(map, emptyCell, is_connected):
    graph = {}
    sizeX = len(map[0])
    sizeY = len(map)

    for y in range(sizeY):
        for x in range(sizeX):

            east = (x+1, y)
            west = (x-1, y)
            north = (x, y-1)
            south = (x, y+1)
            
            neighbours = []
            if map[y][x] != emptyCell:
                if 0 <= east[0] < sizeX and 0 <= east[1] < sizeY:
                    if map[east[1]][east[0]] != emptyCell and is_connected(map, (x,y), east):
                        neighbours.append(east)
                
                if 0 <= west[0] < sizeX and 0 <= west[1] < sizeY:                        
                    if map[west[1]][west[0]] != emptyCell and is_connected(map, (x,y), west):
                        neighbours.append(west)
                
                if 0 <= north[0] < sizeX and 0 <= north[1] < sizeY: 
                    if map[north[1]][north[0]] != emptyCell and is_connected(map, (x,y), north):
                        neighbours.append(north)
                
                if 0 <= south[0] < sizeX and 0 <= south[1] < sizeY: 
                    if map[south[1]][south[0]] != emptyCell and is_connected(map, (x,y), south):
                        neighbours.append(south)
            
            graph[(x,y)] = neighbours
    return graph

def buildGraphFromMap(map):
    graph = {}
    sizeX = len(map[0])
    sizeY = len(map)

    for y in range(1,sizeY-1):
        for x in range(1,sizeX-1):

            east = (x+1, y)
            west = (x-1, y)
            north = (x, y-1)
            south = (x, y+1)
            
            neighbours = []
            if map[y][x] != ' ' and map[y][x] != '#':
                if map[east[1]][east[0]] != ' ' and map[east[1]][east[0]] != '#':
                    neighbours.append(east)
                if map[west[1]][west[0]] != ' ' and map[west[1]][west[0]] != '#':
                    neighbours.append(west)
                if map[north[1]][north[0]] != ' ' and map[north[1]][north[0]] != '#':
                    neighbours.append(north)
                if map[south[1]][south[0]] != ' ' and map[south[1]][south[0]] != '#':
                    neighbours.append(south)
            
            graph[(x,y)] = neighbours
    return graph

def find_starting_point(map, starting_point):
    rows = len(map)
    columns = len(map[0])

    for y in range(rows):          
        for x in range(columns):
            if map[y][x] == starting_point:
                return (x,y)
    
    return None

# fileMode is an old parameter, idea was to have a print in console version as well
def printMap(map, fileMode = True, symbolsMap = {}):
    map = copy.deepcopy(map)
    if symbolsMap:
        rows = len(map)
        columns = len(map[0])
        for y in range(rows) :
            for x in range(columns):
                tile = map[y][x]                
                new_tile = symbolsMap.get(tile)
                tile = new_tile if new_tile else tile
                map[y][x] = tile


    if fileMode:
        file1 = open("MyMap.txt","a") 
    
        for l in map:
            for j in range(len(l)):
                file1.write(l[j])
            file1.write("\n")
        file1.write("\n")
        file1.write("\n")
        file1.close() 


def build_empty_grid(rows, columns, initValue='', withPadding = True):
    '''
    file_dir = os.path.dirname(os.path.realpath(__file__))
    with open("{0}/../common/teste.txt".format(file_dir, 2020, 7), "r") as fileReader:
        data = [line.rstrip('\n') for line in fileReader]
    '''
    rows = rows + 2 if withPadding else rows
    columns = columns + 2 if withPadding else columns
    
    map = [ [ (initValue) for i in range(columns) ] for j in range(rows) ]    
    
    #printMap(map)
    return map

def buildMapGrid(data, initValue='', withPadding = True, symbolMap = {}):
    data = copy.deepcopy(data)
    '''
    file_dir = os.path.dirname(os.path.realpath(__file__))
    with open("{0}/../common/teste.txt".format(file_dir, 2020, 7), "r") as fileReader:
        data = [line.rstrip('\n') for line in fileReader]
    '''
    rows = len(data) + 2 if withPadding else len(data)
    columns = len(data[0]) + 2 if withPadding else len(data[0])

    map = [ [ (initValue) for i in range(columns) ] for j in range(rows) ]    
    
    for y in range(1,rows-1) if withPadding else range(rows):
        for x in range(1,columns-1) if withPadding else range(columns):
            tile = data[y-1][x-1] if withPadding else data[y][x]
            if symbolMap:
                new_tile = symbolMap.get(tile)
                tile = new_tile if new_tile else tile
            map[y][x] = tile

    #printMap(map)
    return map
    
