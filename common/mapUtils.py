from threading import Thread
import functools
from timeit import default_timer as timer
import sys
import os
import copy


# might need some tweaks depending on the map
def buildGraphFromMap_v3(map, emptyCell, is_connected):
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
                        neighbours.append((east, 1))
                
                if 0 <= west[0] < sizeX and 0 <= west[1] < sizeY:                        
                    if map[west[1]][west[0]] != emptyCell and is_connected(map, (x,y), west):
                        neighbours.append((west, 1))
                
                if 0 <= north[0] < sizeX and 0 <= north[1] < sizeY: 
                    if map[north[1]][north[0]] != emptyCell and is_connected(map, (x,y), north):
                        neighbours.append((north, 1))
                
                if 0 <= south[0] < sizeX and 0 <= south[1] < sizeY: 
                    if map[south[1]][south[0]] != emptyCell and is_connected(map, (x,y), south):
                        neighbours.append((south, 1))
            
            graph[(x,y)] = neighbours
    return graph

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

def buildGraphFromMap(map, withWeights=False, noPadding=False):
    graph = {}
    sizeX = len(map[0])
    sizeY = len(map)


    for y in range(0, sizeY) if noPadding else range(1,sizeY-1):
        for x in range(0, sizeX) if noPadding else range(1,sizeX-1):

            east = (x+1, y) if x+1 <= sizeX-1 else None
            west = (x-1, y) if x-1 >= 0 else None
            north = (x, y-1) if y-1 >= 0 else None
            south = (x, y+1) if y+1 <= sizeY-1 else None
            
            neighbours = []
            if map[y][x] != ' ' and map[y][x] != '#':
                
                east_map = map[east[1]][east[0]] if east else ' '
                if east_map != ' ' and east_map != '#' and east:
                    neighbours.append((east, int(east_map))) if withWeights else neighbours.append(east)

                west_map = map[west[1]][west[0]] if west else ' '
                if west_map != ' ' and west_map != '#':
                    neighbours.append((west, int(west_map))) if withWeights else neighbours.append(west)

                north_map = map[north[1]][north[0]] if north else ' '
                if north_map != ' ' and north_map != '#':
                    neighbours.append((north, int(north_map))) if withWeights else neighbours.append(north)

                south_map = map[south[1]][south[0]] if south else ' '
                if south_map != ' ' and south_map != '#':
                    neighbours.append((south, int(south_map))) if withWeights else neighbours.append(south)

            #if x >= 0 and y >= 0 and x <= sizeX-1 and y <= sizeY-1:
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
    
