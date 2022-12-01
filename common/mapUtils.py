from threading import Thread
import functools
from timeit import default_timer as timer
import sys
import os
import copy

# might need some tweaks depending on the map
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

# fileMode is an old parameter, idea was to have a print in console version as well
def printMap(map, fileMode = True):
    if fileMode:
        file1 = open("MyMap.txt","a") 
    
        for l in map:
            for j in range(len(l)):
                file1.write(l[j])
            file1.write("\n")
        file1.write("\n")
        file1.write("\n")
        file1.close() 


def buildMapGrid(data, initValue=''):
    data = copy.deepcopy(data)
    '''
    file_dir = os.path.dirname(os.path.realpath(__file__))
    with open("{0}/../common/teste.txt".format(file_dir, 2020, 7), "r") as fileReader:
        data = [line.rstrip('\n') for line in fileReader]
    '''
    rows = len(data)
    columns = len(data[0])

    map = [ [ (initValue) for i in range(columns) ] for j in range(rows) ]    
    
    for y in range(rows):
        for x in range(columns):
            map[y][x] = data[y][x]

    #printMap(map)
    return map
    
