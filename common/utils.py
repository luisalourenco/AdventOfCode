# Taken from https://github.com/scout719/adventOfCode/tree/master/common

# -*- coding: utf-8 -*
# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=wildcard-import
# pylint: disable=wrong-import-position
# pylint: disable=consider-using-enumerate-
from threading import Thread
import functools
from timeit import default_timer as timer
import sys
import os
import inspect

""" AUX FUNCTIONS """
class SignalCatchingError(Exception):
    """ Base class for exceptions in this module. """

HEAVY_EXERCISE = "nil (too computationally heavy)"
EXERCISE_TIMEOUT = 120  # secs
DEBUG_MODE = False

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


def setDebugMode(mode):
    globals()['DEBUG_MODE'] = mode

def printd(*args):
    if DEBUG_MODE:
        print(args)


def printGridsASCII(grid, printChar):
    rows = len(grid)
    columns = len(grid[0])

    for r in range(rows):
        row = ""
        for c in range(columns):
            if grid[r][c] == printChar:
                row += WHITE_SQUARE
            else:
                row += " "
        print(row)

def clear():
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')

def timeout(seconds_before_timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            res = [SignalCatchingError('function [%s] timeout [%s seconds] exceeded!' %
                                       (func.__name__, seconds_before_timeout))]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except BaseException as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(seconds_before_timeout)
            except Exception as e:
                print('error starting thread')
                raise e
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

def execute_day(_globals, year, day, part, new_timeout):
    #print("Executing year {0} day {1}, part {2}".format(year, day, part))
    func_name = "day{0}_{1}".format(day, part)
    if func_name in _globals:
        start = timer()
        try:
            result = timeout(seconds_before_timeout=new_timeout)(
                _globals[func_name])(read_input(year, day))
        except SignalCatchingError:
            result = HEAVY_EXERCISE
        end = timer()
        print("Day {0}, part {1}: {2} ({3:.3f} secs)".format(
            day, part, result, end - start))

def read_input(year, day):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    with open("{0}/../{1}/input/day{2}".format(file_dir, year, day), "r") as fileReader:
        return [line.rstrip('\n') for line in fileReader]

def AssertExpectedResult(expected, result, part = 0):
    part = inspect.stack()[1].function.split("_")[1]    
    print("( Part",part,") Correct result:", expected == result)

def ints(data):
    return [int(n) for n in data]

def main(argv_, globals_, year, new_timeout = EXERCISE_TIMEOUT):
    start_day = None
    if len(argv_) > 1:
        try:
            if len(argv_) > 2:
                raise ValueError
            start_day = int(argv_[1])
        except ValueError:
            print("Usage: aoc.py [<day>]")
            sys.exit(1)
    initial_day = 1
    end_day = 25
    if start_day is not None:
        initial_day = start_day
        end_day = start_day

    for day in range(initial_day, end_day + 1):
        execute_day(globals_, year, day, 1, new_timeout)
        execute_day(globals_, year, day, 2, new_timeout)
