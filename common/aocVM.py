from threading import Thread
import functools
from timeit import default_timer as timer
import sys
import os
import copy
from common.utils import *
import math


class HandheldMachine:

    def __init__(self, instructions):
        # full program
        self.__program = instructions
        self._initialize()
    
    def _initialize(self):
        self.__program_counter = 0
        self.__accumulator = 0
        # current set of instructions loaded into machine
        self.__instructions = copy.deepcopy(self.__program)

    def reset(self):
        self._initialize()

    @property
    def program_counter(self):
        return self.__program_counter

    @property
    def accumulator(self):
        return self.__accumulator

    def swapOperation(self, pc):
        instruction = self.__instructions[pc]
        op = instruction[:3]    
    
        if op == 'jmp':
            self.__instructions[pc] = self.__instructions[pc].replace('jmp','nop')
        elif op == 'nop':
            self.__instructions[pc] = self.__instructions[pc].replace('nop','jmp')

    def __executeOperation(self, operation, arg):
        if operation == 'acc':
            self.__accumulator += arg
            self.__program_counter += 1
        elif operation == 'jmp':
            self.__program_counter += arg
        elif operation == 'nop':
            self.__program_counter += 1
        else: NotImplementedError

    # execute full loaded program
    def run(self):
        self.__program_counter = 0
        for _ in range(len(self.__instructions)):             
            self.runStep()         

    # execute a step in loaded program
    def runStep(self):    
        instruction = self.__instructions[self.__program_counter] 
        operation = instruction[:3]
        arg = int(instruction[3:])   
        
        self.__executeOperation(operation, arg) 
    
    # execute a step from a given program counter
    def __runAtProgramCounter(self, pc):
        self.__program_counter = pc        
        self.runStep() 


class ALUVM:

    def __init__(self, instructions):
        # full program
        self.__program = instructions
        self._initialize()
    
    def _initialize(self):
        self.__program_counter = 0
        self.__w = 0
        self.__x = 0
        self.__y = 0
        self.__z = 0
        # current set of instructions loaded into machine
        self.__instructions = copy.deepcopy(self.__program)

    def reset(self):
        self._initialize()

    @property
    def program_counter(self):
        return self.__program_counter

    @property
    def w_var(self):
        return self.__w
    
    @property
    def x_var(self):
        return self.__x
    
    @property
    def y_var(self):
        return self.__y

    def z_var(self):
        return self.__z

    def __get_variable(self, var_name):

        if var_name == 'x':                
            return self.__x
        elif var_name == 'y':
            return self.__y 
        elif var_name == 'w':
            return self.__w 
        elif var_name == 'z':
            return self.__z
        else:
            return int(var_name)

    def __set_variable(self, var_name, value):

        if var_name == 'x':                
            self.__x = value
        elif var_name == 'y':
            self.__y = value
        elif var_name == 'w':
            self.__w = value 
        elif var_name == 'z':
            self.__z = value

    def __executeOperation(self, operation, arg, input_data):
        print("executing operation [", operation, "] with arg",arg,"and input", input_data)
        if operation == 'inp':
            if len(input_data) > 0:
                value = input_data.pop()
                print("read from input data:", value)
            else:
                value = input("input:")
            self.__set_variable(arg[0], value)
            self.__program_counter += 1
        elif operation == 'add':
            print(arg)
            a = self.__get_variable(arg[0])
            b = self.__get_variable(arg[1])
            c = int(a) + int(b)
            self.__set_variable(arg[0], c)

            self.__program_counter += 1
        elif operation == 'mul':
            a = int(self.__get_variable(arg[0]))
            b = int(self.__get_variable(arg[1]))
            c = int(a) * int(b)
            self.__set_variable(arg[0], c)
            self.__program_counter += 1
        elif operation == 'div':
            a = int(self.__get_variable(arg[0]))
            b = int(self.__get_variable(arg[1]))
            if b != 0:
                c = math.floor(a / b)
                self.__set_variable(arg[0], c)
            self.__program_counter += 1
        elif operation == 'mod':
            a = int(self.__get_variable(arg[0]))
            b = int(self.__get_variable(arg[1]))
            if not (a < 0 or b <= 0):
                c = a % b
                self.__set_variable(arg[0], c)
            self.__program_counter += 1
        elif operation == 'eql':
            a = int(self.__get_variable(arg[0]))
            b = int(self.__get_variable(arg[1]))
            c = 1 if a == b else 0
            self.__set_variable(arg[0], c)
            self.__program_counter += 1
        
        else: NotImplementedError

    # execute full loaded program
    def run(self, input_data = []):
        self.__program_counter = 0
        self.get_state()
        
        for _ in range(len(self.__instructions)):  
            self.runStep(input_data)  
            self.get_state()       
    
    def get_state(self):
        print("current state:")
        print("w:",self.__w)
        print("x:",self.__x)
        print("y:",self.__y)
        print("z:",self.__z)
        print()
        

    # execute a step in loaded program
    def runStep(self, input_data):    
        instruction = self.__instructions[self.__program_counter] 
        operation = instruction[:3]
        arg = instruction[3:].strip().split(" ")
        
        self.__executeOperation(operation, arg, input_data) 
    
    # execute a step from a given program counter
    def __runAtProgramCounter(self, pc, input_data):
        self.__program_counter = pc        
        self.runStep(input_data)