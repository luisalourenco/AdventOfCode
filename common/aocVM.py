from threading import Thread
import functools
from timeit import default_timer as timer
import sys
import os
import copy

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