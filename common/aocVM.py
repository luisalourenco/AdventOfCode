from threading import Thread
import functools
from timeit import default_timer as timer
import sys
import os
import copy

class HandheldMachine:

    def __init__(self, instructions):
        self.program = instructions
        self._initialize()
    
    def _initialize(self):
        self.program_counter = 0
        self.accumulator = 0
        self.instructions = self.program

    def reset(self):
        self._initialize()

    def swapOperation(self, pc):
        instruction = self.instructions[pc]
        op = instruction[:3]    
    
        if op == 'jmp':
            self.instructions[pc] = self.instructions[pc].replace('jmp','nop')
        elif op == 'nop':
            self.instructions[pc] = self.instructions[pc].replace('nop','jmp')

    def _executeOperation(self, operation, arg):
        if operation == 'acc':
            self.accumulator += arg
            self.program_counter += 1
        elif operation == 'jmp':
            self.program_counter += arg
        elif operation == 'nop':
            self.program_counter += 1

    def run(self):
        # execute full program
        for instruction in self.instructions:
            operation = instruction[:3]
            arg = int(instruction[3:])   
            
            self._executeOperation(operation, arg)          

            #return (self.acumulator, pc)
    
    def runInstruction(self, pc):
        self.program_counter = pc

        # execute step in program
        instruction = self.instructions[self.program_counter] 
        operation = instruction[:3]
        arg = int(instruction[3:])   
        
        self._executeOperation(operation, arg) 


    def handheldMachine(self, program, pc, accumulator):
        instruction = program[pc] 
        op = instruction[:3]
        arg = int(instruction[3:])   
                
        if op == 'acc':
            accumulator += arg
            pc += 1
        elif op == 'jmp':
            pc += arg
        elif op == 'nop':
            pc += 1

        return (accumulator, pc)