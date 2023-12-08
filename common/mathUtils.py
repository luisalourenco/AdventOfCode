from threading import Thread
import functools
from timeit import default_timer as timer
import sys
import os
from functools import lru_cache

# taken from Wikipedia's Euclidean algorithm pseudocode
def gcd(a,b):
    while a != b:
        if a > b:
            a = a - b
        else:
            b = b - a
    return a

def lcm(a,b):
    return int(a*b/gcd(a,b))


def get_euclidian_distance(p, q):
    """ 
    Return euclidean distance between points p and q
    assuming both to have the same number of dimensions
    """
    # sum of squared difference between coordinates
    s_sq_difference = 0
    for p_i,q_i in zip(p,q):
        s_sq_difference += (p_i - q_i)**2
    
    # take sq root of sum of squared difference
    distance = s_sq_difference**0.5
    return distance

# binary exponentian, useful to compute value**n when either is a big number
def binaryPower(value, n):
    res = 1
    while n > 0:
        if n & 1:
            res = res * value
        value = value * value
        n >>= 1
    
    return res

@lru_cache(maxsize=128)
def binaryPower2(value, n):
    if n == 0:
        return 1

    res = binaryPower2(value, n / 2)
    
    if n % 2:
        return res * res * value
    else:
        return res * res
