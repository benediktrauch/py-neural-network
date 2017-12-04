import sys


# Get byte size of object
def getSize(obj):
    return sys.getsizeof(obj)/8


# Linspace function like numpy one
def linspace(start, end, stops):
    i = start
    steps = (end-start)/stops
    arr = []
    while i < end:
        # yield i
        arr.append(round(i, 8))
        i += steps
    return arr
