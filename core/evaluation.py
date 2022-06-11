import numpy as np
from collections import deque

def arr_sqrt(x_r, x):
    return (np.array(x_r) - np.array(x)) ** 2 

def evaluate(x: list,
             y: list):

    """
    evaluates the function with respect to (x, y) coordinate using euclidiean distance
    """
    evaluation = 0 
    (dx, dy) = (deque(x), deque(y))
    dx.rotate(-1)
    dy.rotate(-1)
    dx = arr_sqrt(dx, x)
    dy = arr_sqrt(dy, y)
    evaluation = np.sum(np.sqrt(dx + dy))
    return evaluation



if __name__ == "__main__":
    x = [1, 2, 3]
    y = [4, 5, 6]
    
    total_distance = evaluate(x, y) 
    print(total_distance)
    print(4 * 1.414)
