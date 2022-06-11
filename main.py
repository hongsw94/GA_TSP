import os 
import random
import numpy as np
import matplotlib.pyplot as plt 
from collections import deque
from core.evaluation import evaluate
from core.read_tsp import read_tsp_data
from utils import *



fdir = './problems'
fname = 'a280.tsp'
fpath = os.path.join(fdir, fname)
config=dict()
POOL = 100   # Generation Pool Size
TSIZE = 20   # Tournament Size
crossover_rate = 0.25
mutation_rate = 0.2

# open tsp file and then factorize the information into data that are useful
with open(fpath, 'r') as f:
    NAME = f.readline().split()[-1]
    COMMENT = ' '.join(f.readline().split()[2:])
    FTYPE = f.readline().split()[-1]
    DIMENSION = int(f.readline().split()[-1])
    EDGE_WEIGHT_TYPE = f.readline().split()[-1]
    f.readline()
    infos = f.readlines()[:-1]

infos = [info.split() for info in infos]
tsp_data = dict()
# get coordinate
for (node, x, y) in infos:
    tsp_data[node] = [int(x), int(y)]

xs = [d[0] for d in list(tsp_data.values())]
ys = [d[1] for d in list(tsp_data.values())]


config = dict()
config['DIMENSION'] = DIMENSION
config['POOL'] = POOL
config['TSIZE'] = TSIZE



sols = initialize() 

# main loop
for i in range(10000):
    sols = tournament(sols, config)
    sols = crossover(sols, crate=crossover_rate)
    sols = mutate(sols, mut_rate=mutation_rate)
    if (i+1) % 10 == 0 :
        print(f"step {i+1} : ", end=""); print_evals_min(sols)

plot_result(sols=sols)