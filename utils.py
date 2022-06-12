import os 
import random
import numpy as np
import matplotlib.pyplot as plt 
from collections import deque
from core.evaluation import evaluate
from core.read_tsp import read_tsp_data



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



def initialize():
    sols = [list(range(1, DIMENSION+1)) for _ in range(POOL)]
    evals = [0] * POOL

    for i, sol in enumerate(sols):
        random.shuffle(sol)
        x_sol, y_sol = sol_to_coeff(solution=sol, tsp_data=tsp_data)
        # Evaluation of the solution (Euclidean distance)
        evals[i] = evaluate(x_sol, y_sol)

    # print(evals)
    best_sol = sols[np.argmin(evals)]
    return sols

def sol_to_coeff(solution, tsp_data):
    """
    Gets the solution (number of node) and returns the (x, y) coefficient
    which is capable of plotting / evaluating
    """
    x_sol = [0] * DIMENSION
    y_sol = [0] * DIMENSION
    for i, node in enumerate(solution):
        [x_sol[i], y_sol[i]] = tsp_data[f'{node}']
    return (x_sol, y_sol)

# Tournament selection
def tournament(sols:list,
               config:dict):

    evals = [0] * config['POOL']
    
    for i, sol in enumerate(sols):
        x_sol, y_sol = sol_to_coeff(solution=sol, tsp_data=tsp_data)
        evals[i] = evaluate(x_sol, y_sol)

    new_gen_idx = [evals.index(min(random.sample(evals, config['TSIZE']))) 
                        for _ in range(config['POOL'])]
    # print(new_gen_idx)
    next_gen_sols = [sols[idx] for idx in new_gen_idx]
    return next_gen_sols

# cyclic crossover 
def crossover(sols:list, crate=0.25):
    crossover_rate = 0.25 
    parents = []
    parents_idx = [] 

    for i in range(len(sols)):
        if random.random() < crossover_rate:
            parents.append(sols[i])
            parents_idx.append(i)

    for i, idx in enumerate(parents_idx):
        sols.pop(idx-i)

    if len(parents) % 2 == 1:
        parents.append(sols[random.randint(0, len(sols)-1)])
        parents_idx.append(sols.pop(random.randint(0, len(sols)-1)))

    for i in range(len(parents) // 2):
        o1, o2 = cyclic_crossover(p1=parents[2*i], p2=parents[2*i + 1])
        sols += [o1, o2]

    return sols 

def cyclic_crossover(p1:list, p2:list):
    o1 = p1.copy()
    o2 = p2.copy() 
    idx = 0 
    swap_index = [0]
    while True:
        idx = p1.index(p2[idx])
        if idx in swap_index:
            break
        swap_index.append(idx)
    for idx in swap_index:
        o1[idx] = p2[idx]
        o2[idx] = p1[idx]
    # print(len(swap_index))
    return o1, o2

# Mutation
def mutate(sols : list,
           mut_rate=0.2): 
    for i, sol in enumerate(sols):
        if random.random() < mut_rate:
            sols[i] = displacement(sol)
    return sols 

def pick_random_idx(sol:list):
    idx1 = random.randint(0, len(sol)-1)
    idx2 = random.randint(0, len(sol)-1)
    return idx1, idx2

def displacement(sol:list):

    idx_selection = sorted(pick_random_idx(sol))
    inserted_sol_fragment = sol[idx_selection[0]:idx_selection[1]]

    sol_tmp = [elem for elem in sol if elem not in inserted_sol_fragment]
    idx_selection = random.randint(0, len(sol_tmp)-1)

    for i in range(len(inserted_sol_fragment)):
        sol_tmp.insert(i+idx_selection, inserted_sol_fragment[i])

    sol = sol_tmp 
    return sol 


def print_evals(solution : list, end='\n'):
    evaluations = [0] * POOL
    for i, sol in enumerate(solution):
        x_sol, y_sol = sol_to_coeff(solution=sol, tsp_data=tsp_data)
        evaluations[i] = evaluate(x_sol, y_sol)
    print(evaluations, end=end)

def print_evals_min(solution : list, end='\n'):
    evaluations = [0] * POOL
    for i, sol in enumerate(solution):
        x_sol, y_sol = sol_to_coeff(solution=sol, tsp_data=tsp_data)
        evaluations[i] = evaluate(x_sol, y_sol)
    print(min(evaluations), end=end)
    
def print_evals_avg(solution : list, end='\n'):
    evaluations = [0] * POOL 
    for i, sol in enumerate(solution):
        x_sol, y_sol = sol_to_coeff(solution=sol, tsp_data=tsp_data)
        evaluations[i] = evaluate(x_sol, y_sol)
    avg = sum(evaluations) / len(evaluations)
    print(avg, end=end)

def print_results(solution: list, tts: float, end='\n'):
    evals = get_evals(solution)
    print("Solution: ")
    for e, i in enumerate(solution[evals.index(min(evals))]):
        if (e+1) == len(solution[0]):
            print(f"{i}"); break
        print(f"{i} -> ", end='')
    print("Evaluation of solution:", end=" "); print_evals_min(solution)
    print("Total Time Spent (CPU): ", tts)
    x_sol, y_sol = sol_to_coeff(solution=solution[evals.index(min(evals))], tsp_data=tsp_data) 

def get_evals(sols):
    evals = [0] * len(sols)

    for i, sol in enumerate(sols):
        x_sol, y_sol = sol_to_coeff(solution=sol, tsp_data=tsp_data)
        # Evaluation of the solution (Euclidean distance)
        evals[i] = evaluate(x_sol, y_sol)
    return evals

def plot_result(sols):
    evals = get_evals(sols)
        
    x_sol, y_sol = sol_to_coeff(solution=sols[evals.index(min(evals))], tsp_data=tsp_data)
    
    plt.figure(1) 
    plt.plot(xs + [xs[0]], ys+ [ys[0]], '-o', label='path')
    plt.plot(xs[0], ys[0], 'ro', label='initial_point')
    plt.legend()

    plt.figure(2)
    plt.plot(x_sol + [x_sol[0]], y_sol + [y_sol[0]], '-o', label='path')
    plt.plot(x_sol[0], y_sol[0], 'ro', label='initial_point')
    plt.legend()
    plt.show()