'''
Simulated Annealing - Traveling Salesman Problem

Graph is fully conected
Cost function is Euclidean Distance

Tasks
1) Generate Cities
2) Define Cost Function
3) Create Neighbor Search Function
'''

#Generate Cities
cities = [
    (0,0), (1,2), (3,2), (4,7), (5,5),
    (7,1), (8,7), (7,7), (9,3), (6,3)
]
import random as rnd
#cities = [(rnd.randint(0,1000),rnd.randint(0,1000)) for i in range(0,100000)]

import math
def distance(c1,c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def objectFunction(path):
    cost = distance(path[0],path[len(path)-1])
    for i in range(1,len(path)):
        cost += distance(path[i-1],path[i])
    return cost


import numpy as np

#Init a random path and calculate its cost
path = np.arange(len(cities))
rnd.shuffle(path)
cost = objectFunction([cities[i] for i in path])

optimum = []
MAX_ITERATION = 10
import time
start_time = time.time()
for iteration in range(MAX_ITERATION):
    optimum.append(objectFunction([cities[i] for i in path]))
    neighbor = path.copy()
    i,j = rnd.sample(range(0,len(cities)),2)
    tmp = neighbor[i]
    neighbor[i] = neighbor[j]
    neighbor[j] = tmp
    del i,j,tmp
    print('itr ', iteration)
    print(path,' ', objectFunction([cities[i] for i in path]))
    print(neighbor,' ',objectFunction([cities[i] for i in neighbor]))
    if(objectFunction([cities[i] for i in path]) > objectFunction([cities[i] for i in neighbor])):
        path = neighbor.copy()
        #print('%d Neighbor was better!' % iteration)
    else:
        deltaEnergy = abs(objectFunction([cities[i] for i in path]) - objectFunction([cities[i] for i in neighbor]))
        temperature = MAX_ITERATION - iteration
        probability = math.e**(-1 * (deltaEnergy / temperature))
        #print(probability)
        if(rnd.random() < probability):
            path = neighbor
            #print('%d Worse neighbor was accepted!' % iteration)

elpased_time = time.time() - start_time
print((
    elpased_time,
    objectFunction([cities[i] for i in path])))

import matplotlib.pyplot as plt

plt.plot(optimum)
plt.show()