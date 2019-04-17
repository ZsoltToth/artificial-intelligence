'''
Knapsack Solver with Genetic Algorithm

dataset.csv contains some items defined by weight & price.
Problem: Maximize Price while sum of weight is below a limit.


'''
import pandas
import numpy as np

POPULATION_SIZE = 10
WEIGHT_LIMIT = 30

def objectiveFunction(solution, dataset, weightLimit):
    if (solution * dataset['Weight']).sum() > weightLimit:
        return -1
    else:
        return (solution * dataset['Price']).sum()

'''
Probability based selection of elements.
Probability depends on fitness value.
So it is a fitness value based selection. 
'''
def selectionOperator(fitness_values):
    probabilities = fitness_values / fitness_values.sum()
    solutionA = -1
    solutionB = -1
    while solutionA == solutionB:
        solutionA = _probability_based_selection(probabilities)
        solutionB = _probability_based_selection(probabilities)

    return [solutionA, solutionB]

def _probability_based_selection(probabilities):
    probability = np.random.rand()
    for i in range(0,len(probabilities)):
        probability -= probabilities[i]
        if probability < 0.0:
            return i
    return len(probabilities)-1

def crossover(solutionA, solutionB, cutting_ratio = 0.5):
    cutting_index = int(np.floor(len(solutionA) * cutting_ratio))
    return np.append(solutionA[:cutting_index],solutionB[cutting_index:])
#Read Dataset
dataset = pandas.read_csv('data/dataset.csv', delimiter=';')

#Initialize Population
population = [np.random.random_integers(0,1,dataset.shape[0]) for i in range(0,POPULATION_SIZE)]
population = np.array(population)

#Calculate Fitness Values for Indivitual Solutions
fitness_values = [objectiveFunction(sol,dataset, WEIGHT_LIMIT) for sol in population]
fitness_values = np.array(fitness_values)

#Sort population by fitness values
population = population[fitness_values.argsort()]
fitness_values = fitness_values[fitness_values.argsort()]

#Remove Invalid Solution
population = population[fitness_values > 0]
fitness_values = fitness_values[fitness_values > 0]

for i in range(0,fitness_values.shape[0]):
    print(population[i], ' -> ',fitness_values[i])

next_population = np.zeros([POPULATION_SIZE, dataset.shape[0]])

for i in range(0,POPULATION_SIZE):
    selected = selectionOperator(fitness_values)
    next_population[i] = crossover(
        population[selected[0]],
        population[selected[1]])