#!/usr/bin/env python3
import random
import math
import numpy.random as npr # need to pip install numpy 

###################################################
# Parameters:
M = 20           # mating pool size
G = 20           # max number of generations
p_c = .99        # probability of crossover
p_r = 0.01       # probability of reproduction
p_m = 0.01       # probability of mutation
###################################################

class organism():
    def __init__(self, genome, k, answer):
        self.genome = genome  # a list of 0's and 1's
        self.size = k
        self.fitness_score = self.fitness(answer)  # fitness score

    def crossOver(self, partner, answer):
        '''
        This function takes in two parent organisms (self, partner)
        and the size of their genome (k),
        crosses over their genome at one-point (this could be generalized to
        n-point cross-over if we want), and returns two children
        '''
        q = random.uniform(0, 1)
        if p_c < q:
            self.fitness_score = self.fitness(answer)
            partner.fitness_score = partner.fitness(answer)
            return self, partner
        r = random.randrange(self.size) 
        m = self.genome
        f = partner.genome
        genome1 = m[0:r] + f[r:]
        genome2 = f[0:r] + m[r:]
        child1 = organism(genome1, self.size, answer)
        child2 = organism(genome2, self.size, answer)

        return child1, child2

    def mutate(self, answer):
        '''
        This function takes in an organism and the size. For each element in
        genome, determine if mutation.
        '''
        for i in range(self.size):
            if p_m > random.uniform(0, 1):
                if self.genome[i] == 0:
                    self.genome[i] = 1
                else:
                    self.genome[i] = 0

        self.fitness_score = self.fitness(answer)

    def fitness(self, answer):
        '''
        This function assigns a value determining how fit an organism is
        based on its genome.
        '''
        fitness_score = self.size 
        for i in range(self.size):
            if self.genome[i] == answer[i]: # index of organism's genome is correct
                fitness_score += 1
            else:
                fitness_score -= 1
        return fitness_score

def create_mating_pool(organism_pool):
    '''
    This function takes in a list of all the organisms and returns a list of
    organisms to be selected for reproduction. We will use roulette wheel selection.
    Solution was taken from the following:
    https://stackoverflow.com/questions/10324015/fitness-proportionate-selection-roulette-wheel-selection-in-python
    '''
    new_pool = []
    total_fitness = sum(org.fitness_score for org in organism_pool)
    selection_probs = [org.fitness_score/total_fitness for org in organism_pool]
    for _ in range(M): # roulette wheel selection
        new_pool.append(organism_pool[npr.choice(M, p=selection_probs)])
    return new_pool


def generate_initial_population(k, answer):
    '''
    Randomly generate a popluation of M organisms
    50% chance of index being 0 and 50% chance of being 1
    '''
    organism_pool = []
    for i in range(M):
        genome = []
        for j in range(k):
            if random.uniform(0, 1) < 0.5:
                genome.append(0)
            else:
                genome.append(1)
        
        organism_pool.append(organism(genome, k, answer))
    return organism_pool

def print_organism_pool(organism_pool, generation_num):
    print("\nGeneration {}:".format(generation_num))
    print("-" * 6 * (organism_pool[0].size + 1))
    for i in range(M):
        print("Organism {}: {}; Total Fitness: {}".format(i+1, organism_pool[i].genome, 
            organism_pool[i].fitness_score))
    print("-" * 6 * (organism_pool[0].size + 1))
    return

def linear_normalization(organism_pool):
    '''
    This function assigns new fitness values according to the rank
    of each individual organism's raw fitness_score. This helps reduce
    stagnation and premature convergence
    '''
    organism_pool.sort(key=lambda org: org.fitness_score) # sort pool by raw fitness
    for i in range(0, len(organism_pool)):
        organism_pool[i].fitness_score = i+1
    return

def main():
    first_word = input("Please enter a squence of characters to compare: ")
    second_word = input("Please enter another sequence of characters to compare: ")

    # Hard coding words for testing
    #first_word = "president"
    #second_word = "providence"

    k = min(len(first_word), len(second_word)) # length of shorter
    answer = [] # correct input values
    for i in range(k):
        if first_word[i] == second_word[i]:
            answer.append(1)
        else:
            answer.append(0)

    best_organism = organism([0]*k, k, answer) # current best organism based on fitness
    organism_pool = generate_initial_population(k, answer)
    print_organism_pool(organism_pool, 0)
    generation = 0 

    for i in range(G): # loop until max generation is met
        linear_normalization(organism_pool) # convert raw fitness to rank
        mating_pool = create_mating_pool(organism_pool)
        organism_pool = []
        for j in range(0, M-round(p_r*M), 2): # crossover then mutation
            child1, child2 = mating_pool[j].crossOver(mating_pool[j+1], answer)
            child1.mutate(answer)
            child2.mutate(answer)
            organism_pool.append(child1)
            if len(organism_pool)  == (M-round(p_r*M)):
                break
            organism_pool.append(child2)

        # best_organism's fitness still based on linear_normalization. need to 
        # convert back to raw fitness score
        best_organism.fitness_score = best_organism.fitness(answer)
        for org in organism_pool: # save the best organism
            if org.fitness_score > best_organism.fitness_score:
                best_organism = org
                generation = i+1

        # saved the best organisms based on p_r and append to pool
        for j in range(M-1, M-round(p_r*M)-1, -1): 
            organism_pool.append(best_organism)

        print_organism_pool(organism_pool, i+1)    

    print("The correct sequence and fitness score is: {}; {}".format(
        answer, 2*k))

    print("The organism with the best fitness score is: {}; {} occured on generation {}".format(best_organism.genome, best_organism.fitness_score, generation))

if __name__ == "__main__":
    main()
