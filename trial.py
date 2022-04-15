from sympy import vectorize
from ir import *
import pandas as pd
import pygad
from random import *
from sklearn.cluster import KMeans
import numpy as np
import pickle
import sys
from numpy.random import randint
from numpy.random import rand

class quantity_control:
    def __init__(self, query):
        self.query = query

    def init_population_generator(self, pop_size):
        i = 0
        q = self.query.split()
        final = []
        bin = [0,1]
        while i<pop_size:
            l=[]
            for j in range(0,len(q)):
                l.append(choice(bin))
            if l in final or l.count(0)==len(l):
                continue
            else:
                final.append(l)
                i+=1
                
        return final

    # queries -> list of lists of binary encoded chromosomes
    def sub_population_generator(self, queries, no_of_clusters):
        q = np.array(queries)
        kmeans = KMeans(n_clusters = no_of_clusters, random_state=0).fit(q)
        print(kmeans.labels_)
        d = {}
        for i in range(len(kmeans.labels_)):
            if kmeans.labels_[i] not in d.keys():
                d[kmeans.labels_[i]] = [queries[i]]
            else:
                d[kmeans.labels_[i]].append(queries[i])
        return d
        # {1: [list of chromo], 2: []...}
    
    def binary_to_string(self, chromosome):
        z = zip(chromosome, self.query.split())
        s = ''
        for i in z:
            if i[0] == 1:
                s+=i[1]+' '
        return s

    def return_queries(self, chromosomes):
        q = []
        for clust in chromosomes:
            l = []
            for chromosome in chromosomes[clust]:
                s = self.binary_to_string(chromosome)
                print(s)
                l.append(s)
            q.append(l)
        return q


class ga:
  def __init__(self,n_iter,n_bits,n_pop,r_cross,r_mut,no_of_relevant_docs,penalty_value,min_range,max_range):
    self.n_iter = n_iter
    self.n_bits = n_bits
    self.n_pop = n_pop
    self.r_cross = r_cross
    self.r_mut = r_mut
    self.no_of_relevant_docs = no_of_relevant_docs
    self.penalty_value = penalty_value
    self.min_range = min_range
    self.max_range = max_range
    
  # objective function
  def onemax(self):
    if(self.no_of_relevant_docs <= self.min_range):
      return (self.no_of_relevant_docs - self.penalty_value)
    elif((self.min_range <= self.no_of_relevant_docs) and (self.no_of_relevant_docs <= self.max_range)):
      return sys.maxsize
    elif(self.no_of_relevant_docs <= self.min_range):
      return (self.no_of_relevant_docs - self.penalty_value)
  
  # tournament selection
  def selection(self,pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
      # check if better (e.g. perform a tournament)
      if scores[ix] < scores[selection_ix]:
        selection_ix = ix
    return pop[selection_ix]
  
  # crossover two parents to create two children
  def crossover(self,p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
      # select crossover point that is not on the end of the string
      pt = randint(1, len(p1)-2)
      # perform crossover
      c1 = p1[:pt] + p2[pt:]
      c2 = p2[:pt] + p1[pt:]
    return [c1, c2]
  
  # mutation operator
  def mutation(self,bitstring, r_mut):
    for i in range(len(bitstring)):
      # check for a mutation
      if rand() < r_mut:
        # flip the bit
        bitstring[i] = 1 - bitstring[i]
  
  # genetic algorithm
  def genetic_algorithm(self, n_bits, n_iter, n_pop, r_cross, r_mut,no_of_relevant_docs,penalty_value,min_range,max_range):
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    # print(pop)
    # keep track of best solution
    best, best_eval = 0, no_of_relevant_docs
    # enumerate generations
    for gen in range(n_iter):
      # evaluate all candidates in the population
      scores = [self.onemax(c) for c in pop]
      # check for new best solution
      for i in range(n_pop):
        # print(scores[i], best_eval)  
        if scores[i] < best_eval:
          best, best_eval = pop[i], scores[i]
          print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
    # select parents
    selected = [self.selection(pop, scores) for _ in range(n_pop)]
    # create the next generation
    children = list()
    for i in range(0, n_pop, 2):
        # get selected parents in pairs
        p1, p2 = selected[i], selected[i+1]
        # crossover and mutation
        for c in self.crossover(p1, p2, r_cross):
          # mutation
          self.mutation(c, r_mut)
          # store for next generation
          children.append(c)
      # replace population
    pop = children
    return [best, best_eval]
 

    

q = query()
query1 = "Alice rabbit hole rabbit follow wonderland alice"
query_class = query_preproc(query1)

quant = quantity_control(query_class.query)

f = quant.init_population_generator(8)
c1 = quant.sub_population_generator(f,3)
print(c1)
#{0: [[1, 0, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0]], 1: [[1, 0, 0, 1, 1, 0, 0], [1, 1, 0, 1, 1, 0, 0]], 2: [[1, 1, 1, 1,
# 1, 0, 1], [1, 1, 0, 1, 1, 0, 1]]}
for population in c1.values():
    

    no_of_relevant_docs = len(population)
    # define the total iterations
    n_iter = 100
    # bits
    n_bits = 20
    # define the population size
    n_pop = 100
    # crossover rate
    r_cross = 0.9
    # mutation rate
    r_mut = 1.0 / float(n_bits)
    #penalty value
    penalty_value = 5
    #min range
    min_range = 1
    #max range
    max_range = 1000

    # perform the genetic algorithm search
    g = ga(n_bits,n_iter,n_pop,r_cross,r_mut,no_of_relevant_docs,penalty_value,min_range,max_range)
    best, score = g.genetic_algorithm(n_bits, n_iter, n_pop, r_cross, r_mut,no_of_relevant_docs,penalty_value,min_range,max_range)
    print('Done!')
    print('f(%s) = %f' % (best, score))

queries = quant.return_queries(cl)
#print(queries)
# [['q1','q2','q3'], [], []]

for i in range(len(queries)):
    output = q.query_similarity_ranked_docs(queries[i])
    print(output)
    # break
    print("Cluster", i)
    for j in output:
        print("No of docs" ,output[j])

        
        

















    # def pop(query, pop_size):
    #     pass
    
    # for i in preprocessQueries:
    #     pop(i, 3)
    
# from numpy.random import randint
# from numpy.random import rand
 

    
    

# class ga:

# """
# Given the following function:
# y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
# where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44
# What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize this function.
# """

# function_inputs = [4,-2,3.5,5,-11,-4.7] # Function inputs.
# desired_output = 44 # Function output.

# def fitness_func(solution, solution_idx):
# output = numpy.sum(solution*function_inputs)
# fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
# return fitness

# num_generations = 100 # Number of generations.
# num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.

# sol_per_pop = 20 # Number of solutions in the population.
# num_genes = len(function_inputs)

# last_fitness = 0
# def on_generation(ga_instance):
# global last_fitness
# print("Generation = {generation}".format(generation=ga_instance.generations_completed))
# print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
# print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
# last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

# ga_instance = pygad.GA(num_generations=num_generations,
#             num_parents_mating=num_parents_mating,
#             sol_per_pop=sol_per_pop,
#             num_genes=num_genes,
#             fitness_func=fitness_func,
#             on_generation=on_generation)

# # Running the GA to optimize the parameters of the function.
# ga_instance.run()

# ga_instance.plot_fitness()

# # Returning the details of the best solution.
# solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
# print("Parameters of the best solution : {solution}".format(solution=solution))
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
# print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# prediction = numpy.sum(numpy.array(function_inputs)*solution)
# print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

# if ga_instance.best_solution_generation != -1:
# print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

# # Saving the GA instance.
# filename = 'genetic' # The filename to which the instance is saved. The name is without extension.
# ga_instance.save(filename=filename)

# # Loading the saved GA instance.
# loaded_ga_instance = pygad.load(filename=filename)
# loaded_ga_instance.plot_fitness()