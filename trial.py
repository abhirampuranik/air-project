from sympy import vectorize
from ir import *
import pandas as pd
import pygad
from random import *
from sklearn.cluster import KMeans
import numpy as np
import sys
from numpy.random import randint
from numpy.random import rand

class quantity_control:
    def __init__(self, query=''):
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

    def return_queries(self, pop):
      l = []
      for chromo in pop:
        s = self.binary_to_string(chromo)
        l.append(s)
      return l


    def return_queries_wrap(self, chromosomes):
        q = {}
        for clust in chromosomes:
            l = self.return_queries(chromosomes[clust])
            q[clust] = l
        return q


class ga:
  def __init__(self,population, queries, quant, n_iter,n_bits,n_pop,r_cross,r_mut,penalty_value,min_range,max_range):
    self.pop = population
    self.queries = queries
    self.n_iter = n_iter
    self.n_bits = n_bits
    self.n_pop = n_pop
    self.r_cross = r_cross
    self.r_mut = r_mut
    self.penalty_value = penalty_value
    self.min_range = min_range
    self.max_range = max_range
    self.q = query()
    self.quant = quant
    
  # objective function
  def onemax(self, chromosome, query):
    query = [query]
    op = self.q.query_similarity_ranked_docs(query)
    self.no_of_relevant_docs = op[query[0]]
    print("no of relevant docs: ", self.no_of_relevant_docs)
    if (self.no_of_relevant_docs < self.min_range):
      print("Returning 1: ", self.no_of_relevant_docs - self.min_range)
      return (self.no_of_relevant_docs - self.min_range)
    elif (self.no_of_relevant_docs > self.max_range):
      print("Returning 2: ", self.max_range - self.no_of_relevant_docs)
      return self.max_range - self.no_of_relevant_docs
    else:
      return self.no_of_relevant_docs
  
  # tournament selection
  def selection(self, scores, k=3):
    # first random selection
    selection_ix = randint(len(self.pop)-1)
    for ix in randint(0, len(self.pop)-1, k-1):
      print("IN SLECTION, IX:" , ix, "SELECTION_IX: ", selection_ix)
      # check if better (e.g. perform a tournament)
      if scores[ix] > scores[selection_ix]:
        selection_ix = ix
    print("selection: ", self.pop[selection_ix])
    return self.pop[selection_ix]
 
  
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
  def genetic_algorithm(self, n_iter, n_pop, r_cross, r_mut):
    
    print("Init generation: ", self.pop)
    
    # keep track of best solution
    best, best_eval = self.pop[0], self.onemax(self.pop[0], self.queries[0])

    # enumerate generations
    for gen in range(n_iter):
      # evaluate all candidates in the population
      scores = [self.onemax(self.pop[i], self.queries[i]) for i in range(n_pop)]
      print("SCORES: ", scores)
      # check for new best solution
      for i in range(n_pop):
        if scores[i] > best_eval:
          best, best_eval = self.pop[i], scores[i]
          print(">%d, new best f(%s) = %.3f" % (gen,  self.pop[i], best_eval))

      # select parents
      selected = [self.selection(scores) for _ in range(n_pop)]
      # create the next generation
      children = list()
      for i in range(0, n_pop, 2):
          # get selected parents in pairs
          # if odd number then select selection[0]
          if i+1 < n_pop:
            p1, p2 = selected[i], selected[i+1]
          else:
            p1, p2 = selected[i], selected[0]
          # crossover and mutation
          for c in self.crossover(p1, p2, r_cross):
            # mutation
            self.mutation(c, r_mut)
            # store for next generation
            children.append(c)
        # replace population
      self.pop = children
      print("New generation: ", self.pop)
      self.queries = self.quant.return_queries(self.pop)
      print("New queries: ", self.queries)
    return [best, best_eval]
 

    

# MAIN FUNCTION CALLS
query_1 = "Alice rabbit hole rabbit follow wonderland alice sister window fall into hole window outside"
query_class = query_preproc(query_1)
preprocessed_query = query_class.query

quant = quantity_control(preprocessed_query)
init = quant.init_population_generator(15)
clust_of_chromosomes = quant.sub_population_generator(init,3)
clust_of_queries = quant.return_queries_wrap(clust_of_chromosomes)

for clust in clust_of_queries.keys():
    print("Cluster Number: ", clust)
    print("Population size: ", len(clust_of_queries[clust]))
    population = clust_of_chromosomes[clust]
    queries = clust_of_queries[clust]
    # define the total iterations
    n_iter = 10
    # bits
    n_bits = len(clust_of_queries[clust][0])
    # define the population size
    n_pop = len(clust_of_chromosomes[clust])
    # crossover rate
    r_cross = 0.9
    # mutation rate
    r_mut = 1.0 / float(n_bits)
    #penalty value
    penalty_value = 50
    #min range
    min_range = 200
    #max range
    max_range = 300

    # perform the genetic algorithm search
    g = ga(population, queries, quant, n_bits,n_iter,n_pop,r_cross,r_mut,penalty_value,min_range,max_range)
    best, score = g.genetic_algorithm(n_iter, n_pop, r_cross, r_mut)
    print('\n\nDONE!')
    print('f(%s) = %f' % (best, score))
    print('Best query: ', quant.binary_to_string(best),'\n\n\n\n\n')



# for i in range(len(queries)):
#     output = q.query_similarity_ranked_docs(queries[i])
#     print(output)
#     # break
#     print("Cluster", i)
#     for j in output:
#         print("No of docs" ,output[j])
