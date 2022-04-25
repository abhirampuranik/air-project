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
  def __init__(self,population, queries, quant, n_iter,n_bits,n_pop,r_cross,r_mut,penalty_value,min_range,max_range, no_of_clust):
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
    self.no_of_clust = no_of_clust
    
  # objective function per chromosome
  def fitness(self, chromosome, query):
    query = [query]
    op = self.q.query_similarity_ranked_docs(query)
    self.no_of_relevant_docs = op[query[0]]
    if (self.no_of_relevant_docs < self.min_range):
      return (self.no_of_relevant_docs - self.min_range)
    elif (self.no_of_relevant_docs > self.max_range):
      return self.max_range - self.no_of_relevant_docs
    else:
      return self.no_of_relevant_docs
  
  # tournament selection
  def selection(self, scores, k=3):
    # first random selection
    selection_ix = randint(len(self.clust_of_chromosomes[self.current_clust])-1)
    for ix in randint(0, len(self.clust_of_chromosomes[self.current_clust])-1, k-1):
      print("IN SLECTION, IX:" , ix, "SELECTION_IX: ", selection_ix)
      # check if better (e.g. perform a tournament)
      if scores[ix] > scores[selection_ix]:
        selection_ix = ix
    print("selection: ", self.clust_of_chromosomes[self.current_clust][selection_ix])
    return self.clust_of_chromosomes[self.current_clust][selection_ix]
 
  
  # crossover two parents to create two children (SINGLE POINT CROSSOVER)
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
    
    self.clust_of_chromosomes = self.quant.sub_population_generator(self.pop,self.no_of_clust)
    self.clust_of_queries = self.quant.return_queries_wrap(self.clust_of_chromosomes)
    
    best = [0 for i in range(len(self.clust_of_chromosomes.keys()))]
    best_eval = [0 for i in range(len(self.clust_of_chromosomes.keys()))]
    self.scores = {}
    self.selected = {}
    
    for clust in self.clust_of_chromosomes.keys():
      # keep track of best solution
      best[clust], best_eval[clust] = self.clust_of_chromosomes[clust][0], self.fitness(self.clust_of_chromosomes[clust][0], self.clust_of_queries[clust][0])

    # enumerate generations
    for gen in range(n_iter):
      
      children = list()
      for clust in self.clust_of_chromosomes.keys():
        self.current_clust = clust
        print("CLUSTER: ", clust)
        # evaluate all candidates in the population
        self.scores[clust] = [self.fitness(self.clust_of_chromosomes[clust][i], self.clust_of_queries[clust][i]) for i in range(len(self.clust_of_chromosomes[clust]))]
        print("SCORES: ", self.scores)

        # check for new best solution
        for i in range(len(self.clust_of_chromosomes[clust])):
          if self.scores[clust][i] > best_eval[clust]:
            best[clust], best_eval[clust] = self.clust_of_chromosomes[clust][i], self.scores[clust][i]
            print(">%d, new best f(%s) = %.3f" % (gen,  self.clust_of_chromosomes[clust][i], best_eval[clust]))

        
        # select parents
        self.selected[clust] = [self.selection(self.scores[clust]) for _ in range(len(self.clust_of_chromosomes[clust]))]
      
        # create the next generation
        for i in range(0, len(self.clust_of_chromosomes[clust]), 2):
          # get selected parents in pairs
          # if odd number then select selection[0]
          if i+1 < len(self.clust_of_chromosomes[clust]):
            p1, p2 = self.selected[clust][i], self.selected[clust][i+1]
          else:
            p1, p2 = self.selected[clust][i], self.selected[clust][0]
          
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

      self.clust_of_chromosomes = self.quant.sub_population_generator(self.pop,self.no_of_clust)
      self.clust_of_queries = self.quant.return_queries_wrap(self.clust_of_chromosomes)

    final_scores = [self.fitness(self.pop[i], self.queries[i]) for i in range(len(self.pop))]
    

    print('\n\n\n\n\n\nFINAL RESULTS:')
    
    # PRINTING TOP no_of_clust QUERIES FROM ENTIRE POP
    for i in range(self.no_of_clust):
      s = max(final_scores)
      q = final_scores.index(s)
      print(i+1, ':')
      print(self.pop[q], s, self.queries[q])
      self.pop.remove(self.pop[q])
      self.queries.remove(self.queries[q])
      final_scores.remove(s)

    # RETURNING THE BEST FROM EACH CLUST
    return [best, best_eval]

 

    

# MAIN FUNCTION CALLS
query_1 = "Alice rabbit hole rabbit follow wonderland alice sister window fall into hole window outside"
query_class = query_preproc(query_1)
preprocessed_query = query_class.query

quant = quantity_control(preprocessed_query)
init = quant.init_population_generator(15)
# clust_of_chromosomes = quant.sub_population_generator(init,3)
# clust_of_queries = quant.return_queries_wrap(clust_of_chromosomes)
# {0: list of strings, 1: list of strings, 2: list of strings}

n_iter = 10
r_cross = 0.9
penalty_value = 50
min_range = 200
max_range = 350
n_bits = len(init[0])
n_pop = len(init)
r_mut = 1.0 / float(n_bits)
population = init
queries = quant.return_queries(init)

g = ga(population, queries, quant, n_bits,n_iter,n_pop,r_cross,r_mut,penalty_value,min_range,max_range, 3)
best, score = g.genetic_algorithm(n_iter, n_pop, r_cross, r_mut)
print('\n\nDONE!')
# print(best, score)
# print('Best queries: ')
# for i in best:
#   print(quant.binary_to_string(i))
# print('\n\n\n\n\n\n\n')



# for clust in clust_of_queries.keys():
#     print("Cluster Number: ", clust)
#     print("Population size: ", len(clust_of_queries[clust]))
#     n_bits = len(clust_of_queries[clust][0])
#     n_pop = len(clust_of_chromosomes[clust])
#     r_mut = 1.0 / float(n_bits)
#     population = clust_of_chromosomes[clust]
#     queries = clust_of_queries[clust]
    

#     # perform the genetic algorithm search
#     g = ga(population, queries, quant, n_bits,n_iter,n_pop,r_cross,r_mut,penalty_value,min_range,max_range)
#     best, score = g.genetic_algorithm(n_iter, n_pop, r_cross, r_mut)
#     print('\n\nDONE!')
#     print('f(%s) = %f' % (best, score))
#     print('Best query: ', quant.binary_to_string(best),'\n\n\n\n\n')



# FOR TESTING IR
# for i in range(len(queries)):
#     output = q.query_similarity_ranked_docs(queries[i])
#     print(output)
#     # break
#     print("Cluster", i)
#     for j in output:
#         print("No of docs" ,output[j])
