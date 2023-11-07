import pandas as pd
import numpy as np
import random

df = pd.read_csv("data_ubon.csv")
df

df = df.iloc[:]
df

pop = df['population'].to_numpy()
variable_cost = df['variable cost'].to_numpy()
Xs = df['X_longitude'].to_numpy()
Ys = df['Y_latitude'].to_numpy()

import math
def distance(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx*dx + dy*dy)*100

s = []
for i in range(len(Xs)):
    sr = []
    for j in range(len(Xs)):
        d = distance(Xs[i], Ys[i], Xs[j], Ys[j])
        sr.append(d)
    s.append(sr)

s = np.array(s)
s.shape, s

len(s)

np.where(pop > 2500)

base_cost=100000
T1=8
capacity=2500

countData=len(s)
lookup_sort = {}
for j in range(countData):
    data={}
    for i in range(countData):
        data[i]= s[int(j), i] #(s[4,'1']) #row:column
    sort_distance = sorted(data.items(), key=lambda x: x[1], reverse=False) #False=ASC, True=DESC4
    lookup_sort[j] = sort_distance

len(sort_distance), sort_distance[15]

def getBitrate(distance):
    bitrate=0
    if(distance <=2):
        bitrate=4
    elif(distance <=4):
        bitrate=3
    elif(distance <=6):
        bitrate=2
    elif(distance <=8):
        bitrate=1
    return bitrate

def getPopulaton(getway, closeset=None):
  endvice={"ids":[]}

  collect_capacity=0
  collect_cost=0
  collect_bitrate=0
  sort_distance = lookup_sort[getway]
  countDevice=len(sort_distance)
  for j in range(countDevice) :
      cid = sort_distance[j][0]
      dis = sort_distance[j][1]
      if closeset != None and cid in closeset:
          continue

      if dis > 8:
          break

      check_capcity=collect_capacity+pop[cid]
      if(check_capcity <= capacity):
          endvice['ids'].append( sort_distance[j][0])
          collect_capacity += pop[sort_distance[j][0]]
          collect_bitrate += getBitrate(sort_distance[j][1])
      else:
          break

  total_endvice=len(endvice['ids'])
  collect_bitrate =collect_bitrate/total_endvice
  collect_bitrate = "%.3f" % round(collect_bitrate, 3)
  endvice["totalVillage"]=total_endvice
  endvice["bitrate"]=collect_bitrate
  endvice["gateway"]=getway
  endvice["capacity"]=collect_capacity
  endvice["cost"]=variable_cost[getway]+base_cost
  return endvice

getPopulaton(0, set([2, 3, 4]))

closeset = set([])
closeset = closeset.union(getPopulaton(52)['ids'])
closeset

np.random.seed(0)
x = np.random.rand(len(Xs))
idx= np.argsort(x)

def cost(idx):
    closeset = set([])
    solution = []
    i = 0
    while len(closeset) < len(idx):
        center = idx[i]
        if center in closeset:
            i = i+1
            continue
        #print(center)
        cluster = getPopulaton(center, closeset)
        solution.append(cluster)

        #print(cluster)
        closeset = closeset.union(cluster['ids'])
        i = i+1
    total_cost = 0
    total_birate = 0
    for cluster in solution:
        #print(cluster)
        total_cost += cluster['cost']
        total_birate += float(cluster['bitrate'])
    return total_cost, total_birate/len(solution), solution

tcost, bitrate, sol = cost(idx)
len(sol)

pip install -U pymoo

from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.soo.nonconvex.de import DE

import numpy as np
from pymoo.core.problem import ElementwiseProblem
import math

class LoraWanProblem(ElementwiseProblem):

    def __init__(self, N):
        self.N = N
        super().__init__(n_var=self.N, n_obj=1, n_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        idx= np.argsort(x)
        tcost, bitrate, solution =  cost(idx)

        #f = tcost/bitrate
        w = 0.9
        f = w*tcost/(base_cost*150) + (1-w)*2.5/bitrate

        out["hash"] = hash(str(idx))
        out["F"] = f
        out["pheno"] = {"values": solution}

from numpy.core.memmap import uint8
import numpy as np

class MyDE:
    def __init__(self, problem, pop_size,  variant, max_iter, CR=0.3 ):
        self.pop_size = pop_size
        self.variant = variant
        self.CR = CR
        self.problem = problem
        self.max_iter = max_iter
        self.Xs = None
        self.fitnessXs = None
        self.bestX = None
        self.bestFitness = -1000
        self.F = 0.6
        self.data = np.zeros((self.pop_size*max_iter, self.problem.n_var))
        self.out = {}
        self.initialize()

    def initialize(self):
        self.Xs = np.random.rand( self.pop_size, self.problem.n_var)
        self.fitnessXs = np.zeros((self.pop_size,))
        self.cal_fitness()

    def cal_fitness(self):
        for i in range(self.pop_size):
            self.problem._evaluate(self.Xs[i], self.out)
            self.fitnessXs[i] = self.out['F']
        max_index = np.argmax(self.fitnessXs)
        self.bestX = np.copy(self.Xs[max_index])
        self.bestFitness = self.fitnessXs[max_index]

    def mutation(self):
        indexs = np.arange(0, self.pop_size, 1).astype(dtype=np.uint8)
        index_r1s = list(np.copy(indexs))
        index_r2s = list(np.copy(indexs))
        index_r3s = list(np.copy(indexs))
        np.random.shuffle(index_r1s)
        np.random.shuffle(index_r2s)
        np.random.shuffle(index_r3s)

        self.Vs = self.Xs[index_r1s] + self.F*(self.Xs[index_r2s] - self.Xs[index_r3s])

        vmin = np.min(self.Vs)
        vmax = np.max(self.Vs)
        if vmin != vmax:
            self.Vs = (self.Vs-vmin)/(vmax-vmin)


    def recombination(self):
        cross_select = np.random.choice(a=[0, 1], size=self.Xs.shape, p=[self.CR, 1-self.CR])
        self.Us = (1-cross_select)*self.Xs + cross_select*self.Vs


    def selection(self):
        fitnessUs = np.zeros((self.pop_size,))
        select_mask = np.zeros((self.pop_size,))
        for i in range(self.pop_size):
            self.problem._evaluate(self.Us[i], self.out)
            fitnessUs[i] = self.out['F']
            if fitnessUs[i] > self.fitnessXs[i]:
                self.fitnessXs[i] = fitnessUs[i]
                self.Xs[i] = np.copy(self.Us[i])
                if fitnessUs[i] > self.bestFitness:
                    self.bestFitness = fitnessUs[i]
                    self.bestX = np.copy(self.Us[i])


    def iterate(self):
        for it in range(self.max_iter):
            self.mutation()
            self.recombination()
            self.selection()
            print(it, round(np.mean(self.fitnessXs), 4), round(self.bestFitness, 4), self.bestX[4:])
            self.data[it*self.pop_size:self.pop_size*(it+1)] = np.copy(self.Xs)

from numpy.core.memmap import uint8
import numpy as np

class AdaptiveWeight:
    def __init__(self, N, Max_iter, F, K):
        self.Ns = np.zeros(( Max_iter+1, N))
        self.As = np.zeros(( Max_iter+1, N))
        self.Is = np.zeros(( Max_iter+1, N))
        self.Iter = 0
        self.Max_iter = Max_iter
        self.N_track = N
        self.pobWeights = np.ones((N,))/N
        self.weights = np.ones((N,))
        self.Ns[0] = np.ones((N,))
        self.As[0] = np.ones((N,))
        self.Is[0] = np.ones((N,))
        self.F = F
        self.K = K
        self.listIndexs = list(np.arange(N, dtype=np.uint8))

    def random_tracks(self, size):
        tracksIds = np.random.choice(self.listIndexs, size, p=self.pobWeights)
        return tracksIds

    def update(self, tracks, objects, bestI):
        self.Iter += 1
        self.Is[self.Iter, bestI] = 1
        for i in range(self.N_track):
            indexs = np.where(tracks==i)[0]
            self.Ns[self.Iter, i] = len(indexs)
            self.As[self.Iter, i] = np.mean(objects[indexs])
            if len(indexs) == 0:
                self.As[self.Iter, i] = 0
            Ii = np.sum(self.Is[:self.Iter, i])

            new_weight = (self.F*self.Ns[self.Iter, i] +
                              (1-self.F)*self.As[self.Iter, i] + self.K*Ii)
            self.weights[i] =  max(0.3*self.weights[i] + 0.7*new_weight, 0)
        total = np.sum(self.weights)
        #print(self.weights, total)
        for i in range(self.N_track):
            self.pobWeights[i] = self.weights[i]/total

class MDE:
    def __init__(self, problem, pop_size,  variant, max_iter, CR=0.3, CRT=0.2, F=0.6):
        self.pop_size = pop_size
        self.variant = variant
        self.CR = CR
        self.problem = problem
        self.max_iter = max_iter
        self.Xs = None
        self.fitnessXs = None
        self.bestX = None
        self.best2X = None
        self.CRT = CRT
        self.bestFitness = 0
        self.F = F
        self.data = np.zeros((3*max_iter, self.problem.n_var))
        self.out = {}
        self.initialize()
        self.N_mutations = 5
        self.adaptiveWeight = AdaptiveWeight(self.N_mutations, max_iter, 0.05, 1)

    def initialize(self):
        self.Xs = np.random.rand( self.pop_size, self.problem.n_var)
        self.fitnessXs = np.zeros((self.pop_size,))
        self.cal_fitness()
        max_index = np.argmax(self.fitnessXs)
        self.bestX = np.copy(self.Xs[max_index])
        self.best2X = np.copy(self.bestX)
        self.bestFitness = self.fitnessXs[max_index]

    def cal_fitness(self):
        for i in range(self.pop_size):
            self.problem._evaluate(self.Xs[i], self.out)
            self.fitnessXs[i] = self.out['F']

    def DE_bin1_mutation(self, xs):
        N = len(xs)
        indexs = np.arange(0, self.pop_size, 1).astype(dtype=np.uint8)
        index_r1s = list(np.copy(indexs))
        index_r2s = list(np.copy(indexs))
        index_r3s = list(np.copy(indexs))
        np.random.shuffle(index_r1s)
        np.random.shuffle(index_r2s)
        np.random.shuffle(index_r3s)
        Vs = self.Xs[index_r1s[:N]] + self.F*(self.Xs[index_r2s[:N]] - self.Xs[index_r3s[:N]])

        vmin = np.min(Vs)
        vmax = np.max(Vs)
        if vmin != vmax:
            Vs = (Vs-vmin)/(vmax-vmin)
        return Vs

    def DE_best_bin1_mutation(self, xs):
        N = len(xs)
        indexs = np.arange(0, self.pop_size, 1).astype(dtype=np.uint8)
        index_r1s = list(np.copy(indexs))
        index_r2s = list(np.copy(indexs))
        index_r3s = list(np.copy(indexs))
        np.random.shuffle(index_r1s)
        np.random.shuffle(index_r2s)
        np.random.shuffle(index_r3s)
        Vs = self.bestX + (self.F*(self.best2X - self.Xs[index_r1s[:N]]) +
                         self.F*(self.Xs[index_r2s[:N]] - self.Xs[index_r3s[:N]]))

        vmin = np.min(Vs)
        vmax = np.max(Vs)
        if vmin != vmax:
            Vs = (Vs-vmin)/(vmax-vmin)
        return Vs

    def DE_random_transit(self, xs):
        N = len(xs)
        rs = np.random.rand(xs.shape[0], xs.shape[1])
        cross_select = np.random.choice(a=[0, 1], size=xs.shape, p=[self.CRT, 1-self.CRT])
        Vs = (1-cross_select)*xs + cross_select*rs
        return Vs

    def DE_best_transit(self, xs):
        N = len(xs)
        rs = np.random.rand(xs.shape[0], xs.shape[1])
        cross_select = np.random.choice(a=[0, 1], size=xs.shape, p=[self.CRT, 1-self.CRT])
        Vs = (1-cross_select)*xs + cross_select*self.bestX
        return Vs

    def DE_scale(self, xs):
        N = len(xs)
        rs = np.random.rand(xs.shape[0], xs.shape[1])
        cross_select = np.random.choice(a=[0, 1], size=xs.shape, p=[self.CRT, 1-self.CRT])
        Vs =  ((1 - cross_select)*rs)*xs +   cross_select*xs
        return Vs

    def mutation(self):
        self.Vs = np.copy(self.Xs)
        for i in range(self.N_mutations):
            indexI = list(np.where(self.currentTracks == i)[0])
            if len(indexI) == 0:
                continue
            if i == 0:
                self.Vs[indexI] = self.DE_bin1_mutation(self.Xs[indexI])
            elif i==1:
                self.Vs[indexI] = self.DE_best_bin1_mutation(self.Xs[indexI])
            elif i==2:
                self.Vs[indexI] = self.DE_random_transit(self.Xs[indexI])
            elif i==3:
                self.Vs[indexI] = self.DE_random_transit(self.Xs[indexI])
            elif i==4:
                self.Vs[indexI] = self.DE_scale(self.Xs[indexI])

    def recombination(self):
        cross_select = np.random.choice(a=[0, 1], size=self.Xs.shape, p=[self.CR, 1-self.CR])
        self.Us = (1-cross_select)*self.Xs + cross_select*self.Vs

    def accept_all(self):
        fitnessUs = np.zeros((self.pop_size,))
        select_mask = np.zeros((self.pop_size,))
        for i in range(self.pop_size):
            self.problem._evaluate(self.Us[i], self.out)
            fitnessUs[i] = self.out['F']
            self.fitnessXs[i] = fitnessUs[i]
            self.Xs[i] = np.copy(self.Us[i])
            if fitnessUs[i] > self.bestFitness:
                self.bestFitness = fitnessUs[i]
                self.best2X = np.copy(self.bestX)
                self.bestX = np.copy(self.Us[i])

    def selection(self):
        fitnessUs = np.zeros((self.pop_size,))
        select_mask = np.zeros((self.pop_size,))
        for i in range(self.pop_size):
            self.problem._evaluate(self.Us[i], self.out)
            fitnessUs[i] = self.out['F']
            delta = fitnessUs[i] - self.fitnessXs[i]
            if (delta < 0 or
                np.random.rand() + 5 < np.exp(-delta*100) or
                np.random.rand()  < 0.00001):
                self.fitnessXs[i] = fitnessUs[i]
                self.Xs[i] = np.copy(self.Us[i])
                if fitnessUs[i] < self.bestFitness:
                    self.bestFitness = fitnessUs[i]
                    self.best2X = np.copy(self.bestX)
                    self.bestX = np.copy(self.Us[i])

    def iterate(self):
        for it in range(self.max_iter):
            self.currentTracks = self.adaptiveWeight.random_tracks(self.pop_size)
            self.mutation()
            arg_max = np.argmax(self.fitnessXs)
            #print(self.adaptiveWeight.pobWeights)
            self.adaptiveWeight.update(self.currentTracks, self.fitnessXs, self.currentTracks[arg_max])
            #print(self.adaptiveWeight.pobWeights)
            self.recombination()
            self.selection()

            idx= np.argsort(self.bestX)
            #print(idx, len(x), x)
            tcost, bitrate, solution =  cost(idx)

            print(it, round(np.mean(self.fitnessXs), 4), round(self.bestFitness, 4),[tcost, bitrate], self.adaptiveWeight.pobWeights)
            arg_maxs = np.argsort(self.fitnessXs)
            self.data[it*3:3*(it+1)] = np.copy(self.Xs[arg_maxs[:3]])

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.optimize import minimize

problem = LoraWanProblem(len(Xs))

np.random.seed(1)
de = DE(
    pop_size=20,
    variant="DE/rand/1/bin",
    CR=0.3,
)

problem = LoraWanProblem(len(Xs))
np.random.seed(2)
algorithm = VaNSAS(problem,
    pop_size=200,
    CR=0.97,
     CRT=0.95,
     F=1.67,
    max_iter = 2000,
)
algorithm.iterate()

idx= np.argsort(algorithm.bestX)
tcost, bitrate, solution =  cost(idx)
tcost, bitrate

import numpy as np
np.save("data_points_v3.npy", algorithm.data )
dataz = np.load("data_points.npy")
dataz.shape

from tqdm import tqdm
popx = dataz
tcosts = []
bitrates = []
fs = set([])
for i in tqdm(range(len(popx))):
    x = popx[i]
    idx= np.argsort(x)
    tcost, bitrate, solution =  cost(idx)
    f = round(tcost + bitrate, 2)
    if f in fs:
        continue
    fs.add(f)

    tcosts.append(tcost)
    bitrates.append(bitrate)

len(tcosts),tcosts

!pip install optuna

import optuna
import matplotlib.pyplot as plt

global ti
ti = 0
def objectiveCxB(trial):
    global ti
    x = tcosts[ti]
    y = bitrates[ti]
    ti += 1
    return x, y

study = optuna.create_study(directions=["minimize", "maximize"])
study.optimize(objectiveCxB, n_trials=len(bitrates))
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
ax = optuna.visualization.matplotlib.plot_pareto_front(study, )
ax.set_title("2-Objectives")
ax.set_xlabel("Cost")
ax.set_ylabel("Bitrate")
angle = 60
plt.show()

from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence

from optuna._experimental import experimental_func
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._pareto_front import _get_pareto_front_info
from optuna.visualization._pareto_front import _ParetoFrontInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports

global ti
ti = 0
def objectiveCxB(trial):
    global ti
    x = tcosts[ti]
    y = bitrates[ti]
    ti += 1
    return x, y

study = optuna.create_study(directions=["minimize", "maximize"])
study.optimize(objectiveCxB, n_trials=len(tcosts))

infos = _get_pareto_front_info(
        study, None, True, None, None, None
    )

fronts = []
N = (len(infos.best_trials_with_values))
for k in range(N):
    fronts.append(infos.best_trials_with_values[k][-1])

import numpy as np
fronts = np.array(fronts)

x = fronts[:, 0]
y = fronts[:, 1]

for i in range(len(fronts)):
  print(x[i], y[i])

fig = plt.figure()
ax = plt.gca()

ax.set_title("Pareto-front Multi-objective")
ax.set_ylabel("Bitrate")
ax.set_xlabel("Cost")

ax.scatter(x,y)

plt.show()

import numpy as np
import warnings

class Topsis():
    evaluation_matrix = np.array([])  # Matrix
    weighted_normalized = np.array([])  # Weight matrix
    normalized_decision = np.array([])  # Normalisation matrix
    M = 0  # Number of rows
    N = 0  # Number of columns

    '''
    # Step 1
	Create an evaluation matrix consisting of m alternatives and n criteria,
	with the intersection of each alternative and criteria given as {\displaystyle x_{ij}}x_{ij},
	we therefore have a matrix {\displaystyle (x_{ij})_{m\times n}}(x_{{ij}})_{{m\times n}}.
	'''

    def __init__(self, evaluation_matrix, weight_matrix, criteria):
        # M×N matrix
        self.evaluation_matrix = np.array(evaluation_matrix, dtype="float")

        # M alternatives (options)
        self.row_size = len(self.evaluation_matrix)

        # N attributes/criteria
        self.column_size = len(self.evaluation_matrix[0])

        # N size weight matrix
        self.weight_matrix = np.array(weight_matrix, dtype="float")
        self.weight_matrix = self.weight_matrix/sum(self.weight_matrix)
        self.criteria = np.array(criteria, dtype="float")

    '''
	# Step 2
	The matrix {\displaystyle (x_{ij})_{m\times n}}(x_{{ij}})_{{m\times n}} is then normalised to form the matrix
	'''

    def step_2(self):
        # normalized scores
        self.normalized_decision = np.copy(self.evaluation_matrix)
        sqrd_sum = np.zeros(self.column_size)
        for i in range(self.row_size):
            for j in range(self.column_size):
                sqrd_sum[j] += self.evaluation_matrix[i, j]**2
        for i in range(self.row_size):
            for j in range(self.column_size):
                self.normalized_decision[i,
                                         j] = self.evaluation_matrix[i, j]/(sqrd_sum[j]**0.5)

    '''
	# Step 3
	Calculate the weighted normalised decision matrix
	'''

    def step_3(self):
        from pdb import set_trace
        self.weighted_normalized = np.copy(self.normalized_decision)
        for i in range(self.row_size):
            for j in range(self.column_size):
                self.weighted_normalized[i, j] *= self.weight_matrix[j]

    '''
	# Step 4
	Determine the worst alternative {\displaystyle (A_{w})}(A_{w}) and the best alternative {\displaystyle (A_{b})}(A_{b}):
	'''

    def step_4(self):
        self.worst_alternatives = np.zeros(self.column_size)
        self.best_alternatives = np.zeros(self.column_size)
        for i in range(self.column_size):
            if self.criteria[i]:
                self.worst_alternatives[i] = min(
                    self.weighted_normalized[:, i])
                self.best_alternatives[i] = max(self.weighted_normalized[:, i])
            else:
                self.worst_alternatives[i] = max(
                    self.weighted_normalized[:, i])
                self.best_alternatives[i] = min(self.weighted_normalized[:, i])

    '''
	# Step 5
	Calculate the L2-distance between the target alternative {\displaystyle i}i and the worst condition {\displaystyle A_{w}}A_{w}
	{\displaystyle d_{iw}={\sqrt {\sum _{j=1}^{n}(t_{ij}-t_{wj})^{2}}},\quad i=1,2,\ldots ,m,}
	and the distance between the alternative {\displaystyle i}i and the best condition {\displaystyle A_{b}}A_b
	{\displaystyle d_{ib}={\sqrt {\sum _{j=1}^{n}(t_{ij}-t_{bj})^{2}}},\quad i=1,2,\ldots ,m}
	where {\displaystyle d_{iw}}d_{{iw}} and {\displaystyle d_{ib}}d_{{ib}} are L2-norm distances
	from the target alternative {\displaystyle i}i to the worst and best conditions, respectively.
	'''

    def step_5(self):
        self.worst_distance = np.zeros(self.row_size)
        self.best_distance = np.zeros(self.row_size)

        self.worst_distance_mat = np.copy(self.weighted_normalized)
        self.best_distance_mat = np.copy(self.weighted_normalized)

        for i in range(self.row_size):
            for j in range(self.column_size):
                self.worst_distance_mat[i][j] = (self.weighted_normalized[i][j]-self.worst_alternatives[j])**2
                self.best_distance_mat[i][j] = (self.weighted_normalized[i][j]-self.best_alternatives[j])**2

                self.worst_distance[i] += self.worst_distance_mat[i][j]
                self.best_distance[i] += self.best_distance_mat[i][j]

        for i in range(self.row_size):
            self.worst_distance[i] = self.worst_distance[i]**0.5
            self.best_distance[i] = self.best_distance[i]**0.5

    '''
	# Step 6
	Calculate the similarity
	'''
    def step_6(self):
        np.seterr(all='ignore')
        self.worst_similarity = np.zeros(self.row_size)
        self.best_similarity = np.zeros(self.row_size)

        for i in range(self.row_size):
            # calculate the similarity to the worst condition
            self.worst_similarity[i] = self.worst_distance[i] / \
                (self.worst_distance[i]+self.best_distance[i])

            # calculate the similarity to the best condition
            self.best_similarity[i] = self.best_distance[i] / \
                (self.worst_distance[i]+self.best_distance[i])

    def ranking(self, data):
        return [i+1 for i in data.argsort()]

    def rank_to_worst_similarity(self):
        # return rankdata(self.worst_similarity, method="min").astype(int)
        return self.ranking(self.worst_similarity)

    def rank_to_best_similarity(self):
        # return rankdata(self.best_similarity, method='min').astype(int)
        return self.ranking(self.best_similarity)

    def calc(self):
        print("Step 1\n", self.evaluation_matrix, end="\n\n")
        self.step_2()
        print("Step 2\n", self.normalized_decision, end="\n\n")
        self.step_3()
        print("Step 3\n", self.weighted_normalized, end="\n\n")
        self.step_4()
        print("Step 4\n", self.worst_alternatives,
              self.best_alternatives, end="\n\n")
        self.step_5()
        print("Step 5\n", self.worst_distance, self.best_distance, end="\n\n")
        self.step_6()
        print("Step 6\n", self.worst_similarity,
              self.best_similarity, end="\n\n")

fronts
len(fronts)
evaluation_matrix = fronts
weights = [0.5, 0.5] # cost , bitrate
criterias = np.array([False, True])
t = Topsis(evaluation_matrix, weights, criterias)
t.calc()

print("best_distance\t", t.best_distance)
print("worst_distance\t", t.worst_distance)

print("worst_similarity\t", t.worst_similarity)
print("rank_to_worst_similarity\t", t.rank_to_worst_similarity())

print("best_similarity\t", t.best_similarity)
print("rank_to_best_similarity\t", t.rank_to_best_similarity())

tcost, bitrate = fronts[13-1] #0.5 0.5
tcost, bitrate

import numpy as np
fronts = np.array(fronts)
x = fronts[:, 0]

y = fronts[:, 1]


fig = plt.figure()
ax = plt.gca()

ax.set_title("Pareto-front Multi-objective")
ax.set_ylabel("Bitrate")
ax.set_xlabel("Cost")

ax.scatter(x,y)
ax.scatter([tcost],[bitrate], color='blue', marker='^',s=60)

plt.show()

from tqdm import tqdm
popx = dataz
tcosts = []
bitrates = []
fs = set([])
bestTopsis = None
for i in tqdm(range(len(popx))):
    x = popx[i]
    idx= np.argsort(x)
    tcosta, bitratea, solution =  cost(idx)

    f = round(tcosta + bitratea, 2)
    if f in fs:
        continue
    fs.add(f)

    if tcost == tcosta and bitratea == bitrate:
        print(tcost, bitrate)
        bestTopsis = x
        break

    tcosts.append(tcost)
    bitrates.append(bitrate)

idx= np.argsort(x)
tcosta, bitratea, solution =  cost(idx)

tcost,bitrate, len(solution)

for cluster in solution:
    print(cluster)

import matplotlib.pyplot as plt
plt.style.use('classic')
plt.figure(figsize=(20, 20))
plt.scatter(Xs, Ys)

for cluster in solution:
    center_id = cluster['ids'][0]

    idlines = []
    for j in range(len(cluster['ids'])):
        idlines.append(center_id)
        idlines.append(cluster['ids'][j])
        idlines.append(center_id)
    plt.plot(Xs[idlines], Ys[idlines], lw =0.25)
    plt.scatter(Xs[[center_id]], Ys[[center_id]], color='red', s=70)

for cluster in solution:
    center_id = cluster['ids'][0]
    plt.scatter(Xs[[center_id]], Ys[[center_id]], color='red', s=100,zorder=10)

def getPopulaton(getway, closeset=None):
    print("Hello")
    endvice={"ids":[]}

    collect_capacity=0
    collect_cost=0
    collect_bitrate=0
    sort_distance = lookup_sort[getway]
    countDevice=len(sort_distance)
    for j in range(countDevice) :
        cid = sort_distance[j][0]
        dis = sort_distance[j][1]
        if closeset != None and cid in closeset:
            continue

        if dis > 8:
            break
        print("cid", cid)
        check_capcity=collect_capacity+pop[cid]
        print("check_capcity", check_capcity)
        if(check_capcity <= capacity):
            endvice['ids'].append( sort_distance[j][0])
            collect_capacity += pop[sort_distance[j][0]]
            collect_bitrate += getBitrate(sort_distance[j][1])
        else:
            break

    total_endvice=len(endvice['ids'])
    collect_bitrate =collect_bitrate/total_endvice
    collect_bitrate = "%.3f" % round(collect_bitrate, 3)
    endvice["totalVillage"]=total_endvice
    endvice["bitrate"]=collect_bitrate
    endvice["gateway"]=getway
    endvice["capacity"]=collect_capacity
    endvice["cost"]=variable_cost[getway]+base_cost
    return endvice


def cost(idx):
    closeset = set([])
    solution = []
    i = 0
    while len(closeset) < len(idx):
        center = idx[i]
        if center in closeset:
            i = i+1
            continue
        print(center)
        cluster = getPopulaton(center, closeset)
        solution.append(cluster)

        #print(cluster)
        closeset = closeset.union(cluster['ids'])
        i = i+1
    total_cost = 0
    total_birate = 0
    for cluster in solution:
        #print(cluster)
        total_cost += cluster['cost']
        total_birate += float(cluster['bitrate'])
    return total_cost, total_birate/len(solution), solution