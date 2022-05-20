import math
from MNL import *
from matplotlib import pyplot as plt
import numpy as np

#----------MLCG parameters--------#
seed = 9
a = 572
m = 16381
N = 200
M = 500
#-------Random Walk using MLCG----------#
def mlcg_randwalk(N, M, seed, m, a, rNums):
    # M walks of N steps
    rms = 0
    d = []
    for i in range(0, M):
        d = random_walk(N,seed,m,a,rNums)
        rms += (d[2][-1])**2
    return math.sqrt(rms/M)
#------------Storing random numbers-------#
rNums = [0]*N
rms = mlcg_randwalk(N, M, seed, m, a, rNums)
print("RMS value obtained from random walk :",rms,"\n","Square root of N is                :",math.sqrt(N))

#----------------Results-------------#
#RMS value obtained from random walk : 15.383076564051871
#Square root of N is                : 14.142135623730951








