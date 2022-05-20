import math
from MNL import *
from matplotlib import pyplot as plt
import numpy as np

#----------MLCG parameters--------#
seed = 7
a = 572
m = 16381
n = 200
m = 500
#-------Random Walk using MLCG----------#
#-----Takes seed,a,m,n,m as input, generates a list of random numbers
#--------and use them for random walks------#
#---returns avg rms value after 500 walks
def mlcg_randwalk(N, M, seed, m, a, rNums):
    rms = 0
    d = []
    #500 random walks with 200 step size
    for i in range(0, M):
        #Takes a set of random values gives back rms value
        d = random_walk(N,seed,m,a,rNums)
        rms += (d[0][-1])**2
    return math.sqrt(rms/M)

#------------Storing random numbers-------#
val = [0]*n
#---------Calling the new randomly generated walks function-----#
rms = mlcg_randwalk(n, m, seed, m, a, val)
print("RMS distance obtained from 500 random walks :",rms,"\n","Square root of N is                :",math.sqrt(n))

#----------------Results-------------#
#RMS distance obtained from random walk : 14.089723524998776
#Square root of N is                : 14.142135623730951

#From the values obtained after M random walks of N steps we can see the
#the avg rms distance is proportional to square root of N or independent
#of the number of walks done.






