from MNL import *
import numpy as np
import pylab
import random
import math
n = 500
x = np.zeros(n)
y = np.zeros(n)
r = np.zeros(n)
for i in range(1, n):
    val = random.randint(1, 4)
    if val == 1:
        x[i] = x[i - 1] + 1
        y[i] = y[i - 1]
        r[i] = math.sqrt(x[i]**2 + y[i]**2)
    elif val == 2:
        x[i] = x[i - 1] - 1
        y[i] = y[i - 1]
        r[i] = math.sqrt(x[i]**2 + y[i]**2)
    elif val == 3:
        x[i] = x[i - 1]
        y[i] = y[i - 1] + 1
        r[i] = math.sqrt(x[i]**2 + y[i]**2)
    else:
        x[i] = x[i - 1]
        y[i] = y[i - 1] - 1
        r[i] = math.sqrt(x[i]**2 + y[i]**2)
d = 0
for i in range(1,n):
    d+=r[i]**2
print("Rms distance: ",math.sqrt(d/500))
# plotting stuff:
pylab.title("Random Walk ($n = " + str(n) + "$ steps)")
pylab.plot(x, y)
pylab.savefig("rand_walk" + str(n) + ".png", bbox_inches="tight", dpi=600)
pylab.show()

#-----------Number of steps in random walk-------#
N = 200
a = 572
m = 16381
s = 3







