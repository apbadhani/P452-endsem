import numpy as np
import matplotlib.pyplot as plt
h = 0.1
k = 0.008
x = np.arange(0,2+h,h).round(3)
t = np.arange(0,4+k,k).round(3)
#----Boundary Condition-----#
boundary = [0,0]
#----Initial condition------#
initial = 20*np.sin(np.pi*x)
n = len(x)
m = len(t)
T = np.zeros((n,m))
T[0,:] = boundary[0]
T[-1,:] = boundary[1]
T[:,0] = initial
print(T.round(3))
fac = k/h**2
for j in range(1,m):
    for i in range(1,n-1):
        T[i,j] = fac*T[i-1,j-1] + (1-2*fac)*T[i,j-1] + fac*T[i+1,j-1]
T = T.round(3)
plt.plot(T)
m = [0,10,20,50,100,200,500]
plt.legend(m)
plt.xlabel("Index number")
plt.ylabel("Temperature")


#The curvature of the plot depends on the accuracy/precision of the factor.
#If we increase the time step size, the maxima of the curve starts decreasing.