import numpy as np
import matplotlib.pyplot as plt
#-----Defining heat equation solver---------#
def sol_explicit(u, t_steps, alpha):
    for i in range(t_steps-1):
        u[i+1, 1:-1] = u[i, 1:-1]+alpha*(u[i, 0:-2]-2*u[i, 1:-1]+u[i, 2:])
    return u
#----X distance-------#
x = np.linspace(0, 2, 21)
#-----time steps--------#
t = np.linspace(0, 40, 5001)
#----Alpha factor------#
alpha = 0.4
u = np.zeros((5001, 21))
#------Initial condiitons------#
u[0, :] = 20*np.abs(np.sin(np.pi*x))
#-----Solving the equation---#
u = sol_explicit(u, 5000, alpha)
#--------Plotting------#
#--time = 0----------#
plt.plot(x, u[0, :])
plt.xlabel('X')
plt.ylabel('Temperature')
plt.show()
#--time = 10----------#
plt.plot(x, u[10, :])
plt.xlabel('X')
plt.ylabel('Temperature')
plt.show()
#--time = 20----------#
plt.plot(x, u[20, :])
plt.xlabel('X')
plt.ylabel('Temperature')
plt.show()
#--time = 50----------#
plt.plot(x, u[50, :])
plt.xlabel('X')
plt.ylabel('Temperature')
plt.show()
#--time = 100----------#
plt.plot(x, u[100, :])
plt.xlabel('X')
plt.ylabel('Temperature')
plt.show()
#--time = 200----------#
plt.plot(x, u[200, :])
plt.xlabel('X')
plt.ylabel('Temperature')
plt.show()
#--time = 500----------#
plt.plot(x, u[500, :])
plt.xlabel('X')
plt.ylabel('Temperature')
plt.show()


#From the figures we can see that the temperature at the midpoint starts rising and attains a maxima.
#As the boundary condition says that the temperature should be 0 at boundaries.
#Most of the heat flows in the middle because of boundary conditions
