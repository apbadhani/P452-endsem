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
#----Alpha factor------#--- a = dt/dx**2
a = 0.08
u = np.zeros((5001, 21))
#------Initial condiitons------#
u[0, :] = 20*np.abs(np.sin(np.pi*x))
#-----Solving the equation---#
u = sol_explicit(u, 5000, a)
#--------Plotting------#
#--time = 0----------#
plt.plot(x, u[0, :],color='orange')
plt.xlabel('X')
plt.ylabel('Temperature')
plt.show()
#--time = 10----------#
plt.plot(x, u[10, :],color="red")
plt.xlabel('X')
plt.ylabel('Temperature')
plt.show()
#--time = 20----------#
plt.plot(x, u[20, :],color="yellow")
plt.xlabel('X')
plt.ylabel('Temperature')
plt.show()
#--time = 50----------#
plt.plot(x, u[50, :],color="green")
plt.xlabel('X')
plt.ylabel('Temperature')
plt.show()
#--time = 100----------#
plt.plot(x, u[100, :])
plt.xlabel('X')
plt.ylabel('Temperature')
plt.show()
#--time = 200----------#
plt.plot(x, u[200, :],color="brown")
plt.xlabel('X')
plt.ylabel('Temperature')
plt.show()
#--time = 500----------#
plt.plot(x, u[500, :],color="magenta")
plt.xlabel('X')
plt.ylabel('Temperature')
plt.show()

#By analyzing the plots and comparing it with our initial abs(sin(pi*x)) plot
#and how with time the tip changes into a plataeu.
#We can conclude that the temperature at the midpoint starts rising and is
#attaining a maxima starting from a minima. It can also be infered if we look
#at our boundary condition which says that the temperature should be 0 at boundaries.
#So it automatically implies that somewhere in between should exist
#some kind of maxima, most of the heat flows in the middle.
