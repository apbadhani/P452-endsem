from MNL import *
import matplotlib.pyplot as plt
#---------Fiiting function with orthonormal Legendre basis-----------#
def fitw_lege(xvals, yvals, deg):
    n = len(xvals)
    para = deg + 1
    A = np.zeros((para, para))
    b = np.zeros(para)

    for i in range(para):
        for j in range(para):
            total = 0
            for k in range(n):
                total += lege_poly(xvals[k], j) * lege_poly(xvals[k], i)

            A[i, j] = total

    for i in range(para):
        total = 0
        for k in range(n):
            total += lege_poly(xvals[k], i) * yvals[k]

        b[i] = total

    para = lu_decomp(A, b)
    return para
#----Data reading
file = read_csv("esem4fit.txt")
u = [sub[0] for sub in file]
v = [i.split('\t',1) for i in u]
x = []
y = []
for i in range(len(v)):
    for j in range(1):
        x.append(v[i][0])
        y.append(v[i][1])
x = list(map(float,x))
y = list(map(float,y))
#--------Defining Legendre Polynomials----------#
def lege_poly(x,order):
    if order == 0:
        return 1
    elif order == 1:
        return x
    elif order == 2:
        return 0.5*(3*x**2 - 1)
    elif order == 3:
        return 0.5*(5*x**3 - 3*x)
    elif order == 4:
        return 1/8*(35*x**4 -30*x**2 +3)
    elif order == 5:
        return 1/8*(63*x**5 -70*x**3 + 15*x)
    elif order == 6:
        return 1/16*(231*x**6 - 315*x**4 + 105*x**2 -5)
coef = fitw_lege(x,y,6)
#--------Plotting the data points--------#
plt.plot(x,y,"r.")
print('Obtained coefficients in Legendre basis for degree = 6: ',coef)
#---------Function for plotting new fit------#
def plot_fit(coeff,lege_poly):
    xval = []
    for x_val in x:
        sum = 0
        for i in range(6):
            sum= sum + coeff[i]*lege_poly(x_val,i)
        xval.append(sum)
    return xval
xval = plot_fit(coef,lege_poly)
#----------Plotting the modified fit-------#
plt.plot(x, np.array(xval))
plt.title('Degree = 6')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Obtained coefficients in Legendre basis for degree = 6:  [0.07003196671971398, 0.004301685837864321, -0.010166710608800473, 0.013083743602879212, 0.11411855049286529, -0.006726972223322476, -0.0123845597126462]
