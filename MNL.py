import numpy as np
import sys
import math
import copy
from scipy.special.orthogonal import p_roots
import random


#Creating a zero matrix of order m*n
def zeromatrix(m,n):
        p= [[0 for i in range(n)] for j in range(m)]
        return(p)
#Calculates the norm of a vector
def norm(x):
    total = 0
    for i in range(len(x)):
        total += x[i]**2

    return total**(1/2)

# Function for printing the matrix
def mat_print(a):
    for i in range(len(a)):
        print(a[i])

# Fucnction for vector subtraction
def vec_sub(a, b):
    if (len(a) != len(b)):
        exit()
    else:
        return [x1 - x2 for (x1, x2) in zip(a, b)]

# Function for reading a csv file
def read_csv(path):
    with open(path, 'r+') as file:
        results = []

        for line in file:
            line = line.rstrip('\n') # remove `\n` at the end of line
            items = line.split(',')
            results.append(list(items))

        # after for-loop
        return results
#Forward backward substitution
def for_back(U, L, b):
    n = len(b)
    y = [0 for i in range(n)]

    for i in range(n):
        total = 0
        for j in range(i):
            total += L[i][j] * y[j]
        y[i] = (b[i] - total)/L[i][i]

    x = [0 for i in range(n)]

    for i in reversed(range(n)):
        total = 0
        for j in range(i+1, n):
            total += U[i][j] * x[j]
        x[i] = (y[i] - total)/U[i][i]

    return x
# Matrix multiplication
def matmul(a, b):
    product = [[sum(i*j for i,j in zip(a_row, b_col)) for b_col in zip(*b)] for
            a_row in a]

    return product
#function for matrix vector multiplication
def mat_vec_mult(A, B):
    n = len(B)
    if len(A[0]) == n:
        p = [0 for i in range(n)]
        for i in range(n):
            for j in range(n):
                p[i] = p[i] + (A[i][j] * B[j])
        return (p)
    else:
        print('This combination is not suitable for multiplication')

def partial_pivot(A, b):
    count = 0
    n = len(A)
    for i in range(n-1):
        if abs(A[i][i]) < 1e-10:
            for j in range(i+1,n):
                if abs(A[j][i]) > abs(A[i][i]):
                    A[j], A[i] = A[i], A[j]  # interchange ith and jth rows of matrix 'A'
                    count += 1
                    b[j], b[i] = b[i], b[j]  # interchange ith and jth elements of vector 'b'
    return A, b,count
#Gauss-Jordan
def gau_jor(A, b):
    n = len(A)
    partial_pivot(A, b)
    for i in range(n):
        pivot = A[i][i]
        b[i] = b[i] / pivot
        for c in range(i, n):
            A[i][c] = A[i][c] / pivot

        for k in range(n):
            if k != i and A[k][i] != 0:
                factor = A[k][i]
                b[k] = b[k] - factor*b[i]
                for j in range(i, n):
                    A[k][j] = A[k][j] - factor*A[i][j]

    x = b
    return x

#LU Decomposition
def lu_decomp(A, b):
    def cr_out(A):
        U = [[0 for i in range(len(A))] for j in range(len(A))]
        L = [[0 for i in range(len(A))] for j in range(len(A))]

        for i in range(len(A)):
            L[i][i] = 1

        for j in range(len(A)):
            for i in range(len(A)):
                total = 0
                for k in range(i):
                    total += L[i][k] * U[k][j]

                if i == j:
                    U[i][j] = A[i][j] - total

                elif i > j:
                    L[i][j] = (A[i][j] - total)/U[j][j]

                else :
                    U[i][j] = A[i][j] - total

        return U, L

    partial_pivot(A, b)
    U, L = cr_out(A)
    x = for_back(U, L, b)
    return x
#----------Cholesky Decomposition--------#
def lu_w_choles(A,k):
    def chole_decom(A):
        n = len(A)
    # Create zero matrix for L
        L = [[0 for i in range(n)] for j in range(n)]
        Lt = [[0 for i in range(n)] for j in range(n)]
    # Perform the Cholesky decomposition
        for i in range(n):
            for k in range(i + 1):
                tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))

                if (i == k):  # Diagonal elements
                    L[i][k] = math.sqrt(A[i][i] - tmp_sum)
                else:
                    L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))
        for i in range(n):
            for j in range(n):
                Lt[i][j] = L[j][i]
        return L,Lt
    M,N = chole_decom(A)
    x = for_back(N,M,k)
    return x,M
# Jacobi Method
def jacobi(A, b, tol):
    n = len(A)
# define a dummy vector for storing solution vector
    x = [1 for i in range(n)]
    xold = [0 for i in range(n)]
    iterations = []; residue = [];
    count = 0
    while norm(vec_sub(xold, x)) > tol:
        iterations.append(count)
        count += 1
        residue.append(norm(vec_sub(xold, x)))
        xold = x.copy()
        for i in range(n):
            total = 0
            for j in range(n):
                if i != j:
                    total += A[i][j] * x[j]

            x[i] = 1/A[i][i] * (b[i] - total)

    return x, iterations,residue

#Gauss Seidel
def gauss_seidel(A, b, tol):
    n = len(A)
    x = [0 for i in range(n)]
    xold = [1 for i in range(n)]
    iterations = []; residue = [];
    count = 0

    while norm(vec_sub(x, xold)) > tol:
        xold = x.copy()
        iterations.append(count)
        count += 1
        for i in range(n):
            d = b[i]
            for j in range(n):
                if j != i:
                    d -= A[i][j] * x[j]

            x[i] = d / A[i][i]

        residue.append(norm(vec_sub(x, xold)))

    return x, iterations,residue
#Conjugate Gradient
def conjgrad(A, b, tol):
    n = len(b)
    x = [1 for i in range(n)]
    r = vec_sub(b, vecmul(A, x))
    d = r.copy()
    rprevdot = dotprod(r, r)
    iterations = []; residue = [];
    count = 0       # counts the number of iterations

    # convergence in n steps
    for i in range(n):
        iterations.append(count)
        Ad = vecmul(A, d)
        alpha = rprevdot / dotprod(d, Ad)
        for j in range(n):
            x[j] += alpha*d[j]
            r[j] -= alpha*Ad[j]
        rnextdot = dotprod(r, r)
        residue.append(sqrt(rnextdot))
        count += 1

        if sqrt(rnextdot) < tol:
            return x, iterations, residue

        else:
            beta = rnextdot / rprevdot
            for j in range(n):
                d[j] = r[j] + beta*d[j]
            rprevdot = rnextdot
#Givans method
def crossprod(A,B):
    if len(A[0]) == len(B):
        crossprod = [[0 for i in range(len(B[0]))]for j in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for m in range(len(A)):
                    crossprod[i][j] = crossprod[i][j] + A[i][m]*B[m][j]
        return crossprod
    else:
        print("Matrices cannot be multiplied")
# crossprod is used in the function gaussgivan
def maxoff(A):
    maxtemp = A[0][1]
    k = 0
    l = 1
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            if abs(A[i][j]) > abs(maxtemp):
                maxtemp = A[i][j]
                k = i
                l = j
    return maxtemp, k, l


def gaussgivan(A, ep):
    max, i, j = maxoff(A)
    while abs(max) >= ep:
        #calculating theta
        if A[i][i] - A[j][j] == 0:
            theta = math.pi / 4
        else:
            theta = math.atan((2 * A[i][j]) / (A[i][i] - A[j][j])) / 2
        #Identity matrix
        P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        #Making P matix tridiagonal
        P[i][i] = P[j][j] = math.cos(theta)
        P[i][j] = -1 * math.sin(theta)
        P[j][i] = math.sin(theta)
        AP = crossprod(A, P)
        #making P an array so to use transpose function
        P = np.array(P)
        #Transpose of P
        PT = P.T.tolist()
        #getting back the matrix in tridiagonal form
        A = crossprod(PT, AP)
        #checking the offset in the matrix obtained
        max, i, j = maxoff(A)
    return A

#frobenius norm
def frob_norm(A):
    sum = 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            sum = sum + (A[i][j] ** 2)
    return math.sqrt(sum)
#gives the norm of A
def pow_norm(A):
    max = 0
    for i in range(len(A)):
        if max <= A[i][0]:
            max = A[i][0]
    normA = scaler_matrix_division(max, A)
    return normA

# Power method
def pow_method(A, x0=[[1], [1], [1]], eps=1.0e-4):
    i = 0
    lam0 = 1
    lam1 = 0
    while abs(lam1 - lam0) >= eps:
        # print("error=",abs(lam1-lam0))
        if i != 0:
            lam0 = lam1

        Ax0 = mat_mult(A, x0)
        AAx0 = mat_mult(A, Ax0)
        # print("Ax0=",Ax0)
        # print("AAx0=",AAx0)
        dotU = inner_product(AAx0, Ax0)
        dotL = inner_product(Ax0, Ax0)
        # print("U=",dotU)
        # print("L=",dotL)
        lam1 = dotU / dotL

        x0 = Ax0
        i = i + 1
        # print("i=",i)

        # print("eigenvalue=",lam1)
        ev = pow_norm(x0)
        # print ("eigenvector=",ev)
    return lam1, ev  # returns lam1=largest eigen value and ev = coressponding eigen vec
#gives mean
def Mean(A):
    n = len(A)
    sum = 0
    mean = 0
    for i in range(n):
        sum = sum + A[i]
    return sum/n
#gives variance
def Variance(A):
    n = len(A)
    mean = Mean(A)
    sum = 0
    for i in range(n):
        sum = sum + (A[i]-mean)**2
    return sum/n
#solves equation
def solveeqn(m, qw):
    m = Invert(m)

    X = []
    X.append(m[0][0]*qw[0] + m[0][1]*qw[1])
    X.append(m[1][0]*qw[0] + m[1][1]*qw[1])
    return(X)
def sum1(X, n):
    n = n + 1
    suMatrix = []
    j = 0
    while j<2*n:
        sum = 0
        i = 0
        while i< len(X):
            sum = sum + (X[i])**j
            i = i + 1
        suMatrix.append(sum)
        j = j+1
    return suMatrix
#makes a new matrix
def makemat(suMatrix, n):
    n = n + 1
    m = [[0 for i in range(n)]for j in range(n)]
    i = 0
    while i<n:
        j = 0
        while j<n:
            m[i][j] = suMatrix[j+i]
            j = j+1
        i = i + 1
    return m

def sum2(X, Y, n):
    n = n+1
    suMatrix = []
    j = 0
    while j<n:
        sum = 0
        i = 0
        while i< len(X):
            sum = sum + ((X[i])**j)*Y[i]
            i = i + 1
        suMatrix.append(sum)
        j = j+1
    return suMatrix

#chi square fit function
def fit(X,Y):
    k = sum1(X, 1)         #taking all the sigma_x
    m = makemat(k, 1)      #sigma_x**i matrix

    qw = sum2(X, Y, 1)     #sigma_x**i*y matrix

    X = solveeqn(m, qw)
    return X[0],X[1]
#----------Backward substitution-----------#
def bkwdsub(A ,B):
    global X
    X = []
    for k in range(len(A)):
        X.append(float(0))
    for i in reversed(range(len(A))):
        val = 0
        for j in reversed(range(0, len(A))):
            if j > i:
                val += A[i][j]*X[j]
        X[i] += (1/A[i][i])*(B[i] - val)
    return X
#---------Forward substitution-------------#
def fwrdsub(A,B):
    global Y
    Y = []
    for k in range(len(A)):
        Y.append(float(0))
    for i in range(0, len(A)):
        val = 0
        for j in range(0, i):
            val += A[i][j]*Y[j]
        Y[i] += (B[i] - val)
    return Y
#-----Finite time difference method------#
def finite_dif_method(func, n, min, max):
    h = (max - min)/n
    A = np.zeros((n + 1, n + 1))
    A[0, 0] = 1
    A[n, n] = 1
    for i in range(1,n):
        A[i, i-1] = 1
        A[i, i] = -2
        A[i, i+1] = 1

    b = np.zeros(n+1)
    b[1:-1] = func
    b[-1] = 50
    print(b)

    # solve linear equation
    guess = [0.10* (i+1) for i in range(10+1)]
    y = jacobi(A,b,guess,50)[0]
    print(y)

    LU = lu_decomp(A,b)
    F = fwrdsub(LU, b)
    B = bkwdsub(LU, F)
    print(B)

    # Plot linear equation
    t = np.linspace(min, max, n+1)

    # plt.figure(figsize=(10,8))
    # plt.plot(t, y)
    # plt.plot(5, 50, 'ro')
    # plt.xlabel('')
    # plt.ylabel('')
    # plt.show()
# Bootstrap method
def bootstrap(A,b):
    mean = []
    vari = []
    for i in range(b):
        #making bootstrap dataset
        resample = random.choices(A,k=len(A))
        #calculating mean of the resampled data
        m = Mean(resample)
        mean.append(m)
        var = Variance(resample)
        vari.append(var)
    #to get confidence levels we calculate Standard deviation of this distribution
    x = (Mean(mean))
    y = (Mean(var))
    #plotting the mean values as a histogram
    plt.hist(mean)
    return x,y

#jackknife method
def jkknife(A):
    n = len(A)
    yi = []
    for i in range(n):
        B = A.copy()
        del(B[i])
        #calculating mean excluding one element
        mean = Mean(B)
        #MAking a new y vector, stores all means
        yi.append(mean)
    #mean of the new formed set
    yibar = Mean(yi)
    sum = 0
    for i in range(n):
        sum = sum + (yi[i] - yibar)**2
    #calculating error
    err = ((n-1)/n)*sum
    return yibar,err
#-----Polysquare fit-------#
def sq_fit_poly(x, y, pow):
    n = len(x)
    u = n + 1
    M = np.zeros((u,u))
    a = np.zeros(u)
    for i in range(u):
         for j in range(u):
            total = 0
            for k in range(n):
                total += x[k] ** (i + j)
            M[i, j] = total
    for i in range(u):
        total = 0
        for k in range(n):
            total += x[k] ** i * y[k]
        a[i] = total
    coef = lu_decomp(M, a)
    return coef, M
#----Data reading
#file = read_csv(file_in_directory.txt)
#u = [sub[0] for sub in file]
#v = [i.split('\t',1) for i in u]
#x = []
#y = []
#for i in range(len(v)):
    #for j in range(1):
        #x.append(v[i][0])
        #y.append(v[i][1])
#-------Random Walk in 2D--------#
def randwalk(n): #n is the number of steps(increase in the value of n increses the compelxity of graph)
    x = np.zeros(n) # x and y are arrays which store the coordinates of the position
    y = np.zeros(n)
    direction=["NORTH","SOUTH","EAST","WEST"] # Assuming the four directions of movement.
    for i in range(1, n):
        step = random.choice(direction) #Randomly choosing the direction of movement.
        if step == "EAST": #updating the direction with respect to the direction of motion choosen.
            x[i] = x[i - 1] + 1
            y[i] = y[i - 1]
        elif step == "WEST":
            x[i] = x[i - 1] - 1
            y[i] = y[i - 1]
        elif step == "NORTH":
            x[i] = x[i - 1]
            y[i] = y[i - 1] + 1
        else:
            x[i] = x[i - 1]
            y[i] = y[i - 1] - 1
        plt.title("Random Walk ($n = " + str(n) + "$ steps)")
        plt.plot(x,y)
        plt.show()
#Defining Chebyshev function
def chebyshev(x, order):
    if order == 0:
        return 1
    elif order == 1:
        return 2*x - 1
    elif order == 2:
        return 8*x**2 - 8*x + 1
    elif order == 3:
        return 32*x**3 - 48*x**2 + 18*x - 1

#Defining the function for chebyshev fit
def fitw_cheby(xvals, yvals, degree):
    n = len(xvals)
    para = degree + 1
    A = np.zeros((para, para))
    b = np.zeros(para)

    for i in range(para):
        for j in range(para):
            total = 0
            for k in range(n):
                total += chebyshev(xvals[k], j) * chebyshev(xvals[k], i)

            A[i, j] = total

    for i in range(para):
        total = 0
        for k in range(n):
            total += chebyshev(xvals[k], i) * yvals[k]

        b[i] = total

    para = lu_decomp(A, b)
    return para,A
# Function for Pseudo random number generator
def mu_li_co_ge(seed, a, m, num):
    x = seed
    rands = []
    for i in range(num):
        x = (a*x) % m
        rands.append(x)
    return rands
#---------Gauss Legendre Quadrature--------#
def gau_leg(f,n,a,b):
    #Two point gauss quadrature
    #For Gauss-Legendre qudrture - w = 1
    [x,w] = p_roots(n+1)
    G=0.5*(b-a)*sum(w*f(0.5*(b-a)*x+0.5*(b+a)))
    return G
#-----First derivative-----------#
def der1(f,x_pre,h=0.0002):
    return (f(x_pre + h) - f(x_pre - h))/2*h
#---------Explicit Euler method---------#
def ex_eul(f,s0,h):
    #h is the step size, less order more precise
    # Numerical grid
    t = np.arange(0, 1 + h, h)
    # Explicit Euler Method
    s = np.zeros(len(t))
    s[0] = s0
    for i in range(0, len(t) - 1):
        s[i + 1] = s[i] + h*f(t[i], s[i])
    return t,s
#--------Implicit Euler method-------#
def both_ex_imp_eu(f,x0,xi,xf,h):
    nt = int((xf-xi)/h)
    N = np.empty((nt + 1, 2))
    N[0] = Ni, Ni
    #define own function
    coef_imp = (1. + alpha * dt) ** (-1)

    for i in range(nt):
        N[i + 1, 0] = N[i, 0] - alpha * N[i, 0] * dt

        N[i + 1, 1] = coef_imp * N[i, 1]
    return N[:,0],N[:,1]


#------Newton Raphson method---------#
def newrap(x,f,max=118):
    errors = []
    file1=open("file3.txt","w+")
    for i in range(max):
        if der1(f, x) == 0.0:
            print('Divide by zero error!')
            return
        x_prev = x
        # update x as per newton-raphson formula
        x = x - f(x) / der1(f,x)
        file1.write("%f\r\n" % abs((x - x_prev)))
        # append the absolute error in list
        errors.append(abs(x - x_prev))
        # check if convergence criteria satisfied
        if abs(x - x_prev) < 10 ** (-6):
            file1.close()
            return x
#------Second derivative-------#
def der2(f,a,h=0.0002):
    return ((f(x + h) - 2*f(x) + f(x - h))/(2*h*h))
#-----------------RK4 method------------#
def RK4(x, y, h, range, func):
    X = [];
    Y = []  # ; Z = []
    while x <= range:
        k1 = h * func(x, y)
        k2 = h * func(x + h / 2, y + k1 / 2)
        k3 = h * func(x + h / 2, y + k2 / 2)
        k4 = h * func(x + h, y + k3)

        x = x + h
        y = y + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        X.append(x)
        Y.append(y)
    return X, Y
#-------------------Coupled RK4-------------#
def RK4coupled(x,y,z,h,r,funcf,funcg):
    #Values stored in arrays
    X = []
    Y = []
    Z = []
    #Calculate when x less than range
    while x <= r and x>= -r:
        k1 = h*funcf(x, y, z)
        l1 = h*funcg(x, y, z)

        k2 = h * funcf(x + h/2, y, z + l1/2)
        l2 = h * funcg(x + h/2, y + k1/2, z + l1/2)

        k3 = h * funcf(x + h/2, y, z + l2 / 2)
        l3 = h * funcg(x + h/2, y + k2 / 2, z + l2 / 2)

        k4 = h * funcf(x + h, y, z + l3)
        l4 = h * funcg(x + h, y + k3, z + l3)
        #Calculate for y and v i.e z for next iteration
        y = y + 1/6*(k1+2*k2+2*k3+k4)
        z = z + 1/6*(l1+2*l2+2*l3+l4)
        x = x + h
        X.append(x)
        Y.append(y)
        Z.append(z)
    return X,Y
