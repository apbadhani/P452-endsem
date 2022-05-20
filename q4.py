from MNL import *
import math
#--------Defining potential function---------#
def pote(x):
    return 1.0/math.sqrt(1+x**2)
#---------Legendre zeros and weights------#
#------------ This function takes degree as the input and returns the set of zeros and weights-------# 
def leg_zero_wei(deg):
    if deg == 4:
        return [0.861136311, 0.339981043, -0.339981043, -0.861136311],[0.347854845, 0.652145154, 0.652145154, 0.347854845]
    elif deg == 5:
        return [0.906179845, 0.538469310, 0, -0.538469310, -0.906179845],[0.236926885, 0.478628670, 0.568888889, 0.478628670, 0.236926885]
    elif deg == 6:
        return [0.932469514, 0.661209386, 0.238619186, -0.238619186, -0.661209386, -0.932469514],[0.171324492, 0.360761573, 0.467913934, 0.467913934, 0.360761573, 0.171324492]

#---------- Gauss Quadrature function------#
def gaus_lege(f,n):
    s = 0
    #For weights and zeros
    zero, wei = leg_zero_wei(n)
    for i in range(n):
        s += wei[i-1] * f(zero[i-1])
    return s
num = [4,5,6]
for i in num:
    print("For degree ",i,", Potential is :",gaus_lege(pote,i))

#-----------Results----------#
#For degree  4 , Potential is : 1.7620541789046658
#For degree  5 , Potential is : 1.7628552954010728
#For degree  6 , Potential is : 1.762730048499759

#Gaussian Quadrature method provides 2N-1 accuarcy with N points(roots).
#For degree 4 to degree 6 , Gaussian quadrature gives the same integration value for first 3 decimal places.
#As we increase the degree we approach towards more precise integration value.

