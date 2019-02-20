'''
Numpy replaces the arrays with ndarray and provides a wide range of matrix operations.
'''
import numpy as np
import random as rnd
'''
Arithmetic oprations are used elementwise with ndarrays
'''
t = np.linspace(0,10,1000)
y = t**2 + 5*t -3

'''
You can solve simple linear equation systems with Gauss-Jordan Method.
'''
a = np.array([[3,1],[1,2]])
b = np.array([9,8])

x = np.linalg.solve(a,b)

print(x)

del t,y,a,b,x

'''
numpy can be used for linear regression and curve fitting
Let generate a simple quadratic function with some noise (-1,+1)
'''
x = np.linspace(-10,10,100)
noise = [2* rnd.random() -1 for i in x]
y = x**2 + noise
#polyfit function fits a n-degree polynom to the measurement
approxPolynom = np.polyfit(x,y,deg=2)
polynom = np.poly1d(approxPolynom)
yApprox = np.array([polynom(i) for i in x])

print("Error: %lf" % ((y - yApprox)**2).sum())