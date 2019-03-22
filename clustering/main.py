'''
Some quick warm up exercises for part time students
'''

def fact(n):
    if(n <= 1):
        return 1
    else:
        return n * fact(n-1)

print(fact(4))
print([fact(i) for i in range(5)])

import numpy as np

a = [
    [1,-2,3],
    [2,1,1],
    [-3,2,-2]
]

b = [7,4,-10]
x = np.linalg.solve(a,b)
print(x)
print(a @ x)

del a, b, x

x = np.linspace(0,10,1e3)
y = np.sin(x)

for i in range(0,x.size):
    print(x[i],y[i])

import matplotlib.pyplot as plt

plt.plot(x,y)
plt.show()