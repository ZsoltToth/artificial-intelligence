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

print(np.linalg.solve(a,b))
