import numpy as np
import matplotlib.pyplot as plt

u = 10
def V(x,y):
    return np.power(x,2) + np.power(y,2) + u * x*y

N_iso = 5 # nombre d'isopotentielles à tracer
#

print(np.array([[2,2],[2,3]])[1,0])

print('é', np.array([  2*np.random.binomial(1,1/2) -1 for _ in range(10)]))


print(np.random.randint(2,size=1) )
print(5.2/1.6)


print(np.array([[0,1,2],[3,4,5]]),np.power(np.array([[0,1,2],[3,4,5]]), 2)   )

print(np.array([[0,1,2],[3,4,5]]), np.mean(np.array([[0,1,2],[3,4,5]]), axis=0))