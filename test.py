import numpy as np
from scipy.special import gamma
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

v, gamma_0 = 1, 0.4
def P_analy(X):
    return np.where(np.abs(X)<v, np.power(np.power(v,2) -  np.power(X,2), -1 + 1/gamma_0 )* gamma(0.5 + gamma_0)/((np.power(v,-1+ 2*gamma_0))* gamma(0.5)*gamma(gamma_0)) , -1)      
print(P_analy(3))


from scipy.special import gamma, jv

def J(nu, phi, omega):
    z = omega *v*np.exp(-alpha/phi) / alpha
    return gamma(1 + nu)*(2**(nu))*jv(nu,z)/np.power(z,nu)

alpha = 1-0.1
print(J(nu=gamma_0/(alpha) -1/2 , phi=100000, omega=15))

print(np.multiply(np.arange(3),np.arange(3)) )