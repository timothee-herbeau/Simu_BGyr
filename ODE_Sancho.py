import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv
from numba import jit
import pandas as pd
import acces_fnc

from scipy.special import gamma
from scipy.integrate import solve_ivp

t0=0.1
tf=50
u=0.
gamma_0 = 1.5
omega_min,omega_max,N_om = 0,50,10
alpha = 1-u
v=1


# def f_mapped_om(t,y,om):
#         return  np.array([ y[1], -(2*gamma_0+1)*y[1]*alpha - y[0]*(alpha*v*om*np.exp(-alpha*t))**2   ])

def f_mapped_om(t,y,om):
    return np.array([ y[1], (1 - 2*gamma_0 - 2*t/alpha)*y[1]*alpha/(t**2) * np.exp(-alpha/t) - y[0]*(alpha*v*om*np.exp(-3*alpha/(2*t)) /t**2)**2   ])

#def tracage(omega_min = 1, omega_max = 40, N_om=5):
om = np.linspace(omega_min,omega_max,N_om) - omega_max/2
final_Phi = np.zeros((N_om))
plt.figure()
for k in range(len(om)):
    omg = om[k]
    sol = solve_ivp( f_mapped_om ,(t0,tf),args=(omg,),y0=np.array([1,0]))
    plt.plot(sol.t, sol.y[0,:], label=f'omega={omg}')
    plt.legend()
    #plt.yscale('log')
    final_Phi[k] = sol.y[0,-1]
plt.show()
#print(final_Phi)    
plt.figure()
plt.title('Phi(infty)')
plt.scatter(om,final_Phi) #we have now the value of the caractéristic function at phi infty as a function of omega 
plt.show()

# Tot_value = np.zeros((2*omega_max)) #semi continuous array with value of Phi. i.e. the FFT of IP
# for j in range(len(om)):
#      Tot_value[int(omega_max+om[j])] = final_Phi[j]

# plt.figure()
# plt.scatter(np.linspace(-omega_max,omega_max,2*omega_max), Tot_value)
# plt.title('Phi(omega)')
# plt.show()
dw = (om[1]-om[0])
P = np.fft.ifft(final_Phi)
X = np.linspace(-len(P)//2 ,len(P)//2 , len(P))

def P_analy(X):
    return np.where(np.abs(X)<v, np.power(np.power(v,2) -  np.power(X,2), -1 + 1/gamma_0 )* gamma(0.5 + gamma_0)/((np.power(v,-1+ 2*gamma_0))* gamma(0.5)*gamma(gamma_0)), 0)  

for x in X:
    print(x, P_analy(x*dw))
plt.plot(X, P)
plt.plot(X,P_analy(X))
plt.title('P(x)')
plt.show()

print('dw', (om[1]-om[0]))
# plt.figure()
# plt.plot(np.linspace(-len(P)//2,len(P)//2, len(P)), P_analy(np.linspace(-len(P)//2,len(P)//2,len(P))))
# plt.show()