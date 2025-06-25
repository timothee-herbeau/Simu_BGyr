import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv
from numba import jit
import pandas as pd
import acces_fnc

from scipy.integrate import solve_ivp

v = 1
omega = -5
u = 0.4
t0=0.01
tf=100
gamma=0.2

def f(t, y):
    return np.array([y[1]  , -y[1]*(1-2*gamma+u*np.tanh(u*t)) - y[0]*v*omega*np.cosh(u*t)**2 * np.exp(-2*t)*y[0]  ])
    
def f_mapped(t,y):
    return np.array([y[1]  , -y[1]*(2 - (1-2*gamma-u*np.tanh(u/t)))/(t) - y[0]*v*omega*np.cosh(u/t)**2 * np.exp(-2/t)/(t**4)  ])

# sol = solve_ivp(f_mapped,(t0,tf), np.array([1,0]))
# t,Phi = sol.t, sol.y[0,:]
# plt.figure()
# plt.plot(t,Phi)
# #plt.xscale('log')
# plt.show()

def f_mapped_om(t,y,om):
        return np.array([ y[1]  , -y[1]*(2 - (1-2*gamma-u*np.tanh(u/t)))/(t) - y[0]*v*om*np.cosh(u/t)**2 * np.exp(-2/t)/(t**4)  ])

omega_min,omega_max,N_om = 0.1,5,10

#def tracage(omega_min = 1, omega_max = 40, N_om=5):
om = np.linspace(omega_min,omega_max,N_om) - omega_max/2
final_Phi = np.zeros((N_om))
plt.figure()
for k in range(len(om)):
    omg = om[k]
    sol = solve_ivp( f_mapped_om ,(t0,tf),args=(omg,),y0=np.array([1,0]))
    plt.plot(sol.t, sol.y[0,:], label=f'omega={omg}')
    plt.legend()
    plt.yscale('log')
    final_Phi[k] = sol.y[0,-1]
plt.show()
print(final_Phi)    
plt.figure()
plt.scatter(om,final_Phi)
plt.show()

Tot_value = np.zeros((2*omega_max)) #semi continuous array with value of Phi. i.e. the FFT of IP
for j in range(len(om)):
     Tot_value[int(omega_max+om[j])] = final_Phi[j]

plt.figure()
plt.scatter(np.linspace(-omega_max,omega_max,2*omega_max), Tot_value)
plt.show()

P = np.fft.ifft(Tot_value)
plt.figure()
plt.plot(np.linspace(-len(P)//2,len(P)//2, len(P)), P)
plt.show()
