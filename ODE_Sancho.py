#code to verify the use of the ODE solver + inversion to probaility density. We aim to find Sancho's results

import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv
from numba import jit
import pandas as pd
import acces_fnc
import time
from scipy.special import gamma, jv
from scipy.integrate import solve_ivp

t0=0.01
tf=200
u=0.1  #coupling parameter
gamma_0 = 1.5 #transition rate
omega_min,omega_max,N_om = 0, 20 ,151 #N_om must be odd so as to comply with fft rcquirements
N_add = 25 # number of omega component we add to finish the spectrum
alpha = 1-u
v=5   #intensity of the random telegraph signal

t_init = time.time()

def f_mapped_om(t,y,om):
    return np.array([ y[1], (alpha - 2*gamma_0 - 2*t)*y[1]/np.power(t,2) - y[0]*np.power(v*om*np.exp(-alpha/(t)) /np.power(t,2) ,2)   ])

def f_mapped_QCkernel(t,y,om):
    return np.array([ y[1], (1 - 2*gamma_0 - 2*t + u/np.tanh(u/t))*y[1]/np.power(t,2) - y[0]*np.power(v*om*np.exp(-1/(t))*np.sinh(u/t) /np.power(t,2) ,2)   ])


def J(nu, phi, omega):
    z = np.abs(omega) *v*np.exp(-alpha/phi) / alpha
    return gamma(1 + nu)*(2**(nu))*jv(nu,z)/np.power(z,nu)

om = np.linspace(-omega_max,omega_max,N_om)  
print('om is of shape',len(om))
final_Phi = np.zeros((N_om))
plt.figure()

for k in range(len(om)):
    omg = om[k]
    sol = solve_ivp( f_mapped_om ,(t0,tf),args=(omg,),y0=np.array([1,0]))
    plt.plot(sol.t, sol.y[0,:], label=f'omega={omg}')
    plt.scatter(sol.t, J(nu=gamma_0/(alpha) -1/2 , phi=sol.t, omega=omg), label=f'{omg} for bessel')
    plt.legend()
    # plt.yscale('log')
    final_Phi[k] =sol.y[0,-1] 
    #print('la',omg,J(nu=gamma_0/(alpha) - 1/2 , phi=100000*tf, omega=omg))

print('time',time.time()-t_init)

plt.show()   
plt.figure()
plt.title('Phi_infty(omega))')
plt.scatter(om,final_Phi) #we have now the value of the caractéristic function at phi infty as a function of omega 
plt.show()

dw = (om[1]-om[0])

extended_phi = np.concatenate((np.zeros((N_add)), final_Phi, np.zeros((N_add))))
final_Phi_reordered = np.zeros((len(extended_phi)))      
final_Phi_reordered[len(extended_phi)//2 +1:] = extended_phi[: len(extended_phi)//2]
final_Phi_reordered[:len(extended_phi)//2 +1 ] = extended_phi[ len(extended_phi)//2 : ]


P = np.fft.fft(final_Phi_reordered, norm='forward')
X = np.linspace(-len(P) ,len(P) , 2*len(P))


def P_analy(X):
    return np.where((np.abs(X)<v)&(np.power(v,2) -  np.power(X,2)!=0),        np.power(np.power(v,2) -  np.power(X,2), gamma_0 - 1 )* gamma(0.5 + gamma_0)/((np.power(v,-1+ 2*gamma_0))* gamma(0.5)*gamma(gamma_0))         , 0)  

print('sumP_analytc', 1/(len(final_Phi_reordered)*dw),np.sum(P_analy(X*2*np.pi/(len(final_Phi_reordered)*dw)))*2*np.pi/(len(final_Phi_reordered)*dw), 'computed P norm',np.sum(P), 'dw', dw) 
    
plt.figure()
plt.title('Phi_infty(omega))')
plt.scatter(np.concatenate((np.array([-omega_max - dw*(N_add-j) for j in range(N_add)]),om,np.array([omega_max + dw*j for j in range(N_add)]))),extended_phi) #we have now the value of the caractéristic function at phi infty as a function of omega 
plt.show()

plt.figure()
plt.plot(X[::1] , np.concatenate((P[::1],P)) )
plt.plot(X ,P_analy(X*2*np.pi/(len(final_Phi_reordered)*dw))*2*np.pi/(len(final_Phi_reordered)*dw) )
plt.title('P(x)')
plt.show()