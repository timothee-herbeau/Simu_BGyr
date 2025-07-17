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
u=0.0  #coupling parameter
gamma_0 = 1.5 #transition rate
omega_min,omega_max,N_om = 0, 20 ,191 #N_om must be odd so as to comply with fft rcquirements
N_add = 25 # number of omega component we add to finish the spectrum
alpha = 1-u
v=5   #intensity of the random telegraph signal

t_init = time.time()

def f_mapped_om(t,y,om,u,gm):
    return np.array([ y[1], (1-u - 2*gm - 2*t)*y[1]/np.power(t,2) - y[0]*np.power(v*om*np.exp(-(1-u)/(t)) /np.power(t,2) ,2)   ])

def f_mapped_QCkernel(t,y,om):
    return np.array([ y[1], (1 - 2*gamma_0 - 2*t + u/np.tanh(u/t))*y[1]/np.power(t,2) - y[0]*np.power(v*om*np.exp(-1/(t))*np.sinh(u/t) /np.power(t,2) ,2)   ])


def J(nu, phi, omega):
    z = np.abs(omega) *v*np.exp(-alpha/phi) / alpha
    return gamma(1 + nu)*(2**(nu))*jv(nu,z)/np.power(z,nu)

def Solveur(omega_max,N_om,u,gm, t0 = t0, tf=tf):
    om = np.linspace(-omega_max,omega_max,N_om)  
    print('om is of shape',len(om))
    final_Phi = np.zeros((N_om))
    #plt.figure()

    for k in range(len(om)):
        omg = om[k]
        sol = solve_ivp( f_mapped_om ,(t0,tf),args=(omg,u,gm),y0=np.array([1,0]))
        #plt.plot(sol.t, sol.y[0,:], label=f'omega={omg}')
        #plt.scatter(sol.t, J(nu=gamma_0/(alpha) -1/2 , phi=sol.t, omega=omg), label=f'{omg} for bessel')
        #plt.legend()
        # plt.yscale('log')
        final_Phi[k] = sol.y[0,-1] 
        #print('la',omg,J(nu=gamma_0/(alpha) - 1/2 , phi=100000*tf, omega=omg))
        
    print('time',time.time()-t_init)
    return om,final_Phi

def MSE(a,b):
    if np.shape(a)==np.shape(b):
        return np.real(np.mean(np.power(a-b,2)))
    return -1

def relativ(a,b):
    if np.shape(a)==np.shape(b):
        return np.abs(np.mean((a-b)/ b ))

    return -1

def P_analy(X,gamma_0): #P sancho
    return np.where((np.abs(X)<v)&(np.power(v,2) -  np.power(X,2)!=0),        np.power(np.power(v,2) -  np.power(X,2), gamma_0 - 1 )* gamma(0.5 + gamma_0)/((np.power(v,-1+ 2*gamma_0))* gamma(0.5)*gamma(gamma_0))         , 0)  

def several_gamma(N_g,u=u,omega_max=omega_max, N_om =N_om,N_add=N_add):
    
    final_Phi_forg = np.zeros((N_g, N_om))
    P_forg = np.zeros((N_g, N_om+2*N_add))
    dw_forg = np.zeros((N_g))
    tab_g = np.linspace(0.5,1.4,N_g)
    plt.figure()
    for k in range(N_g):
        g_j = tab_g[k]
        alpha = 1-u
        om,final_Phi = Solveur(omega_max=omega_max, N_om=N_om,u=u, gm=g_j)
        final_Phi_forg[k,:] = final_Phi
       
        om_nonzeros = np.concatenate((om[:N_om//2], om[1+N_om//2:]))
        #print(om_nonzeros)
        plt.scatter(om,final_Phi, label=f'Numerical solution at gamma={np.round(g_j,2)}, MSE={np.round(relativ(np.concatenate((final_Phi[:N_om//2], final_Phi[1+N_om//2:])), J(nu=g_j/(alpha) -1/2 ,phi=1e8,omega=om_nonzeros ) ),2)} ',marker = 'x') #we have now the value of the caractéristic function at phi infty as a function of omega 
        plt.plot(om_nonzeros, J(nu=g_j/(alpha) -1/2 ,phi=1e8,omega=om_nonzeros ), label=f'Analytical solution at gamma={np.round(g_j,2)}' )
        dw_forg[k] = om[1]-om[0]
    plt.legend()
    plt.title(r'$\Phi(\omega)$ at $\varphi = \infty$')
    plt.show()

    
    plt.figure()
    for k in range(N_g):
        extended_phi = np.concatenate((np.zeros((N_add)), final_Phi_forg[k,:], np.zeros((N_add))))
        final_Phi_reordered = np.zeros((len(extended_phi)))      
        final_Phi_reordered[len(extended_phi)//2 +1:] = extended_phi[: len(extended_phi)//2]
        final_Phi_reordered[:len(extended_phi)//2 +1 ] = extended_phi[ len(extended_phi)//2 : ]


        P = np.fft.fft(final_Phi_reordered, norm='forward')
        P_forg[k,:] = P
        X = np.linspace(-len(P) ,len(P) , 2*len(P))
        #tracé relevant only at u=0 pour comparer à Sancho.
        
        dw = dw_forg[k]
        plt.scatter(X[::1] , np.where(np.concatenate((P[::1],P))>0,np.concatenate((P[::1],P)), 0 ),label= f'Solved Probability at gamma = {np.round(tab_g[k],2)} with MSE={np.round( MSE(P_analy( X*2*np.pi/(len(final_Phi_reordered)*dw) , gamma_0=tab_g[k])*2*np.pi/(len(final_Phi_reordered)*dw), np.where(np.concatenate((P[::1],P))>0,np.concatenate((P[::1],P)), 0 ))   ,7)} ',marker = 'x' )
        plt.plot(X ,P_analy( X*2*np.pi/(len(final_Phi_reordered)*dw) , gamma_0=tab_g[k])*2*np.pi/(len(final_Phi_reordered)*dw), label=f"Sancho's 1D probability at gamma = {np.round(tab_g[k],2)}" )
        #print('sumP_analytc', 1/(len(final_Phi_reordered)*dw),np.sum(P_analy(X*2*np.pi/(len(final_Phi_reordered)*dw)))*2*np.pi/(len(final_Phi_reordered)*dw), 'computed P norm',np.sum(P), 'dw', dw) 
    plt.legend()
    plt.yscale('log')
    plt.title(r'$\mathbb{P}(x)$ in uncoupled case')
    plt.show()
    return

        
    # plt.figure()
    # plt.title('Phi_infty(omega))')
    # plt.scatter(np.concatenate((np.array([-omega_max - dw*(N_add-j) for j in range(N_add)]),om,np.array([omega_max + dw*j for j in range(N_add)]))),extended_phi) #we have now the value of the caractéristic function at phi infty as a function of omega 
    # plt.show()
several_gamma(N_g=5)