#aims to determine u-dependency of exp(i w int1 + i w int1) 

import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv
from numba import jit
import pandas as pd
import acces_fnc
import scipy.integrate as integrate
from scipy.special import gamma, jv

N_step = 500
N_u = 30
dt = 1e-2

Fx = 1       
Fy = 2

Tau_RT_x = 1e-1
Tau_RT_y = 1e-1  

Tx = Fx**2 * Tau_RT_x
Ty = Fy**2 * Tau_RT_y

N_traj = 1000


def init_traj_randm(u, Tx=Tx, Ty=Ty, tau_x=Tau_RT_x, tau_y = Tau_RT_y, dt = dt, N=N_step):
    Fx,Fy = np.sqrt(Tx/tau_x), np.sqrt(Ty/tau_y)
    X = np.zeros((2, N))
    #création de la force:
    XI = acces_fnc.create_force_2(N, dt, tau_x, tau_y, Fx, Fy)    #Le _2 indique que j'utilise la méthode de Pascal
    return X,XI

def k_1(t,u):
    return np.exp(-(1+u)*(t))

def k_2(t,u):
    return np.exp(-(1-u)*(t))

def analytical_solution(u,omega, phi,v, gamma_0):
    nu = gamma_0/(1-u) - u/(2*(1-u)) - 1/2
    Z = omega*v/(1-u) * np.exp(-(1-u)*0/phi)
    return gamma(1+nu)*jv(nu,Z)/np.power(Z,nu)


def func(omega, Tx=Tx,Ty=Ty,tau_x=Tau_RT_x,tau_y=Tau_RT_y, N_traj=N_traj, N_step=N_step,N_u=N_u, Fx=Fx, Fy=Fy, dt=dt):
    u_tab = np.linspace(-0.5,0.5,N_u)
    Fx,Fy = np.sqrt(Tx/tau_x), np.sqrt(Ty/tau_y)
    E = np.zeros((N_u,N_traj),dtype='complex128') 
    E_1 = np.zeros((N_u,N_traj),dtype='complex128') 
    E_2 = np.zeros((N_u,N_traj),dtype='complex128') 
    analytc = np.zeros((N_u))
    for j in range(N_u):
        for k in range(N_traj):
            X,XI = init_traj_randm(u_tab[j],Tx,Ty,tau_x,tau_y, dt, N_step)
            time = np.linspace(0,int(N_step*dt),N_step)
            Q_1 = np.array([ k_1(N_step*dt - time,u_tab[j]) ])
            Q_2 = np.array([ k_2(N_step*dt - time,u_tab[j]) ])

            E[j,k] = np.exp(1j*omega/2*(   np.trapezoid(y=np.multiply(Q_1,XI[0,:]), x=time)  +  np.trapezoid(y=np.multiply(Q_2,XI[0,:]), x=time)  )   )
            E_1[j,k] = np.exp(1j*omega/2*(   np.trapezoid( y=np.multiply(Q_1,XI[0,:]) ,x=time)                                                           ))
            E_2[j,k] = np.exp(1j*omega/2*(                                                           np.trapezoid( y=np.multiply(Q_2,XI[0,:]) ,x=time)   ))
    plt.figure()
    plt.plot(u_tab, analytical_solution(u_tab,omega,phi=1e8,v=Fx,gamma_0=tau_x), label='Analytic')
    plt.plot(u_tab,np.mean(E,axis=1),label='full functionnal')
    plt.plot(u_tab,np.mean(E_1,axis=1),label='Negative kernel (1-u) only') 
    plt.plot(u_tab,np.mean(E_2,axis=1),label='Positive kernel (1+u) only')
    plt.legend()
    plt.xlabel('u')
    plt.ylabel('Functionnal of the two kernels')
    plt.show()
    return

func(omega=10)