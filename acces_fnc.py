import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv
from numba import jit
import pandas as pd

def create_force(N, dt, Tau_RT_x, Tau_RT_y, Tx, Ty):
    XI = np.zeros((2,N))
    i=0
    while i <= N:
        tau_x, direction_x = np.random.poisson(Tau_RT_x/dt), 2*np.random.binomial(1,1/2)-1
        dnx = int(tau_x)+1
        XI[0,i: i + min(dnx,N-i)] = direction_x * np.sqrt(8*Tx/Tau_RT_x) *dt
        i += dnx

    j=0
    while j <= N:
        tau_y, direction_y = np.random.poisson(Tau_RT_y/dt), 2*np.random.binomial(1,1/2)-1
        dny = int(tau_y)+1
        XI[1,j: j + min(dny,N-j)] = direction_y * np.sqrt(8*Ty/Tau_RT_x) *dt
        j += dny
    return XI

def isop(R_max,us, nb = 5):
    V0 = np.linspace(0.2,R_max, nb)
    dthet = 1e-2
    J_f = int(2*np.pi/dthet)

    for i in range(nb):
        ISO = np.array([ [ np.sqrt(2*V0[i]/(1-us*np.sin(2*j*dthet))) * np.cos(j*dthet) for j in range(J_f)  ] ,
                      [  np.sqrt(2*V0[i]/(1-us*np.sin(2*j*dthet))) * np.sin(j*dthet) for j in range(J_f)   ] ])
        #X2 = np.array([ [ np.sqrt(2*V0[i]/(1-0*us*np.sin(j*dthet))) * np.cos(j*dthet) for j in range(J_f)  ] ,
         #           [  np.sqrt(2*V0[i]/(1-0*us*np.sin(j*dthet))) * np.sin(j*dthet) for j in range(J_f)   ] ])
        plt.plot(ISO[0,:], ISO[1,:], color= 'b', linestyle='dashed')
        #plt.plot(X2[0,:], X2[1,:], color= 'b', linestyle='dashed')
    
def Tau_poisson(Tau_RT ):
    return np.random.poisson(Tau_RT)
     
def autocorr(x):
    result = np.correlate(x, x, mode='same')
    return result[result.size // 2:]


def omega(X, N_step, dt):
     #Calcul deltatheta et Omega
    deltatheta = np.copy(np.diff(   np.arctan(np.copy(X[1,10:N_step]),np.copy(X[0,10:N_step]))  ) )
    #deltatheta = np.remainder(deltatheta, np.pi)
    deltatheta[deltatheta<-np.pi] += 2*np.pi
    deltatheta[deltatheta>np.pi] += -2*np.pi
    omega = deltatheta/dt
    print(omega)
    P_omega = np.copy(omega)/len(omega)

    plt.figure()
    plt.hist(P_omega)
    plt.show()

    plt.figure()
    plt.plot(np.linspace(10*dt,dt*N_step, len(P_omega)), P_omega)
    plt.show()
    return omega, np.mean(omega), P_omega

def coo_h(X, t0,t_f,dt):
    plt.figure()
    plt.plot(np.arange(t0,t_f,dt), X[0,:])
    plt.plot(np.arange(t0,t_f,dt), X[1,:], color='r')
    plt.show()
    return

def traj(X, u):
    plt.figure()
    plt.scatter(X[0,:], X[1,:])
    isop(R_max=np.max(np.sqrt(np.power(X[0,:],2)+np.power(X[1,:],2))), us=u)
    plt.show()

def force(XI, t0,t_f,dt, Tx, Tau_RT_x):
    plt.figure()
    plt.plot(np.arange(t0,t_f,dt), XI[0,:])#, label='F_x' )
    plt.plot(np.arange(t0,t_f,dt), XI[1,:])#, label='F_y')
    plt.legend(['F_x','F_y'])
    plt.show()

    XI_x_var = autocorr(XI[0,:])
    print('Variance',XI_x_var[0], Tx/(2*Tau_RT_x))
    plt.figure()
    plt.plot(np.arange(0,len(XI_x_var)), XI_x_var)
    plt.show()