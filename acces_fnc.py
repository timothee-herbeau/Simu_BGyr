import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv
from numba import jit
import pandas as pd


def create_force_2(N, dt, Tau_RT_x, Tau_RT_y, Fx, Fy):
    XI = np.zeros((2,N))
    bino,temps=2*np.random.randint(2, size=int(1.2*N*dt/Tau_RT_x))-1 ,np.random.exponential(scale=Tau_RT_x,size=int(1.2*N*dt/Tau_RT_x))
    binoy,tempsy=2*np.random.randint(2, size=int(1.2*N*dt/Tau_RT_y))-1 ,np.random.exponential(scale=Tau_RT_y,size=int(1.2*N*dt/Tau_RT_y))
    #calcul des instants cumulés dans le temps
    temps=np.cumsum(temps)
    tempsy=np.cumsum(tempsy)
    #conversion en temps discret
    tempsint=np.int64(temps/dt)
    tempsyint=np.int64(tempsy/dt)
    #on ne garde que les temps inférieure à n2 sauf le dernier
    tempsiint=tempsint[tempsint<N]
    tempsyiint=tempsyint[tempsyint<N]
    bino=bino[0:len(tempsiint)]
    binoy=binoy[0:len(tempsyiint)]
    XI[0,:tempsint[0]]=(2*np.random.randint(2)-1)*Fx
    XI[1,:tempsyint[0]]=(2*np.random.randint(2)-1)*Fy
    for ii in range(len(bino)-1):
        XI[0,tempsint[ii]:tempsint[ii+1]]=bino[ii]*Fx
    for ii in range(len(binoy)-1):
        XI[1,tempsyint[ii]:tempsyint[ii+1]]=binoy[ii]*Fy
    
    return XI


def create_force(N, dt, Tau_RT_x, Tau_RT_y, Fx, Fy):
    XI = np.zeros((2,N))
    i=0
    while i <= N:
        tau_x, direction_x = np.random.poisson(Tau_RT_x/dt), 2*np.random.randint(2,size=1) -1
        dnx = int(tau_x)
        XI[0,i: i + min(dnx,N-i)] = direction_x * Fx # np.sqrt(8*Tx/Tau_RT_x) *dt
        i += dnx

    j=0
    while j <= N:
        tau_y, direction_y = np.random.poisson(Tau_RT_y/dt), 2*np.random.randint(2,size=1)-1
        dny = int(tau_y)
        XI[1,j: j + min(dny,N-j)] = direction_y * Fy #np.sqrt(8*Ty/Tau_RT_x) *dt
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
    #print(omega)
    P_omega = np.copy(omega)/len(omega)

    # plt.figure()
    # plt.hist(P_omega)
    # plt.show()

    # plt.figure()
    # plt.plot(np.linspace(10*dt,dt*N_step, len(P_omega)), P_omega)
    # plt.show()
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

def force(XI, t0,t_f,dt, Fx, Tau_RT_x):
    plt.figure()
    plt.plot(np.arange(t0,t_f,dt), XI[0,:])#, label='F_x' )
    plt.plot(np.arange(t0,t_f,dt), XI[1,:])#, label='F_y')
    plt.legend(['F_x','F_y'])
    plt.show()

    XI_x_var = autocorr(XI[0,:])
    print('Variance',np.var(XI[0,:]), (Fx**2))
    plt.figure()
    plt.plot(np.arange(0,len(XI_x_var)), XI_x_var)
    plt.show()