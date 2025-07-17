#but: moyenner exp(i omega (Q_c*xi)(t) ) pour pouvoir comparer mes prédictions de transitions dynamiques sur le problème auxiliaire

import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv, jv, gamma, hyp2f1
from numba import jit
import pandas as pd
import acces_fnc

      

t_f = 220
t0 = 0
dt = 1e-2
N_step  = int((t_f - t0)/dt) 



def init_traj_randm(u, v,  N, tau_x, dt):
    Fx = v
    #print(Fx)
    X = np.zeros((2, N))
    #X[:,0] = np.random.normal(0,1,2)
    #création de la force:
    XI = acces_fnc.create_force(N, dt, tau_x, tau_x, Fx, Fx)    #Le _2 indique que j'utilise la méthode de Pascal
    return X,XI


@jit(nopython=True)
def dyna(X,XI,u,N=N_step, dt=dt):
    #Propagation de la dynamique
    L_tot = np.zeros((N_step-1))
    for k in range(1,N):

        X[:,k] =   X[:,k-1]* (1-dt) + dt*u *X[::-1,k-1] + XI[:,k-1]*dt   #plus rapide pour numba
        
        #dx, dy = X[0,k]- X[0,k-1] , X[1,k]- X[1,k-1]
        #L_tot[k-1] = X[0,k]* dy/dt -dx/dt * X[1,k]        # x*ypoint - y xpoint  : moment cinétique calculé "exactement" à t 

    return X  
        

@jit(nopython=True)
def Q_c(t,u):
    return np.exp(-t)*np.cosh(u*t)


def exact_bess(gamma_0,u,v,omg):
    alpha = 2*gamma_0/(1-u)
    #nu = (2*gamma_0-1)/(2-2*u)
    nu = gamma_0/(1-u) - u/(2*(1-u)) - 1/2
    z = v*np.abs(omg)/(1-u)
    return jv(nu,z)/np.power(z,nu) * np.power(2,nu)* gamma(1+nu)


def analytical_solution(gamma_0, u,v,omega):
    nu = gamma_0/(1-u) - u/(2*(1-u)) - 1/2
    Z = np.abs(omega)*v/(1-u)  
    return    gamma(1+nu)*jv(nu,Z)/np.power(Z/2,nu)


def inte_QC(XI,time,u,dt):
    return np.sum(np.multiply( Q_c(time[-1] - time ,u),XI ))*dt


def comp(N_try,N_step,N_om ,omg_max,N_gamma,N_add ,u,tau,v,t0=t0,tf=t_f):
    dt = (tf-t0)/N_step
    
    arr_time = np.linspace(t0,tf,N_step)
    
    
    P_forg = np.zeros((N_gamma,1+2*N_om+2*N_add), dtype='complex')
    dg = (1.9/N_gamma)
    t_init = time.time()
    gm_tab = np.array([ 2.4 + dg*(j) for j in range(N_gamma)])
    omg_max_tab = np.array([2*omg_max - j*omg_max/N_gamma for j in range(N_gamma)])
    for j in range(N_gamma):
        Expectation = np.zeros((N_try,2*N_om+1),dtype='complex')
        omg_max_gm = omg_max_tab[j]
        om = np.linspace(-omg_max_gm, omg_max_gm, 2*N_om+1)
        
        tau_j = 1/gm_tab[j]
        for k in range(N_try):
            X,XI = init_traj_randm(u=u, v=v,  N=N_step, tau_x=tau_j, dt=dt)
            Expectation[k,:] =   1j * inte_QC(XI[0,:],arr_time,u,dt) *np.ones((2*N_om +1),dtype='complex')  

        Expectation = np.exp(np.multiply(om,Expectation) )
        Phi = np.mean(Expectation,0)        
        
        extended_phi = np.concatenate((np.zeros((N_add)), Phi, np.zeros((N_add))),dtype='complex')
        final_Phi_reordered = np.zeros((len(extended_phi)),dtype='complex')      
        final_Phi_reordered[len(extended_phi)//2 +1:] = extended_phi[: len(extended_phi)//2]
        final_Phi_reordered[:len(extended_phi)//2 +1 ] = extended_phi[ len(extended_phi)//2 : ]
        #plt.scatter(np.concatenate((np.zeros(N_add),om,np.zeros(N_add))),extended_phi)
         
        dw = om[1]-om[0]
        #plt.scatter(np.concatenate((np.array([-omg_max_gm + (k-N_add)*dw for k in range(N_add)]),om, np.array([omg_max_gm + (k)*dw for k in range(N_add)]))),extended_phi)
        #plt.plot(np.concatenate((np.array([-omg_max_gm + (k-N_add)*dw for k in range(N_add)]),om, np.array([omg_max_gm + (k)*dw for k in range(N_add)]))), exact_bess(gm_tab[j],u,v,np.concatenate((np.array([-omg_max_gm + (k-N_add)*dw for k in range(N_add)]),om, np.array([omg_max_gm + (k)*dw for k in range(N_add)])))))
        # plt.show()
        # plt.figure()
        om_extended = np.concatenate((np.array([-omg_max_gm + (k-N_add)*dw for k in range(N_add)]),om, np.array([omg_max_gm + (k)*dw for k in range(N_add)]) )) 
        Y_0 = exact_bess( gm_tab[j],u,v,om_extended )
        Y_om = np.where(np.isnan(Y_0)==True ,1, Y_0 )
        print(time.time() - t_init)
        plt.figure()
        plt.scatter(om_extended, extended_phi)
        plt.plot(om_extended, extended_phi)
        plt.show()

        P = np.fft.fft(final_Phi_reordered,norm='forward')
        #print(Y_om)
        #print(np.sum(P))
        #P_forg[j,:] = P
        #polyfit
        

        plt.figure()
        X = np.linspace(-len(P) ,len(P) , 2*len(P))
        P_rem = np.where(np.concatenate((P[::1],P))>0,np.concatenate((P[::1],P)), 0 )
        X_center = X[len(X)//2 - 9:len(X)//2 +9]
        
        arr = np.polyfit(X_center/(2*np.pi)*dw, P_rem[len(X)//2 - 9:len(X)//2 +9], deg=2)
        p = np.poly1d(arr)
        print(arr[0])#, X_center, p(X_center))
        plt.scatter(X_center/(2*np.pi)*dw, p(X_center/(2*np.pi)*dw))

        plt.scatter(X[::1]/(2*np.pi)*dw , P_rem, label=f'tau = {tau_j}, gm={gm_tab[j]}')
        plt.show()
    #plt.yscale('log')
    #plt.scatter(X[::1] , np.where(np.concatenate((P[::1],P))>0,np.concatenate((P[::1],P)), 0 ))#,label= f'Solved Probability at gamma = {np.round(gm_tab[k],2)} with MSE={np.round( MSE(P_analy( X*2*np.pi/(len(final_Phi_reordered)*dw) , gamma_0=tab_g[k])*2*np.pi/(len(final_Phi_reordered)*dw), np.where(np.concatenate((P[::1],P))>0,np.concatenate((P[::1],P)), 0 ))   ,7)} ',marker = 'x' )
    plt.show()
    return 


comp(N_try=7000, N_step=N_step, N_om = 311,omg_max=5,N_gamma=1,N_add =0,u=0.3,tau=0.4,v=3,t0=t0,tf=t_f)


def trace_bessel():
    plt.figure()
    for k in range(15):
        x = np.linspace(0,20,50)
        nu = 1/2 -k/4
        plt.plot(x, jv(nu,x),label=f'nu={nu}')
    plt.legend()
    plt.yscale('symlog')
    plt.show()
    return
#trace_bessel()