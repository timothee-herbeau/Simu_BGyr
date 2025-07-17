import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv
from numba import jit
import pandas as pd
import acces_fnc
#from sklearn.metrics import mean_squared_error

t_f = 30
t0 = 0
dt = 5e-3
N_step  = int((t_f - t0)/dt) 

u = 0.4 #should be useless

Fx = 1       #change les deux variances de la même façon
Fy = 1
gamma = 1e-1    # N.s.m-1 coeff frottement


Tau_RT_x = 0.5
Tau_RT_y = 0.5  #change les deux de la même façon

N_traj = 3000


def init_traj_randm(u=u, Fx=Fx, Fy=Fy, N=N_step,Tau_RT_x=Tau_RT_x, Tau_RT_y=Tau_RT_y):
    
    #def Matrice dynamique A et position X:
    A = np.array([[1, -u],
                  [-u, 1]]) 
    X = np.zeros((2, N))
    #initialisation
    #X[1,0] = 1/2
    #création de la force:
    XI = acces_fnc.create_force(N, dt, Tau_RT_x, Tau_RT_y, Fx, Fy)
    #print(' Mean Force',np.mean(XI, axis=1) )
    return A,X,XI


@jit(nopython=True)
def dyna(A,X,XI,N=N_step, dt=dt):
    #Propagation de la dynamique
    for k in range(1,N):
        U = np.identity(2)- A*dt
        X[:,k] =   X[:,k-1]   + XI[:,k-1]*dt      # OU X[:,k-1]* (1-dt) - dt*u *X[::-1,k-1] + XI[:,k-1]*dt   #plus rapide pour numba 
    return X
        

def Analyze(X,XI,N=N_step, Fx=Fx, Fy=Fy, Tau_RT_X = Tau_RT_x):
    #Affiche les coordonnées horaires
    #acces_fnc.coo_h(X, t0,t_f,dt)

    #Affiche la trajectoire
    #acces_fnc.traj(X, u)

    #Affiche la force
    #acces_fnc.force(XI, t0,t_f,dt, Fx, Tau_RT_x)

    #df = pd.DataFrame(X)
    #df.to_csv('your_file_name.csv', index=False)

    #Calcule la vitesse angulaire
    omega ,mean_omg, P_omega = acces_fnc.omega(X, N, dt) 
    #print('mean_omg',mean_omg, 'rad/s')

    #print(X)
    return P_omega, mean_omg, omega


#comparaison à la diffusion


def exact(t,T,tau):
    return 2*T*(t-tau*(1-np.exp(-t/tau)))

def comp(N):
    timeT = time.time()
    plt.figure()
    v_and_tau = np.array([[1,5e-1],[1,1.5], [4,5e-1], [4,1.5] ])
    for k in range(N):
        v,tau = v_and_tau[k,0], v_and_tau[k,1]
        Force = np.zeros((N_traj))
        X2 = np.zeros((N_traj,N_step))
        X_f = np.zeros((N_traj,N_step))
        Var = np.zeros((N_step))
        OMG = np.zeros((N_traj,N_step))
        tab_mean_omega = np.zeros((N_traj))
        for incr in range(N_traj):
            A,X,XI = init_traj_randm(u=u, Fx=v, Fy=v, N=N_step,Tau_RT_x=tau,Tau_RT_y=tau)
            X = dyna(A,X,XI)
            P_omega, mean_omg, omega = Analyze(X,XI)
            Force[incr] = np.mean(XI[0,:])

            X2[incr,:] = np.power(X[0,:],2) *0.5    + 0.5*np.power(X[1,:],2)
            
            X_f[incr,:] = X[0,:]
            OMG[incr,:len(omega)] = omega
            tab_mean_omega[incr] = mean_omg 
            
        Var[:] = np.var(X_f, axis=0)
        X2_mean = np.mean(X2, axis=0)
        X_mean = np.mean(X_f,axis=0)

        Temp = tau* (v)**2

        plt.scatter(np.linspace(1*dt,N_step*dt,N_step-1),X2_mean[1:], label=f'Numerical $<r^2(t)>$, $v$ = {v} and $tau$ = {tau}')
        plt.plot(np.linspace(1*dt,N_step*dt,N_step-1),exact(np.linspace(1*dt,N_step*dt,N_step-1), T= Temp , tau=tau), label=f'Analytic <r^2(t)>', color='red')
        Relativ_err = np.abs(np.mean((X2_mean[10:] - exact(np.linspace(10*dt,N_step*dt,N_step-10), T= Temp , tau=tau) )/ exact(np.linspace(dt*10,N_step*dt,N_step-10), T= Temp , tau=tau) ))
        
        #Mean_OMG = np.mean(OMG, axis=0)

    print('Took',time.time()-timeT,'s')

    plt.ylabel("$<x^2(t)>$")
    plt.xlabel('Time t')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title(f'Relative error for numerical vs analytical solution is : {np.round(100*Relativ_err,1)} %')
    plt.show()


    # plt.figure()
    # plt.plot(np.linspace(0,N_step*dt,N_step),Mean_OMG)
    # plt.ylabel("$ \Omega $")
    # plt.xlabel('Time t')
    # plt.show()

    # plt.figure()
    # plt.hist(tab_mean_omega)
    # plt.show()

    plt.figure()
    plt.plot(np.linspace(dt,N_step*dt,N_step-1),exact(np.linspace(dt,N_step*dt,N_step-1), T= Temp , tau=Tau_RT_x), label='Analytic <x^2(t)>')
    plt.plot(np.linspace(0,N_step*dt,N_step),X2_mean, label='Numerical <x^2(t)>')
    Relativ_err = np.abs(np.mean((X2_mean[1:] - exact(np.linspace(dt,N_step*dt,N_step-1), T= Temp , tau=Tau_RT_x) )/ exact(np.linspace(dt,N_step*dt,N_step-1), T=Temp , tau=Tau_RT_x) ))
    #plt.plot(np.linspace(0,N_step*dt,N_step), Var)
    plt.ylabel("$<x^2(t)>$")
    plt.xlabel('Time t')
    plt.legend()
    plt.title(f'Relative error for numerical vs analytical solution is : {np.round(100*Relativ_err,1)} %')
    plt.show()


    plt.figure()
    plt.scatter(np.linspace(1*dt,N_step*dt,N_step-1),X2_mean[1:], label='Numerical <x^2(t)>')
    plt.plot(np.linspace(1*dt,N_step*dt,N_step-1),exact(np.linspace(1*dt,N_step*dt,N_step-1), T= Temp , tau=Tau_RT_x), label='Analytic <r^2(t)>', color='red')
    Relativ_err = np.abs(np.mean((X2_mean[10:] - exact(np.linspace(10*dt,N_step*dt,N_step-10), T= Temp , tau=Tau_RT_x) )/ exact(np.linspace(dt*10,N_step*dt,N_step-10), T= Temp , tau=Tau_RT_x) ))
    plt.ylabel("$<x^2(t)>$")
    plt.xlabel('Time t')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title(f'Relative error for numerical vs analytical solution is : {np.round(100*Relativ_err,1)} %')
    plt.show()



    plt.figure()
    plt.plot(np.linspace(0,N_step*dt,N_step),X_mean, label='Analytic <x^2(t)>')
    plt.show()
    print('moyen', np.mean(X_mean))
    print('force mean tot', np.mean(Force))
    print(X2_mean[-10:],exact(np.linspace(0,N_step*dt,N_step), T= Temp , tau=Tau_RT_x)[-10:] )
    return 

comp(4)



# plt.figure()
# plt.plot(np.linspace(0,N_step*dt,N_step),X_mean)
# plt.title('x moyen')
# plt.show()





    # r2=r[1000:n2,0]**2+r[1000:n2,1]**2
    # r2b=(r[1000:n2,0]**2-r[1000:n2,1]**2)
    # rtrunc=r[1000:n2]
    # deltaangbis=0.5*(r2[:-1]+r2[1:])*deltatheta
    # deltaang=iu*r2b+np.sqrt(2*Ty)*gny_sample[1000:n2]*rtrunc[:,0]-np.sqrt(2*Tx)*gnx_sample[1000:n2]*rtrunc[:,1]
    # deltathetabis=deltaang/r2

  #Autre façon de générer la force, à Trois états, -1, 0 et 1
    # i=0
    # while i <= N:
    #     tau_x, direction_x = np.random.poisson(Tau_RT_x/dt), 2*np.random.binomial(1,1/2)-1
    #     dnx = int(tau_x)+1
    #     XI[0,i: i + min(dnx,N-i)] = direction_x * np.sqrt(Tx*dt/8) * (1 - (XI[0,i-1]>0) )
    #     i += dnx
        #print(i,tau_x, dnx, len(XI[0,:]))

    # j=0
    # while j <= N:
    #     tau_y, direction_y = np.random.poisson(Tau_RT_y/dt), 2*np.random.binomial(1,1/2)-1
    #     dny = int(tau_y )+1
    #     #print(dny,tau_y)
    #     XI[1, j : j + min(dny,N-j)] = direction_y * np.sqrt(Ty*dt/8)  * (1 - (XI[1,j-1]>0) )
    #     j += dny