import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv
from numba import jit
import pandas as pd

import acces_fnc

t_f = 100
t0 = 0
dt = 1e-2
N_step  = int((t_f - t0)/dt) 

u = 0.4
Tx = 1e-1       #change les deux variances de la même façon
Ty = 1e-1
gamma = 1e-1 # N.s.m-1 coeff frottement


Tau_RT_x = 1e-1
Tau_RT_y = 2e-1   #change mes deux de la même façon

N_rep = 10

#@jit

def traj_randm(u=u, Tx=Tx, Ty=Ty, N=N_step):
    
    #def Matrice dynamique A et position X:
    A = np.array([[1, -u],
                  [-u, 1]]) 
    X = np.zeros((2, N))
    
    #initialisation
    X[1,0] = 1/2


    #création de la force:
    XI = acces_fnc.create_force(N, dt, Tau_RT_x, Tau_RT_y, Tx, Ty)
    #print(' Mean Force',np.mean(XI, axis=1) )


   
    #Propagation de la dynamique
    for k in range(1,N_step):
        U = np.identity(2)- A*dt
        X[:,k] = X[:,k-1] - X[:,k-1]*dt  + XI[:,k-1] 
        

    #Affiche les coordonnées horaires
    #acces_fnc.coo_h(X, t0,t_f,dt)

    #Affiche la trajectoire
    #acces_fnc.traj(X, u)

    #Affiche la force
    #acces_fnc.force(XI, t0,t_f,dt, Tx, Tau_RT_x)


    #df = pd.DataFrame(X)
    #df.to_csv('your_file_name.csv', index=False)

    #Calcule la vitesse angulaire
    omega ,mean_omg, P_omega = acces_fnc.omega(X, N_step, dt) 
    print('mean_omg',mean_omg, 'rad/s')


    #print(X)
    return X, P_omega, mean_omg

Res = np.zeros((1,N_rep))

for incr in range(N_rep):

    X, P_omega, mean_omg = traj_randm()
    Res[0,incr] = mean_omg

plt.figure()
plt.hist(Res)
plt.show()

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