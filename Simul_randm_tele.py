import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv
from numba import jit
import pandas as pd
import acces_fnc

t_f = 100
t0 = 0
dt = 2e-3
N_step  = int((t_f - t0)/dt) 

u = 0.4
Fx = 1       #change les deux variances de la même façon
Fy = 3
gamma = 1e-1    # N.s.m-1 coeff frottement


Tau_RT_x = 5e-1
Tau_RT_y = 5e-1  #change les deux de la même façon

N_rep = 19


def init_traj_randm(u=u, Fx=Fx, Fy=Fy, N=N_step):
    
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
        X[:,k] = U@X[:,k-1] + XI[:,k-1]*dt   

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
    print('mean_omg',mean_omg, 'rad/s')

    #print(X)
    return P_omega, mean_omg, omega



#comparaison à la diffusion


def exact(t,T,tau):
    return 2*T*(t-tau*(1-np.exp(-t/tau)))

timeT = time.time()
r2 = np.zeros((N_rep,N_step))
X_f = np.zeros((N_rep,N_step))
OMG = np.zeros((N_rep,N_step))

for incr in range(N_rep):
    A,X,XI = init_traj_randm()
    X = dyna(A,X,XI)
    P_omega, mean_omg, omega = Analyze(X,XI)

    r2[incr,:] = np.power(X[0,:],2)+ np.power(X[1,:],2)
    X_f[incr,:] = X[0,:]
    OMG[incr,:len(omega)] = omega
    
r2_mean = np.mean(r2, axis=0)
X_mean = np.mean(X_f,axis=0)
Mean_OMG = np.mean(OMG, axis=0)

print('Took',time.time()-timeT,'s')

plt.figure()
plt.plot(np.linspace(0,N_step*dt,N_step),Mean_OMG)
plt.ylabel("$ \Omega $")
plt.xlabel('Time t')
plt.show()

plt.figure()
#plt.plot(np.linspace(0,N_step*dt,N_step),exact(np.linspace(0,N_step*dt,N_step),T= Tau_RT_x* (Fx)**2 , tau=Tau_RT_x))
plt.plot(np.linspace(0,N_step*dt,N_step),r2_mean)
plt.ylabel("$<r^2(t)>$")
plt.xlabel('Time t')
plt.show()

plt.figure()
plt.plot(np.linspace(0,N_step*dt,N_step),X_mean)
plt.title('x moyen')
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