import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv
from numba import jit
import pandas as pd
import acces_fnc


t_f = 50
t0 = 0
dt = 2e-3
N_step  = int((t_f - t0)/dt) 

u = 0.3
Fx = 1       
Fy = 2

Tau_RT_x = 9e-1
Tau_RT_y = 1e-1  

Tx = Fx**2 * Tau_RT_x
Ty = Fy**2 * Tau_RT_y

N_traj = 2000

def init_traj_randm(u=u, Fx=Fx, Fy=Fy, N=N_step):
    X = np.zeros((2, N))
    #création de la force:
    XI = acces_fnc.create_force_2(N, dt, Tau_RT_x, Tau_RT_y, Fx, Fy)    #Le _2 indique que j'utilise ta méthode
    return X,XI


@jit(nopython=True)
def dyna(X,XI,N=N_step, dt=dt):
    #Propagation de la dynamique
    L_tot = np.zeros((N_step-1))
    for k in range(1,N):

        X[:,k] =   X[:,k-1]* (1-dt) + dt*u *X[::-1,k-1] + XI[:,k-1]*dt   #plus rapide pour numba
        
        dx, dy = X[0,k]- X[0,k-1] , X[1,k]- X[1,k-1]
        L_tot[k-1] =  -dx/dt * X[1,k] + X[0,k]* dy/dt       # x ypoint - y xpoint  

    return X, L_tot
        

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

def L(X2, Y2,u):
    return u*(X2 - Y2)

def A(u, tau):  #utile pour le second moment de la position en régime stationnaire
    return (2 - u**2 + tau)/(2*(1-u**2)*(1+2*tau + (1-u**2)*tau**2))
def B(u,tau):
    return u**2 * (1 + tau)/(2*(1-u**2)*(1+2*tau + (1-u**2)*tau**2))

def Var_analy(u, Tau_RT_x, Tau_RT_y, Tx, Ty):
    return np.array([ A(u,Tau_RT_x)*Tx + B(u,Tau_RT_y)*Ty,  B(u,Tau_RT_x)*Tx + A(u,Tau_RT_y)*Ty ])


timeT = time.time()

X2 = np.zeros((N_traj,N_step))
Y2 = np.zeros((N_traj,N_step))
OM = np.zeros((N_traj,N_step,2))
OMG = np.zeros((N_traj,N_step-1))       #regroupe les valeurs instantanée des Omegas pour chaque trajectoire
tab_mean_omega = np.zeros((N_traj))     #regroupe le oméga moyen pour chaque trajectoire

L_analytic =  np.zeros((N_traj,N_step))
L_numeric = np.zeros((N_traj,N_step-1))
L_numeric2 = np.zeros((N_traj,N_step-1))

for incr in range(N_traj):
    X,XI = init_traj_randm()
    X, L_numeric[incr,:] = dyna(X,XI)
    OM[incr,:,0], OM[incr,:,1] = X[0,:], X[1,:]
    P_omega, mean_omg, omega = Analyze(X,XI)
    X2[incr,:] = np.power(X[0,:],2)
    Y2[incr,:] = np.power(X[1,:],2)
    L_numeric2[incr,:] = np.multiply(omega,X2[incr,:-1]+Y2[incr,:-1] )      #L = omega * r^2
    L_analytic[incr,:] = L(X2[incr,:], Y2[incr,:],u)                        #D'après les notes de Gleb

    OMG[incr,:] = omega
    tab_mean_omega[incr] = mean_omg 
    

r2_mean = np.mean(X2+Y2, axis=0)
X2_analy = Var_analy(u, Tau_RT_x, Tau_RT_y, Tx, Ty)[0]
Y2_analy = Var_analy(u, Tau_RT_x, Tau_RT_y, Tx, Ty)[1]
R2_anal = X2_analy + Y2_analy

L_analytic2 = np.array([u*(1+1/Tau_RT_x)*(Fx**2 - Fy**2)/((1 + 1/Tau_RT_x - u)*(1+ 1/Tau_RT_x + u)) ])


print('Took',time.time()-timeT,'s')


# plt.figure()
# plt.plot(np.linspace(0,dt*N_step,N_step), r2_mean)
# plt.plot(np.linspace(0,dt*N_step,N_step), R2_anal*np.ones_like(r2_mean))
# plt.title('r2')
# plt.show()

# plt.figure()
# plt.plot(np.linspace(0,dt*(N_step-11),N_step-11), np.mean(OMG[:,10:],axis=0))
# plt.title('Omega intantané moyenné sur les traj')
# plt.show()

# plt.figure()
# plt.hist(tab_mean_omega,density=True, bins=12)
# plt.title('Occurrence des omega moyen')
# plt.show()
# print('Mean Omega',np.mean(tab_mean_omega))


# #variance analytic vs numérique
# plt.figure()
# plt.title('R2 distribution')
# plt.hist(np.reshape(X2[:,2000:] + Y2[:,2000:],(-1)), label='numeric', alpha=0.6, bins=60, density = 1)
# plt.hist(X2_analy + Y2_analy, label='analytic', alpha=0.6, bins=60, density=0)
# plt.hist(np.mean(np.reshape(X2[:,2000:]+ Y2[:,2000:],(-1))), label='moyenne numérique', alpha=0.6, bins=60, density=0)
# plt.legend()
# plt.show()


# ##L analytic vs numérique
# plt.figure()
# plt.title('Analytic vs numeric L')
# plt.hist(np.reshape(L_numeric,(-1)), label='numeric', alpha=0.6, bins=40, density=1)
# plt.hist(np.reshape(L_analytic,(-1)), label='analytic',alpha=0.7, bins=40, density=1)#) 
# plt.hist(np.reshape(L_analytic2,(-1)), label='analytic2',alpha=0.7, bins=30, density=0)#) 
# plt.hist(np.reshape(L_numeric2,(-1)), label='numeric2',alpha=0.6, bins=60, density=1)#) 
# plt.legend()
# plt.show()

print('<L> numeric', np.mean(L_numeric), '<L> semi-analy', np.mean(L_analytic), 'True L', L_analytic2)




## Probability density function
plt.figure()
plt.hist2d(np.reshape(OM[:,:,0],(-1)) , np.reshape(OM[:,:,1],(-1)) , bins=(80,80) )
plt.show()

Abcss_x = np.sort(np.reshape(OM[:,:,0], (-1)))
Abcss_x_trunc = np.round(Abcss_x,2)
Sorted_abcs_unique, Occurrences = np.unique( Abcss_x_trunc, return_counts=True) 
plt.figure()
plt.plot(np.linspace(Sorted_abcs_unique[0], Sorted_abcs_unique[-1], len(Occurrences)), Occurrences/np.sum(Occurrences) )
plt.show()

#print(np.sum(Occurrences/np.sum(Occurrences)))
