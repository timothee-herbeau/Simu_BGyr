import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv
from numba import jit
import pandas as pd
import acces_fnc
#from sklearn.metrics import mean_squared_error

t_f = 50
t0 = 0
dt = 2e-3
N_step  = int((t_f - t0)/dt) 

u = 0.4
Fx = 1       
Fy = 1

Tau_RT_x = 5e-1
Tau_RT_y = 1e-1  

Tx = Fx**2 * Tau_RT_x
Ty = Fy**2 * Tau_RT_y

N_traj = 1000

def init_traj_randm(u=u, Fx=Fx, Fy=Fy, N=N_step):
    X = np.zeros((2, N))
    #création de la force:
    XI = acces_fnc.create_force_2(N, dt, Tau_RT_x, Tau_RT_y, Fx, Fy)
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

def A(u, tau):
    return (2 - u**2 + tau)/(2*(1-u**2)*(1+2*tau + (1-u**2)*tau**2))
def B(u,tau):
    return u**2 * (1 + tau)/(2*(1-u**2)*(1+2*tau + (1-u**2)*tau**2))

def Var_analy(u, Tau_RT_x, Tau_RT_y, Tx, Ty):
    return np.array([ A(u,Tau_RT_x)*Tx + B(u,Tau_RT_y)*Ty,  B(u,Tau_RT_x)*Tx + A(u,Tau_RT_y)*Ty ])


timeT = time.time()

X2 = np.zeros((N_traj,N_step))
Y2 = np.zeros((N_traj,N_step))


OMG = np.zeros((N_traj,N_step-1))
tab_mean_omega = np.zeros((N_traj))


L_analytic =  np.zeros((N_traj,N_step))
L_numeric = np.zeros((N_traj,N_step-1))
L_numeric2 = np.zeros((N_traj,N_step-1))

for incr in range(N_traj):
    X,XI = init_traj_randm()
    X, L_numeric[incr,:] = dyna(X,XI)

    P_omega, mean_omg, omega = Analyze(X,XI)
    X2[incr,:] = np.power(X[0,:],2)
    Y2[incr,:] = np.power(X[1,:],2)
    L_numeric2[incr,:] = np.multiply(omega,X2[incr,:-1]+Y2[incr,:-1] )
    L_analytic[incr,:] = L(X2[incr,:], Y2[incr,:],u)
    #OMG[incr,:] = omega[ int(N_step - 2/dt) : ]
    OMG[incr,:] = omega
    tab_mean_omega[incr] = mean_omg 
    

r2_mean = np.mean(X2+Y2, axis=0)
X2_analy = Var_analy(u, Tau_RT_x, Tau_RT_y, Tx, Ty)[0]
Y2_analy = Var_analy(u, Tau_RT_x, Tau_RT_y, Tx, Ty)[1]
R2_anal = X2_analy + Y2_analy


print('Took',time.time()-timeT,'s')


plt.figure()
plt.plot(np.linspace(0,dt*N_step,N_step), r2_mean)
plt.plot(np.linspace(0,dt*N_step,N_step), R2_anal*np.ones_like(r2_mean))
plt.title('r2')
plt.show()

# plt.figure()
# plt.plot(np.linspace(0,dt*(N_step-11),N_step-11), np.mean(OMG[:,10:],axis=0))
# plt.title('Omega intantané moyenné sur les traj')
# plt.show()

# plt.figure()
# plt.hist(tab_mean_omega,density=True, bins=12)
# plt.title('Occurrence des omega moyen')
# plt.show()
# print('Mean Omega',np.mean(tab_mean_omega))


#variance analytic vs numérique

plt.figure()
plt.title('X2 distribution')
plt.hist(np.reshape(X2[:,2000:],(-1)), label='numeric', alpha=0.6, bins=60, density = True)
plt.hist(X2_analy, label='numeric', alpha=0.6, bins=60, density=True)
plt.hist(np.mean(np.reshape(X2[:,2000:],(-1))), label='numeric', alpha=0.6, bins=60, density=True)
plt.show()


##L analytic vs numérique

plt.figure()
plt.title('Analytic vs numeric L')
plt.hist(np.reshape(L_numeric,(-1)), label='numeric', alpha=0.6, bins=60)
plt.hist(np.reshape(L_analytic,(-1)), label='analytic',alpha=0.7, bins=60) 
plt.hist(np.reshape(L_numeric2,(-1)), label='numeric2',alpha=0.6, bins=60) 
plt.legend()
plt.show()



# plt.figure()
# plt.title('Analytic L')
# plt.hist(np.mean(L_numeric,axis=0), label='numeric', alpha=0.6, bins=2, density=True)
# plt.hist(np.mean(L_analytic,axis=0), label='analytic',alpha=0.6, bins=20, density=True)
# plt.hist(np.mean(L_numeric2,axis=0), label='num2',alpha=0.6, bins=20,density=True)
# plt.legend()
# plt.show()

