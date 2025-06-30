import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv
from numba import jit
import pandas as pd
import acces_fnc


t_f = 500
t0 = 0
dt = 2e-3
N_step  = int((t_f - t0)/dt) 

u = 0.4
Fx = 1       
Fy = 2

Tau_RT_x = 3
Tau_RT_y = 5

Tx = Fx**2 * Tau_RT_x
Ty = Fy**2 * Tau_RT_y

N_traj = 1000

def init_traj_randm(u=u, Tx=Tx, Ty=Ty, N=N_step, tau_x=Tau_RT_x, tau_y = Tau_RT_y, dt = dt):
    Fx,Fy = np.sqrt(Tx/tau_x), np.sqrt(Ty/tau_y)
    X = np.zeros((2, N))
    #création de la force:
    XI = acces_fnc.create_force_2(N, dt, tau_x, tau_y, Fx, Fy)    #Le _2 indique que j'utilise la méthode de Pascal
    return X,XI



@jit(nopython=True)
def dyna(X,XI,N=N_step, dt=dt):
    #Propagation de la dynamique
    L_tot = np.zeros((N_step-1))
    for k in range(1,N):

        X[:,k] =   X[:,k-1]* (1-dt) + dt*u *X[::-1,k-1] + XI[:,k-1]*dt   #plus rapide pour numba
        
        dx, dy = X[0,k]- X[0,k-1] , X[1,k]- X[1,k-1]
        L_tot[k-1] =  -dx/dt * X[1,k] + X[0,k]* dy/dt       # x*ypoint - y xpoint  : moment cinétique calculé "exactement" à t 

    return X, L_tot
        

def Analyze(X,XI,N=N_step, Tx=Tx, Ty=Ty, Tau_RT_X = Tau_RT_x):
    #Affiche les coordonnées horaires
    #acces_fnc.coo_h(X, t0,t_f,dt)

    #Affiche la trajectoire
    #acces_fnc.traj(X, u)

    #Affiche la force
    #acces_fnc.force(XI, t0,t_f,dt, Fx, Tau_RT_x)           need tp change the forces

    #df = pd.DataFrame(X)
    #df.to_csv('your_file_name.csv', index=False)

    #Calcule la vitesse angulaire
    omega ,mean_omg, P_omega = acces_fnc.omega(X, N, dt) 
    #print('mean_omg',mean_omg, 'rad/s')

    #print(X)
    return P_omega, mean_omg, omega

def Global_analyse(OM, N_traj, N_step, omega):

    X2 = np.zeros((N_traj,N_step))
    Y2 = np.zeros((N_traj,N_step))

    L_numeric2 = np.zeros((N_traj,N_step-1))
    L_semi_analytic =  np.zeros((N_traj))

    X2[:,:] = np.power(OM[:,:,0],2)
    Y2[:,:] = np.power(OM[:,:,1],2)

    ## Probability density function
    # plt.figure()
    # plt.hist2d(np.reshape(OM[:,:,0],(-1)) , np.reshape(OM[:,:,1],(-1)) , bins=(80,80) )
    # plt.show()

    # Abcss_x = np.sort(np.reshape(OM[:,:,0], (-1)))
    # Abcss_x_trunc = np.round(Abcss_x,2)
    # Sorted_abcs_unique, Occurrences = np.unique( Abcss_x_trunc, return_counts=True) 
    # plt.figure()
    # plt.plot(np.linspace(Sorted_abcs_unique[0], Sorted_abcs_unique[-1], len(Occurrences)), Occurrences/np.sum(Occurrences) )
    # plt.show()

    L_numeric2 = np.multiply(omega,X2[:,:-1]+Y2[:,:-1] )      #L = omega * r^2
    L_semi_analytic = acces_fnc.L(X2, Y2,u)                        #calcul de L intermédiraire
     
    #print(np.sum(Occurrences/np.sum(Occurrences)))
    return X2, Y2, L_numeric2, L_semi_analytic



def simulation_launcher(N_traj, N_step, X,XI, u,Tx=Tx,Ty=Ty,tau_x=Tau_RT_x,tau_y=Tau_RT_y, dt=dt):
    timeT = time.time()

    # X2 = np.zeros((N_traj,N_step))
    # Y2 = np.zeros((N_traj,N_step))
    OM = np.zeros((N_traj,N_step,2))
    OMG = np.zeros((N_traj,N_step-1))       #regroupe les valeurs instantanée des Omegas pour chaque trajectoire
    tab_mean_omega = np.zeros((N_traj))     #regroupe le oméga moyen pour chaque trajectoire

    L_numeric = np.zeros((N_traj,N_step-1))

    for incr in range(N_traj):
        X,XI = init_traj_randm(u=u, Tx=Tx, Ty=Ty, N=N_step, tau_x=tau_x, tau_y = tau_y, dt = dt)
        X, L_numeric[incr,:] = dyna(X,XI)
        OM[incr,:,0], OM[incr,:,1] = X[0,:], X[1,:]
        P_omega, mean_omg, omega = Analyze(X,XI)
        OMG[incr,:] = omega
        tab_mean_omega[incr] = mean_omg 

    X2, Y2, L_numeric2, L_semi_analytic = Global_analyse(OM,N_traj=N_traj, N_step = N_step, omega=omega)

    r2_mean = np.mean(X2+Y2, axis=0)
    X2_analy = acces_fnc.Var_analy(u, Tau_RT_x=tau_x, Tau_RT_y=tau_y, Tx=Tx, Ty=Ty)[0]
    Y2_analy = acces_fnc.Var_analy(u, tau_x, tau_y, Tx, Ty)[1]
    R2_anal = X2_analy + Y2_analy


    L_analytic_Gleb = np.array([u*(1+1/tau_x)*((Tx/tau_x) - (Ty/tau_y))/((1 + 1/tau_x - u)*(1+ 1/tau_y + u)) ]) #only works for identical tau_y
    L_analytic_Pascal = acces_fnc.L_Pascal(u,Tx=Tx,Ty=Ty,tau_x=tau_x,tau_y=tau_y)

    #print('Took',time.time()-timeT,'s')
    return r2_mean,R2_anal,OMG, tab_mean_omega,L_numeric,L_numeric2,L_semi_analytic,L_analytic_Gleb,L_analytic_Pascal



def comparison_L(N_T,u=u,Tx=Tx,Ty=Ty,tau_x=Tau_RT_x,tau_y=Tau_RT_y,dt=dt, N_step=N_step,N_traj = N_traj  ): #changes Tx but same tau 
    dTx = Tx*5
    TempX = np.array([Tx + j*dTx for j in range(N_T)])
    Tab_comparatif = np.zeros((N_T, 4))
    list_L = ['L_numeric', 'L_semi_analytic', 'L_analytic_Gleb','L_analytic_Pascal' ]
    plt.figure()
    for j in range(N_T):
        X,XI = init_traj_randm(u=u, Tx=Tx, Ty=Ty, N=N_step, tau_x=tau_x, tau_y = tau_y, dt = dt)
        r2_mean,R2_anal,OMG, tab_mean_omega,L_numeric,L_numeric2,L_semi_analytic,L_analytic_Gleb,L_analytic_Pascal = simulation_launcher(N_traj=N_traj,N_step=N_step,X=X,XI=XI,u=u,Tx=TempX[j],Ty=Ty,tau_x=tau_x,tau_y=tau_y,dt=dt)
        Tab_comparatif[j,0] = np.mean(L_numeric)
        Tab_comparatif[j,1] = np.mean(L_semi_analytic)
        Tab_comparatif[j,2] = np.mean(L_analytic_Gleb)
        Tab_comparatif[j,3] = np.mean(L_analytic_Pascal)
    for k in range(4):
        plt.plot(TempX-Ty, Tab_comparatif[:,k], label=f'L_{list_L[k]}')
        plt.xlabel('Tx-Ty_0')
        plt.legend()
    plt.show()
    return 



def comparison_L_over_tau(N_Tau,u=u,Tx=Tx,Ty=Ty,tau_x=Tau_RT_x,tau_y=Tau_RT_y,dt=dt, N_step=N_step, N_traj = N_traj ): #changes tau (but both identical) at fixed Tx and Ty
    dtau_x = tau_x*5
    Tau_Tab = np.array([Tx + j*dtau_x for j in range(N_Tau)])
    Tab_comparatif = np.zeros((N_Tau, 4))
    list_L = ['L_numeric', 'L_semi_analytic', 'L_analytic_Gleb','L_analytic_Pascal' ]
    plt.figure()
    for j in range(N_Tau):
        tau_j = Tau_Tab[j]
        X,XI = init_traj_randm(u=u, Tx=Tx, Ty=Ty, N=N_step, tau_x=tau_j, tau_y = tau_j, dt = dt)
        r2_mean,R2_anal,OMG, tab_mean_omega,L_numeric,L_numeric2,L_semi_analytic,L_analytic_Gleb,L_analytic_Pascal = simulation_launcher(N_traj=N_traj,N_step=N_step,X=X,XI=XI,u=u,Tx=Tx,Ty=Ty,tau_x=tau_j,tau_y=tau_j,dt=dt)
        Tab_comparatif[j,0] = np.mean(L_numeric)
        Tab_comparatif[j,1] = np.mean(L_semi_analytic)
        Tab_comparatif[j,2] = L_analytic_Gleb
        Tab_comparatif[j,3] = L_analytic_Pascal
    for k in range(4):
        plt.plot(Tau_Tab, Tab_comparatif[:,k], label=f'L_{list_L[k]}')
        plt.xlabel('Tau')
        plt.legend()
    plt.show()
    return 

#comparison_L(N_T=5,u=u,Tx=Tx,Ty=Ty,tau_x=Tau_RT_x,tau_y=Tau_RT_y,dt=dt, N_step=N_step, N_traj= N_traj )
#comparison_L_over_tau(N_Tau=5,u=u,Tx=Tx,Ty=Ty,tau_x=Tau_RT_x,tau_y=Tau_RT_y,dt=dt, N_step=N_step, N_traj=N_traj )



X,XI = init_traj_randm(u=u, Tx=Tx, Ty=Ty, N=N_step, tau_x=Tau_RT_x, tau_y = Tau_RT_y, dt = dt)
r2_mean,R2_anal,OMG, tab_mean_omega,L_numeric,L_numeric2,L_semi_analytic,L_analytic_Gleb,L_analytic_Pascal = simulation_launcher(N_traj=N_traj,N_step=N_step,X=X,XI=XI,u=u,Tx=Tx,Ty=Ty,tau_x=Tau_RT_x,tau_y=Tau_RT_y)

plt.figure()
plt.plot(np.linspace(0,dt*N_step,N_step), r2_mean)
plt.plot(np.linspace(0,dt*N_step,N_step), R2_anal*np.ones_like(r2_mean))
plt.title('r2')
plt.show()

plt.figure()
plt.plot(np.linspace(0,dt*(N_step-11),N_step-11), np.mean(OMG[:,10:],axis=0))
plt.title('Omega intantané moyenné sur les traj')
plt.show()

plt.figure()
plt.hist(tab_mean_omega,density=True, bins=12)
plt.title('Occurrence des omega moyen')
plt.show()
print('Mean Omega',np.mean(tab_mean_omega))


# #variance analytic vs numérique
# plt.figure()
# plt.title('R2 distribution')
# plt.hist(np.reshape(X2[:,2000:] + Y2[:,2000:],(-1)), label='numeric', alpha=0.6, bins=60, density = 1)
# plt.hist(X2_analy + Y2_analy, label='analytic', alpha=0.6, bins=60, density=0)
# plt.hist(np.mean(np.reshape(X2[:,2000:]+ Y2[:,2000:],(-1))), label='moyenne numérique', alpha=0.6, bins=60, density=0)
# plt.legend()
# plt.show()


##L analytic vs numérique
range_center = 2
plt.figure()
plt.title('Analytic vs numeric L')
plt.hist(np.reshape(L_numeric,(-1)), label='instantaneaous L numeric', alpha=0.6, bins=60, density=1, range=(-range_center,range_center)) # L = xdot y - dotx y
plt.hist(np.reshape(L_numeric2,(-1)), label='L= omega * r^2  numeric2', alpha=0.6, bins=60, density=1,range=(-range_center,range_center)) #L = omega * r^2
#plt.hist(np.reshape(L_semi_analytic,(-1)), label='semi analytic L=u(X2-Y2) numeric', alpha=0.6, bins=60, density=1, range=(-range_center,range_center)) # 
plt.hist(L_analytic_Pascal, label='analyticP',alpha=0.7, density=0,range=(-range_center,range_center), bins=100)#) 
plt.hist(L_analytic_Gleb, label='analyticG',alpha=0.7, density=0,range=(-range_center,range_center), bins=100)#) 
plt.hist(np.mean(L_numeric),label='Mean L instantaneaous numeric', density = 0 , bins=100, range=(-range_center,range_center))
plt.hist(np.mean(L_numeric2),label='Mean L= omega * r^2 numeric', density = 0 , bins=100, range=(-range_center,range_center))
plt.hist(np.mean(L_numeric),label='Mean L= xdot y - dotx y', density = 0 , bins=100, range=(-range_center,range_center))

# plt.hist(np.reshape(L_semi_analytic,(-1)), label='interm',alpha=0.6, bins=60, density=1)#) 
plt.legend()
plt.show()

print('<L> numeric', np.mean(L_numeric), '<L> analGleb', np.mean(L_analytic_Gleb),'<L> anal_Pascal', np.mean(L_analytic_Pascal), 'semi anal L = omega * r^2', np.mean(L_semi_analytic))