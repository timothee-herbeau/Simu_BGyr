import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv, jv, gamma, hyp2f1
from numba import jit
import pandas as pd
import acces_fnc


t_f = 150
t0 = 0
dt = 5e-3
N_step  = int((t_f - t0)/dt) 


u = 0.8
Fx = 1
Fy = 1

Tau_RT_x = 1e-2 #*1.2
Tau_RT_y = 1

Tx = Fx**2 * Tau_RT_x
Ty = Fy**2 * Tau_RT_y

N_traj = 1000

def init_traj_randm(u, Tx=Tx, Ty=Ty, N=N_step, tau_x=Tau_RT_x, tau_y = Tau_RT_y, dt = dt):
    Fx,Fy = np.sqrt(Tx/tau_x), np.sqrt(Ty/tau_y)
    #print(Fx)
    X = np.zeros((2, N))
    #X[:,0] = np.random.normal(0,1,2)
    #création de la force:
    XI = acces_fnc.create_force(N, dt, tau_x, tau_y, Fx, Fy)    #Le _2 indique que j'utilise la méthode de Pascal
    return X,XI



@jit(nopython=True)
def dyna(X,XI,u,N=N_step, dt=dt):
    #Propagation de la dynamique
    L_tot = np.zeros((N_step-1))
    for k in range(1,N):

        X[:,k] =   X[:,k-1]* (1-dt) + dt*u *X[::-1,k-1] + XI[:,k-1]*dt   #plus rapide pour numba
        
        dx, dy = X[0,k]- X[0,k-1] , X[1,k]- X[1,k-1]
        L_tot[k-1] = X[0,k]* dy/dt -dx/dt * X[1,k]        # x*ypoint - y xpoint  : moment cinétique calculé "exactement" à t 

    return X, L_tot
        

def Analyze(X,XI,N=N_step, Tx=Tx, Ty=Ty, Tau_RT_X = Tau_RT_x):
    #Affiche les coordonnées horaires
    #acces_fnc.coo_h(X, t0,t_f,dt)

    # #Affiche la trajectoire
    #acces_fnc.traj(X, u)

    # #Affiche la force
    # acces_fnc.force(XI, t0,t_f,dt, Fx, Tau_RT_x)       #    need tp change the forces

    #df = pd.DataFrame(X)
    #df.to_csv('your_file_name.csv', index=False)

    #Calcule la vitesse angulaire
    omega ,mean_omg, P_omega = acces_fnc.omega(X, N, dt) 
    #print('mean_omg',mean_omg, 'rad/s')

    #print(X)
    return P_omega, mean_omg, omega

def Global_analyse(OM, N_traj, N_step, omega,u):

    X2 = np.zeros((N_traj,N_step))
    Y2 = np.zeros((N_traj,N_step))

    L_numeric2 = np.zeros((N_traj,N_step-1))
    L_semi_analytic =  np.zeros((N_traj))

    X2[:,:] = np.power(OM[:,:,0],2)
    Y2[:,:] = np.power(OM[:,:,1],2)

    # # Probability density function
    # plt.figure()
    # plt.hist2d(np.reshape(OM[:,:,0],(-1)) , np.reshape(OM[:,:,1],(-1)) , bins=(80,80) )
    # plt.show()

    # Abcss_x = np.sort(np.reshape(OM[:,:,0], (-1)))
    # Abcss_x_trunc = np.round(Abcss_x,2)
    # Sorted_abcs_unique, Occurrences = np.unique( Abcss_x_trunc, return_counts=True) 
    # plt.figure()
    # print(OM[:,:,0])#,Occurrences/np.sum(Occurrences), Sorted_abcs_unique )
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
        X, L_numeric[incr,:] = dyna(X,XI,u)
        OM[incr,:,0], OM[incr,:,1] = X[0,:], X[1,:]
        P_omega, mean_omg, omega = Analyze(X,XI)
        OMG[incr,:] = omega
        tab_mean_omega[incr] = mean_omg 

    X2, Y2, L_numeric2, L_semi_analytic = Global_analyse(OM,N_traj=N_traj, N_step = N_step, omega=omega,u=u)

    r2_mean = np.mean(X2+Y2, axis=0)
    X2_analy = acces_fnc.Var_analy(u, Tau_RT_x=tau_x, Tau_RT_y=tau_y, Tx=Tx, Ty=Ty)[0]
    Y2_analy = acces_fnc.Var_analy(u, tau_x, tau_y, Tx, Ty)[1]
    R2_anal = X2_analy + Y2_analy


    L_analytic_Gleb = np.array([u*(1+1/tau_x)*((Tx/tau_x) - (Ty/tau_y))/((1 + 1/tau_x - u)*(1+ 1/tau_y + u)) ]) #only works for identical tau_y
    L_analytic_Pascal = acces_fnc.L_Pascal(u,Tx=Tx,Ty=Ty,tau_x=tau_x,tau_y=tau_y)

    #print('Took',time.time()-timeT,'s')
    return r2_mean,R2_anal,OMG, tab_mean_omega,L_numeric,L_numeric2,L_semi_analytic,L_analytic_Gleb,L_analytic_Pascal, X2_analy,X2, OM


########
#Comparaison des moments cinétiques
#######
def comparison_L(N_T,u=u,Tx=Tx,Ty=Ty,tau_x=Tau_RT_x,tau_y=Tau_RT_y,dt=dt, N_step=N_step,N_traj = N_traj  ): #changes Tx but same tau 
    dTx = Tx*5
    TempX = np.array([Tx + j*dTx for j in range(N_T)])
    Tab_comparatif = np.zeros((N_T, 4))
    list_L = ['L_numeric', 'L_semi_analytic', 'L_analytic_Gleb','L_analytic_Pascal' ]
    plt.figure()
    for j in range(N_T):
        X,XI = init_traj_randm(u=u, Tx=Tx, Ty=Ty, N=N_step, tau_x=tau_x, tau_y = tau_y, dt = dt)
        r2_mean,R2_anal,OMG, tab_mean_omega,L_numeric,L_numeric2,L_semi_analytic,L_analytic_Gleb,L_analytic_Pascal,X2_ANAL,X2,OM = simulation_launcher(N_traj=N_traj,N_step=N_step,X=X,XI=XI,u=u,Tx=TempX[j],Ty=Ty,tau_x=tau_x,tau_y=tau_y,dt=dt)
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
        r2_mean,R2_anal,OMG, tab_mean_omega,L_numeric,L_numeric2,L_semi_analytic,L_analytic_Gleb,L_analytic_Pascal,X2_ANAL,X2, OM = simulation_launcher(N_traj=N_traj,N_step=N_step,X=X,XI=XI,u=u,Tx=Tx,Ty=Ty,tau_x=tau_j,tau_y=tau_j,dt=dt)
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

def comparison_L_over_u(N_u,u=u,Tx=Tx,Ty=Ty,tau_x=Tau_RT_x,tau_y=Tau_RT_y,dt=dt, N_step=N_step, N_traj = N_traj ): #changes tau (but both identical) at fixed Tx and Ty
    du = 0.3
    u_Tab = np.array([-0.9 + j*du for j in range(N_u)])
    Tab_comparatif = np.zeros((N_u, 4))
    list_L = ['L_numeric', 'L_semi_analytic', 'L_analytic_Gleb','L_analytic_Pascal' ]
    plt.figure()
    for j in range(N_u):
        u_j = u_Tab[j]
        X,XI = init_traj_randm(u=u_j, Tx=Tx, Ty=Ty, N=N_step, tau_x=tau_x, tau_y = tau_y, dt = dt)
        r2_mean,R2_anal,OMG, tab_mean_omega,L_numeric,L_numeric2,L_semi_analytic,L_analytic_Gleb,L_analytic_Pascal,X2_ANAL,X2,OM = simulation_launcher(N_traj=N_traj,N_step=N_step,X=X,XI=XI,u=u_j,Tx=Tx,Ty=Ty,tau_x=tau_x,tau_y=tau_y,dt=dt)
        Tab_comparatif[j,0] = np.mean(L_numeric)
        Tab_comparatif[j,1] = np.mean(L_semi_analytic)
        Tab_comparatif[j,2] = L_analytic_Gleb
        Tab_comparatif[j,3] = L_analytic_Pascal
    for k in range(4):
        plt.plot(u_Tab, Tab_comparatif[:,k], label=f'L_{list_L[k]}')
        plt.xlabel('u')
        plt.legend()
    plt.show()
    return 

#comparison_L(N_T=5,u=u,Tx=Tx,Ty=Ty,tau_x=Tau_RT_x,tau_y=Tau_RT_y,dt=dt, N_step=N_step, N_traj= N_traj )
#comparison_L_over_tau(N_Tau=5,u=u,Tx=Tx,Ty=Ty,tau_x=Tau_RT_x,tau_y=Tau_RT_y,dt=dt, N_step=N_step, N_traj=N_traj )
#comparison_L_over_u(N_u=7,u=u,Tx=Tx,Ty=Ty,tau_x=Tau_RT_x,tau_y=Tau_RT_y,dt=dt, N_step=N_step, N_traj= N_traj )

def one_try(u=u, Tx=Tx, Ty=Ty, N=N_step, tau_x=Tau_RT_x, tau_y = Tau_RT_y, dt = dt):
    X,XI = init_traj_randm(u=u, Tx=Tx, Ty=Ty, N=N_step, tau_x=Tau_RT_x, tau_y = Tau_RT_y, dt = dt)
    r2_mean,R2_anal,OMG, tab_mean_omega,L_numeric,L_numeric2,L_semi_analytic,L_analytic_Gleb,L_analytic_Pascal,X2_analytic,X2,OM = simulation_launcher(N_traj=N_traj,N_step=N_step,X=X,XI=XI,u=u,Tx=Tx,Ty=Ty,tau_x=Tau_RT_x,tau_y=Tau_RT_y)

    # plt.figure()
    # plt.plot(np.linspace(0,dt*N_step,N_step), r2_mean)
    # plt.plot(np.linspace(0,dt*N_step,N_step), R2_anal*np.ones_like(r2_mean))
    # plt.title('r2')
    # plt.show()

    # plt.figure()
    # plt.plot(np.linspace(0,dt*(N_step-11),N_step-11), np.mean(OMG[:,10:],axis=0))
    # plt.title('Omega intantané moyenné sur les traj')
    # plt.show()

    plt.figure()
    plt.hist(tab_mean_omega,density=True, bins=12)
    plt.title('Occurrence des omega moyen')
    plt.show()
    print('Mean Omega',np.mean(tab_mean_omega))


    #variance analytic vs numérique
    plt.figure()
    plt.title('X2 distribution')
    #plt.hist(np.array([X2_analytic]), label='analytic', alpha=0.6, bins=60, density = 0)
    plt.hist(np.reshape(X2[:,2000:]  ,(-1)), label='Experimental distribution of X2', alpha=0.6, bins=60, density = 1, range=(0,3))
    plt.hist(X2_analytic, label='Analytic expactation of X2', alpha=0.6, bins=40, density=0)
    plt.hist(np.mean(np.reshape(X2[:,2000:]  ,(-1))), alpha=0.6, bins=40, density=0, label='Experimental mean X2' )
    #plt.hist(np.mean(np.reshape(X2_analytic[:,2000:] ,(-1))), label='moyenne numérique', alpha=0.6, bins=60, density=0)
    plt.legend()
    plt.show()


    ##L analytic vs numérique
    range_center = 10
    plt.figure()
    plt.title('Analytic vs numeric L')
    plt.hist(np.reshape(L_numeric,(-1)), label='L= xdot y - dotx y', alpha=0.6, bins=60, density=1, range=(-range_center,range_center)) # L = xdot y - dotx y
    #plt.hist(np.reshape(L_numeric2,(-1)), label='L= omega * r^2  numeric2', alpha=0.6, bins=60, density=1,range=(-range_center,range_center)) #L = omega * r^2
    #plt.hist(np.reshape(L_semi_analytic,(-1)), label='semi analytic L=u(X2-Y2) numeric', alpha=0.6, bins=60, density=1, range=(-range_center,range_center)) # 

    #plt.hist(np.mean(L_numeric),label='Mean L instantaneaous numeric',alpha=1, density = 0 , bins=400, range=(-range_center,range_center))
    #plt.hist(np.mean(L_numeric2),label='Mean L= omega * r^2 numeric',alpha=0.8, density = 0 , bins=400, range=(-range_center,range_center))
    plt.hist(np.mean(L_numeric),label='Mean L= xdot y - dotx y',alpha=0.8, density = 0 , bins=400, range=(-range_center,range_center))

    plt.hist(L_analytic_Pascal, label='analyticP',alpha=0.8, density=0,range=(-range_center,range_center), bins=400)#) 
    plt.hist(L_analytic_Gleb, label='analyticG',alpha=0.8, density=0,range=(-range_center,range_center), bins=400)
    # plt.hist(np.reshape(L_semi_analytic,(-1)), label='interm',alpha=0.6, bins=60, density=1)#) 
    plt.legend()
    plt.show()

    print('<L> numeric', np.mean(L_numeric), '<L> analGleb', np.mean(L_analytic_Gleb),'<L> anal_Pascal', np.mean(L_analytic_Pascal), 'semi anal L = omega * r^2', np.mean(L_semi_analytic))
    return



def P_u(x,u,P0, a):
    return P0+ np.power(x,2)*a

def phase_transition(N_u,Tx=Tx,Ty=Ty,tau_x=Tau_RT_x,tau_y=Tau_RT_y,dt=dt, N_step=N_step, N_traj = N_traj ):
    t0 = time.time()
    Fx = np.sqrt(Tx/tau_x)
    #goal is to have d2P for marginal distribution in x over a variety of u, show a behaviour in (1-u)^3/2
    du = 0.9/N_u
    u_Tab = np.array([ j*du for j in range(N_u)])
    plt.figure()
    for u in u_Tab:
        X,XI = init_traj_randm(u=u, Tx=Tx, Ty=Ty, N=N_step, tau_x=tau_x, tau_y = tau_y, dt = dt)
        r2_mean,R2_anal,OMG, tab_mean_omega,L_numeric,L_numeric2,L_semi_analytic,L_analytic_Gleb,L_analytic_Pascal,X2_analytic,X2,OM = simulation_launcher(N_traj=N_traj,N_step=N_step,X=X,XI=XI,u=u,Tx=Tx,Ty=Ty,tau_x=tau_x,tau_y=tau_y)
        Abcss_x = np.sort(np.reshape(OM[:,:,0], (-1)))
        Abcss_x_trunc = np.round(Abcss_x,5)
        Sorted_abcs_unique, Occurrences = np.unique( Abcss_x_trunc, return_counts=True) 
        P = Occurrences/np.sum(Occurrences)
        second_der = np.array([(-2*P[k]+P[k-1]+P[k+1])/(Sorted_abcs_unique[k+1]-Sorted_abcs_unique[k])**2 for k in range(1,len(P)-1)])
        a = 4*gamma(3/2)/Fx**3 * np.power( (1/tau_x - 1/2)*(1-u),3/2) 
        plt.plot(Sorted_abcs_unique,P,label=f'P for u={u}')
        plt.plot(Sorted_abcs_unique[-100+len(Sorted_abcs_unique)//2 : 100+len(Sorted_abcs_unique)//2], P_u(Sorted_abcs_unique[-100+len(Sorted_abcs_unique)//2 : 100+len(Sorted_abcs_unique)//2],u, P0=P[len(P)//2], a =a), label=f'u = {u}' ) 
        #plt.plot(Sorted_abcs_unique[1:-1],second_der, label=f'Second order u={u}')
        #plt.scatter(np.array([0]), np.array([4*gamma(3/2)/Fx**3 * np.power( (1/tau_x - 1/2)*(1-u),3/2) ]), label=f'Second order deriv u={u}' )
        # plt.figure()
        # plt.plot(np.linspace(Sorted_abcs_unique[0], Sorted_abcs_unique[-1], len(Occurrences)), Occurrences/np.sum(Occurrences) )
    print(np.round(time.time()-t0,1),'seconds',)
    plt.legend()
    plt.show()
    return


def phase_transition_gamma(N_gamma,u=u,Fx=Fx,Fy=Fy,dt=dt, N_step=N_step, N_traj = N_traj ):
    t0 = time.time()
    
    #goal is to have d2P for marginal distribution in x over a variety of u, show a behaviour in (1-u)^3/2
    dgm = 1.5/N_gamma
    gm_Tab = np.array([ 0.05 + j*dgm for j in range(N_gamma)])
    #gm_Tab = np.array([3e-2, 8e-2, 0.15, 0.4, 0.8 , 1 ])
    print(gm_Tab)
    plt.figure()
    for gm in gm_Tab:
        tau_x, tau_y = 1/gm, 1/gm
        Tx,Ty = tau_x*Fx**2, tau_y*Fy**2
        #print('Fxy',Fx,Fy, 'Tx',Tx)
        X,XI = init_traj_randm(u=u, Tx=Tx, Ty=Ty, N=N_step, tau_x=tau_x, tau_y = tau_y, dt = dt)
        r2_mean,R2_anal,OMG, tab_mean_omega,L_numeric,L_numeric2,L_semi_analytic,L_analytic_Gleb,L_analytic_Pascal,X2_analytic,X2,OM = simulation_launcher(N_traj=N_traj,N_step=N_step,X=X,XI=XI,u=u,Tx=Tx,Ty=Ty,tau_x=tau_x,tau_y=tau_y)
        Abcss_x = np.sort(np.reshape(OM[:,50:,0]+OM[:,50:,1] , (-1))) #P(U)
        Abcss_x_trunc = np.round(Abcss_x,2)
        Sorted_abcs_unique, Occurrences = np.unique( Abcss_x_trunc, return_counts=True) 
        P = Occurrences/np.sum(Occurrences)
        print('sum', np.sum(P))
        #print('eq points', (Fx*u**2 -u*Fx+Fx + u*Fy)/(1-u), (Fx*u**2-u*Fx+Fx -u*Fy)/(1-u))
        #second_der = np.array([(-2*P[k]+P[k-1]+P[k+1])/(Sorted_abcs_unique[k+1]-Sorted_abcs_unique[k])**2 for k in range(1,len(P)-1)])
        #a = 4*gamma(3/2)/Fx**3 * np.power( (1/tau_x - 1/2)*(1-u),3/2) 
        plt.plot(Sorted_abcs_unique,P,label=f'P for gm={gm}, alpha_x={np.round(1/2 *gm/(1-u),2)},alpha_y={np.round(1/4 *gm/(1-u),2)}, vx,vy = {(Fx,Fy)}')
        plt.yscale('log')
        #plt.plot(Sorted_abcs_unique[-100+len(Sorted_abcs_unique)//2 : 100+len(Sorted_abcs_unique)//2], P_u(Sorted_abcs_unique[-100+len(Sorted_abcs_unique)//2 : 100+len(Sorted_abcs_unique)//2],u, P0=P[len(P)//2], a =a), label=f'gm= {gm}' ) 
        #plt.plot(Sorted_abcs_unique[1:-1],second_der, label=f'Second order u={u}')
        #plt.scatter(np.array([0]), np.array([4*gamma(3/2)/Fx**3 * np.power( (1/tau_x - 1/2)*(1-u),3/2) ]), label=f'Second order deriv u={u}' )
        # plt.figure()
        # plt.plot(np.linspace(Sorted_abcs_unique[0], Sorted_abcs_unique[-1], len(Occurrences)), Occurrences/np.sum(Occurrences) )
    print(np.round(time.time()-t0,1),'seconds',)
    plt.legend()
    plt.show()

    return

#phase_transition_gamma(6,u=0.1,Fx=1,Fy=2,dt=dt, N_step=N_step, N_traj = 300 )


#compute equilibrium or favorable points 
def chaos_transition(N_gamma,u=u,Fx=Fx,Fy=Fy,dt=dt, N_step=N_step, N_traj = N_traj ):
    N_pts = 60
    t0 = time.time()
    #goal is to have d2P for marginal distribution in x over a variety of u, show a behaviour in (1-u)^3/2
    x_eq = np.zeros((N_gamma, N_pts))
    du = 0.99/N_gamma
    u_Tab = np.array([ 0.0 + j*du for j in range(N_gamma)])
     
    plt.figure()

    for k in range(len(u_Tab)):
        gm = 0.2
        u_j = u_Tab[k]
        tau_x, tau_y = 1/gm, 1/gm
        Tx,Ty = tau_x*Fx**2, tau_y*Fy**2
        #print('Fxy',Fx,Fy, 'Tx',Tx)
        X,XI = init_traj_randm(u=u, Tx=Tx, Ty=Ty, N=N_step, tau_x=tau_x, tau_y = tau_y, dt = dt)
        r2_mean,R2_anal,OMG, tab_mean_omega,L_numeric,L_numeric2,L_semi_analytic,L_analytic_Gleb,L_analytic_Pascal,X2_analytic,X2,OM = simulation_launcher(N_traj=N_traj,N_step=N_step,X=X,XI=XI,u=u_j,Tx=Tx,Ty=Ty,tau_x=tau_x,tau_y=tau_y)
        Abcss_x = np.sort(np.reshape(OM[:,50:,0]+ OM[:,50:,1], (-1))) #P(x)
        Abcss_x_trunc = np.round(Abcss_x,2)
        Sorted_abcs_unique, Occurrences = np.unique( Abcss_x_trunc, return_counts=True)
        P = Occurrences/np.sum(Occurrences)
        x_eq[k,:] = Sorted_abcs_unique[ np.argsort(P)[-N_pts:] ]
        plt.scatter(np.ones((N_pts))*u_j,x_eq[k,:], label=f'Experimental favorable points', color='red')
        #X_EQ = np.array([(Fx+ u_j*Fy)/(1-u_j**2), (Fx-u_j*Fy)/(1-u_j**2)  ])
        U_EQ = np.array([(Fy+Fx)/(1-u_j),(-Fx+Fy)/(1-u_j), -(Fy+Fx)/(1-u_j),-(-Fx+Fy)/(1-u_j) ])
        plt.scatter(np.ones((4))*u_j,U_EQ, marker='x', color='black' )#, label='Predicted')
        #print(  'eq points',Fx,Fy,u, (Fx+ u_j*Fy)/(1-u_j**2), (Fx-u_j*Fy)/(1-u_j**2))
    print('took',time.time()-t0)
    plt.xlabel('u')
    plt.yscale('symlog')
    plt.ylabel('Favorised U')
     
    plt.title(f'Evolution of maxima of center of mass probability depending on u at gamma = {gm}' )
    plt.show()

    return


chaos_transition(20,u=0.2,Fx=1,Fy=3,dt=dt, N_step=N_step, N_traj = 300 )