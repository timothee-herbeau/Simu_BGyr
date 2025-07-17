#Solving main equation for characteristic function of each kernel. Taking FFT of product of characteristic function, we get probability density

# j'ai modifié l'eq diff 2 elle est temporairement faussée (sauf à u=0)
# les equation diff supposent un bruit dichotome symmétrique 

import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv
from numba import jit
import pandas as pd
import acces_fnc
import time
from scipy.special import gamma, jv
from scipy.integrate import solve_ivp


t0x,t0y = 0.1, 0.1
tfx, tfy = 50, 50
u=0.9 #coupling parameter
gamma_0 = 1 #transition rate
gamma_0_y = 1
omega_min,omega_max,N_om = 1, 5 ,5 #N_om must be odd so as to comply with ifft rcquirements
N_add = 15 # number of omega component we add to finish the spectrum
alpha = 1-u
vx,vy = 2,1 #intensity of the random telegraph signal

t_init = time.time()

# One whole kernel for Phi_x 
def f_mapped_om1(t,y,om, gamma, v):
    return np.array([ y[1], (1 - 2*gamma + u*np.tanh(u/t) - 2*t)*y[1]/np.power(t,2) - y[0]*np.power(v*om*np.exp(-1/(t)) * np.cosh(u/t)/np.power(t,2) ,2)   ])

def f_mapped_om2(t,y,om, gamma, v): #u/np.tanh(u*t)
    return np.array([ y[1], (1 - 2*gamma - 2*t + u/np.tanh(u/t))*y[1]/np.power(t,2) - y[0]*np.power( v*om*np.exp(-1/(t))*np.sinh(u/t) /np.power(t,2) ,2 )   ])


def f_full_P_QC(t,y,om_x,om_y, gamma, v): #convolution to xi_x
    om_c, om_s = om_x+om_y, om_x - om_y
    psi = om_c*np.cosh(u/t) + om_s*np.sinh(u/t)
    psi_prime = -u*(om_x*np.sinh(u/t)+ om_y*np.cosh(u/t))/np.power(t,2)
    return np.array([ y[1], (1 - 2*gamma - 2*t - u*psi_prime/psi)*y[1]/np.power(t,2) - y[0]*np.power( v*psi*np.exp(-1/(t))*np.sinh(u/t) /np.power(t,2) ,2 )   ])


def f_full_P_QS(t,y,om_x,om_y, gamma, v): # convolution to xi_y
    om_c, om_s = om_x+om_y, om_x - om_y
    psi = om_c*np.cosh(u/t) - om_s*np.sinh(u/t)
    psi_prime = -u*(om_x*np.sinh(u/t) -  om_y*np.cosh(u/t))/np.power(t,2)
    return np.array([ y[1], (1 - 2*gamma - 2*t - u*psi_prime/psi)*y[1]/np.power(t,2) - y[0]*np.power( v*psi*np.exp(-1/(t))*np.sinh(u/t) /np.power(t,2) ,2 )   ])


def analytical_solution(u,omega, phi,v):
    nu = gamma_0/(1-u) - u/(2*(1-u)) - 1/2
    Z = np.abs(omega)*v/(1-u) * np.exp(-(1-u)/phi) 
    return    gamma(1+nu)*jv(nu,Z)/np.power(Z/2,nu)

def analytical_solution_tim(u,omega, phi,v):
    nu = ((2*gamma_0-1)/(1-u) -u )/2
    #nu = gamma_0/(1-u) - u/(2*(1-u)) - 1/2
    Z = np.abs(omega)*v/(1-u) * np.exp(-(1-u)/phi) 
    return    gamma(1+nu)*jv(nu,Z)/np.power(Z/2,nu)

om_X = np.linspace(-omega_max,omega_max,N_om)  
om_Y = np.linspace(-omega_max,omega_max,N_om)  
print('om is of shape',len(om_X))
final_Phi_1 = np.zeros((N_om))
final_Phi_2 = np.zeros((N_om))

final_Phi_1_S = np.zeros((N_om))
final_Phi_2_S = np.zeros((N_om))

final_2D_Phi = np.zeros((N_om,N_om))

#plt.figure()
for k in range(len(om_X)):
    for j in range(len(om_Y)):
        omg_x, omg_y = om_X[k], om_Y[k]
        sol1 = solve_ivp( f_full_P_QC ,(t0x,tfx),args=(omg_x, omg_y,gamma_0, vx),y0=np.array([1,0]))
    #print(final_Phi_1[k])
        sol2 = solve_ivp( f_full_P_QS ,(t0y,tfy),args=(omg_x,omg_y,gamma_0, vx),y0=np.array([1,0]))
        final_2D_Phi =sol2.y[0,-1] *sol1.y[0,-1] 

    # sol1Y = solve_ivp( f_mapped_om2 ,(t0x,tfx),args=(omg,gamma_0_y, vy),y0=np.array([1,0]))
    # final_Phi_1Y[k] =sol1Y.y[0,-1] 
    # #print(final_Phi_1[k])
    # sol2Y = solve_ivp( f_mapped_om1 ,(t0y,tfy),args=(omg,gamma_0_y, vy),y0=np.array([1,0]))
    # final_Phi_2Y[k] =sol2Y.y[0,-1] 
    #print(final_Phi_2[k]) 
    #plt.plot(sol1.t,sol1.y[0,:], label=f'num {omg}') #we have now the value of the caractéristic function at phi infty as a function of omega 
   #plt.scatter(sol1.t,analytical_solution_tim(u, omega=omg,phi=sol1.t, v=vx), label=f'anal {omg}')
# plt.title('Phi_omega(phi))')
# plt.xscale('log')
# plt.legend()
# plt.show()
    

print('time',time.time()-t_init)




plt.show()   
plt.figure()
plt.title('Phi_infty(omega))')
plt.plot(om_X,final_2D_Phi[0,:],label='numerical') #we have now the value of the caractéristic function at phi infty as a function of omega 
plt.plot(om_X,analytical_solution_tim(u, om_X,phi=1e8, v=vx),label='analytical')
plt.plot(om_X,analytical_solution(u, om_X,phi=1e8, v=vx),label='analytical Gleb')
plt.legend()
plt.show()

dw = (om_X[1]-om_X[0])

extended_phi_1 = np.concatenate((np.zeros((N_add)), final_Phi_1, np.zeros((N_add))))
final_Phi_reordered_1 = np.zeros((len(extended_phi_1)))      
final_Phi_reordered_1[len(extended_phi_1)//2 +1:] = extended_phi_1[: len(extended_phi_1)//2]
final_Phi_reordered_1[:len(extended_phi_1)//2 +1 ] = extended_phi_1[ len(extended_phi_1)//2 : ]

extended_phi_2 = np.concatenate((np.zeros((N_add)), final_Phi_2, np.zeros((N_add))))
final_Phi_reordered_2 = np.zeros((len(extended_phi_2)))      
final_Phi_reordered_2[len(extended_phi_2)//2 +1:] = extended_phi_2[: len(extended_phi_2)//2]
final_Phi_reordered_2[:len(extended_phi_2)//2 +1 ] = extended_phi_2[len(extended_phi_2)//2 : ]


# extended_phi_1Y = np.concatenate((np.zeros((N_add)), final_Phi_1Y, np.zeros((N_add))))
# final_Phi_reordered_1Y = np.zeros((len(extended_phi_1)))      
# final_Phi_reordered_1Y[len(extended_phi_1Y)//2 +1:] = extended_phi_1Y[: len(extended_phi_1Y)//2]
# final_Phi_reordered_1Y[:len(extended_phi_1Y)//2 +1 ] = extended_phi_1Y[ len(extended_phi_1Y)//2 : ]

# extended_phi_2Y = np.concatenate((np.zeros((N_add)), final_Phi_2Y, np.zeros((N_add))))
# final_Phi_reordered_2Y = np.zeros((len(extended_phi_2Y)))      
# final_Phi_reordered_2Y[len(extended_phi_2Y)//2 +1:] = extended_phi_2Y[: len(extended_phi_2Y)//2]
# final_Phi_reordered_2Y[:len(extended_phi_2Y)//2 +1 ] = extended_phi_2Y[len(extended_phi_2Y)//2 : ]

# PHI_X = np.multiply(final_Phi_reordered_1,final_Phi_reordered_2)
# PHI_Y = np.multiply(final_Phi_reordered_1Y,final_Phi_reordered_2Y)


# final_Phi_multiplied = np.matmul( np.reshape(PHI_X,(len(PHI_X),-1)) , np.reshape(PHI_Y,(-1,len(PHI_Y))) )

P = np.fft.fft2(final_2D_Phi, norm='forward')
#Marg_P_X = np.fft.fftshift(np.fft.fft(PHI_X, norm='forward'))
#print(np.real(P))
X = np.linspace(-len(P[0,:]) ,len(P[0,:]) , 2*len(P[0,:]))
x = np.linspace(-len(P[0,:])//2 ,len(P[0,:])//2, len(P[0,:]))
print(np.shape(P), np.shape(np.array(x)))
# # def P_analy(X):
# #     return np.where((np.abs(X)<v)&(np.power(v,2) -  np.power(X,2)!=0),        np.power(np.power(v,2) -  np.power(X,2), gamma_0 - 1 )* gamma(0.5 + gamma_0)/((np.power(v,-1+ 2*gamma_0))* gamma(0.5)*gamma(gamma_0))         , 0)  

# #print('sumP_analytc', 1/(len(final_Phi_reordered_1)*dw),np.sum(P_analy(X*2*np.pi/(len(final_Phi_reordered_1)*dw)))*2*np.pi/(len(final_Phi_reordered_1)*dw), 
print('computed P norm',np.sum(P[:,:]), 'dw', dw) 
    
# plt.figure()
# plt.title('Phi_infty(omega))')
# plt.scatter(np.concatenate((np.array([-omega_max - dw*(N_add-j) for j in range(N_add)]),om,np.array([omega_max + dw*j for j in range(N_add)]))),extended_phi_1) #we have now the value of the caractéristic function at phi infty as a function of omega 
# plt.show()

plt.figure()
plt.plot(X[::1] , np.concatenate((P[:,:],P[:,:])) ) #10+ len(P)//2
#plt.plot(X ,P_analy(X*2*np.pi/(len(final_Phi_reordered_1)*dw))*2*np.pi/(len(final_Phi_reordered_1)*dw) )
plt.title('P(x)')
plt.show()

# plt.figure()
# plt.plot(np.linspace(-len(Marg_P_X)//2, len(Marg_P_X)//2, len(Marg_P_X)), Marg_P_X)
# plt.title('Marginal P(x)')
# plt.show()

# plt.figure()
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(x, x, P)#, vmin=Z.min() * 2, cmap=cm.Blues)
# plt.show()

# Probability density function
nx,ny = np.shape(P)[0], np.shape(P)[1]
M = np.zeros((2*nx,2*ny))
M[:nx,:ny] = P
M[nx:,ny:] = P
M[:nx,ny:] = P
M[nx:,:ny] = P 
#np.array[(np.reshape(np.real( np.concatenate((P,P,P,P))) , (2*len(P[0,:]),2*len(P[0,:]))  ))] 
plt.figure()
plt.imshow(np.real(M[nx//2:nx + nx//2,ny//2:ny+ny//2]))
#plt.pcolormesh(X,X,np.reshape(np.real( np.concatenate((P,P,P,P))) , (2*len(P[0,:]),2*len(P[0,:]))  ))
#plt.hist2d( np.array(x), np.array(x) ,weights = P)
plt.show()


fig = plt.figure(figsize=(14,6))

# `ax` is a 3D-aware axis instance because of the projection='3d' keyword argument to add_subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')
#print( np.log(np.where(np.real(P)>0,np.real(P),1))) np.log(np.where(np.real(P)>0,np.real(P),1))
XX, YY = np.meshgrid(x, x)
p = ax.plot_wireframe(XX, YY, np.where(np.real(M[nx//2:nx + nx//2,ny//2:ny+ny//2])>0,np.real(M[nx//2:nx + nx//2,ny//2:ny+ny//2]),0))#, rstride=4, cstride=4, linewidth=0)
plt.show()