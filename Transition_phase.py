import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, gamma,hyp2f1

u = 0.15
gamma_0 = 0.9
v = 1

def secon_der(x,u,gamma_0):
    nu = (2*gamma_0-1)/(2-2*u)
    C = nu/(1-u)
    return 2*x/C**3 * jv(nu-1,x) * np.power(2/x , nu-1) * gamma(1+nu)

def D2P(x,u,gamma_0):
    nu = (2*gamma_0-1)/(2-2*u)
    C = nu/(1-u)
    
    arr = np.array([ (-1)**k * np.power(x/2, 2*k+1) /(gamma(nu+k) *gamma(k+1)) for k in range(0,1000,2)       ]) * 4/ C**3
    return np.sum(arr)


def simplf(u,gamma_0):
    nu = (2*gamma_0-1)/(2-2*u)
    C = nu/(1-u)
    return 4*gamma(3/2)/C**3 * gamma(1+nu)/gamma(nu-1/2)

def trace():
    tab_u = np.linspace(-0.8, 0.8,60)
    X_arr = np.linspace(1,20,5)
    gm_arr = np.linspace(0.4,2,20)
    plt.figure()
    # for x in X_arr:
    #     print(x)
    #     y = np.array( [D2P(x,u_j,gamma_0) for u_j in tab_u])    #- secon_der(-X,tab_u,gamma_0)
    #     plt.plot(tab_u, y, label=f'x={x}')
    #plt.plot(tab_u,simplf(tab_u,gamma_0))
    for gm in gm_arr:
        
        y = simplf(tab_u,gm) #np.array( [ for u_j in tab_u])    #- secon_der(-X,tab_u,gamma_0)
        plt.plot(tab_u, y, label=f'gm={gm}')
    plt.xlabel('u')
    for j in range(len(tab_u)):
        u_j = tab_u[j]
        plt.scatter(u_j, np.array([simplf(u_j,gamma_0=1/2+ (2+1/2)*(1-u_j)) ]) , label=f'1/2+ (j+1/2)*(1-u) = {1/2+ (j+1/2)*(1-u_j)}')
    plt.grid(1)
    plt.yscale('symlog')
    #plt.yscale('log')
    plt.legend()
    plt.show()
trace()

plt.figure()
X = np.linspace(0,10,100)
k = (2*gamma_0-1)/(2-2*u)  + 1
C = v/(1-u)
eps = 1e-5
plt.plot(X, -1j*hyp2f1(1,3/2,k, -1/np.power(eps+ 1j*X/C , 2)) /np.power(eps+1j*X/C,2))
plt.plot(X, -1j / ( 2**(k-1)* np.power(X,2))* hyp2f1(1,3/2,k,C**2/np.power(X,2)))
#plt.yscale('symlog')
plt.show()
print(-1j*hyp2f1(1/2,1,k, -1/np.power(eps+ 1j*X/C , 2)) /(eps+1j*X/C) )