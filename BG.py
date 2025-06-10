 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:57:39 2022

@author: viot
"""
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.special import iv
from numba import jit


def omega(u,Tx,Ty):                                                     ##vitesse angulaire
    return(u*(Tx-Ty)*np.sqrt((1-u**2)/(4*Tx*Ty+u**2*(Tx-Ty)**2)))

@jit
def traj(r,T1,T2,Deltat,u,n2,gnx_sample,gny_sample):
     for i in np.arange(1,n2):
            r[i,0]=r[i-1,0]+Deltat*(-r[i-1,0]+u*r[i-1,1])+np.sqrt(2*T1*Deltat)*gnx_sample[i]
            r[i,1]=r[i-1,1]+Deltat*(-r[i-1,1]+u*r[i-1,0])+np.sqrt(2*T2*Deltat)*gny_sample[i]
     return(r)



def Pomega(x,u,Tx,Ty):
     return(np.sqrt(1-u**2)*(Tx+Ty)/np.sqrt(4*Tx*Ty+u**2*(Tx-Ty)**2)/np.abs(x)**3) 


def PL(x,u,Tx,Ty):
     return(1/np.sqrt(8*(Tx*Ty))*np.exp(-np.abs(x)/(np.sqrt(2*(Tx*Ty)-(np.sign(x)+3)*u*(Tx-Ty)/np.sqrt(1-u**2))  ) ) ) 

def PdeI(x,u,Tx,Ty):
    factor=(4*Tx*Ty+u**2*(Tx-Ty)**2)
    factor2=np.sqrt(u**2/factor+(1-u**2)*(Tx-Ty)**2/factor**2) 
    return(np.sqrt((1-u**2)/factor)*np.exp(-x*(Tx+Ty)/factor)*iv(0,factor2*x))
    

def compa(tabL,tabLbis,tabang,tabangbis,Tx,Ty,cc):
    fig,ax=plt.subplots(1,2,figsize=(12,6)) 
    # tabang=tabang[tabang>0]
    # tabang=np.log(tabang)
    ax[0].hist(tabL,bins=200,density=True,range=(-50,50),log=True,label=r'$T_1={}\, T_2={}$ formula'.format(Tx,Ty),histtype='step')
    ax[0].hist(tabLbis,bins=200,range=(-50,50),density=True,log=True,label=r'$T_1={}\, T_2={} angle$'.format(Tx,Ty),histtype='step',color='black')
    ax[1].hist(tabang,bins=400,range=(-100,100),density=True,log=True,label=r'$T_1={}\, T_2={} angle$'.format(Tx,Ty),histtype='step')
    ax[1].hist(tabangbis,bins=2000,range=(-500,500),density=True,log=True,label=r'$T_1={}\, T_2={}$ formula'.format(Tx,Ty),histtype='step',color='black') 

    if iu== 0.5: 
        exact=np.loadtxt('PdeLTy5u0.5.dat')    
        ax[0].plot(exact[:,0],exact[:,1],'g-.',lw=3)
    if iu== 0.2: 
        exact=np.loadtxt('PdeLTy5u0.2.dat')
        ax[0].plot(exact[:,0],exact[:,1],'g-.',lw=3)    
    if iu== 0.0: 
        exact=np.loadtxt('PdeLTy5u0.0.dat')
        ax[0].plot(exact[:,0],exact[:,1],'g-.',lw=3,)  
    if iu== 0.8: 
            exact=np.loadtxt('PdeLTy5u0.8.dat')
            ax[0].plot(exact[:,0],exact[:,1],'g-.',lw=3,)  
    ax[0].legend()
    ax[0].set_xlim(-50,50)
    ax[0].set_xlabel("L")
    ax[0].set_ylabel("P(L)")
    #ax[0].set_ylim(0.0000001,0.16)
    ax[0].set_ylim(0.0000001,0.16)
    ax[1].legend()
    x=np.geomspace(3,80,60)
    ax[1].plot(x,Pomega(x,iu,Tx,Ty),color='red')
    ax[1].plot(-x,Pomega(-x,iu,Tx,Ty),color='red')
    ax[1].set_xlabel(r'$\omega$')
    ax[1].set_ylabel(r"$P(\omega)$")
    ax[1].set_ylim(0.00001,0.6)
    ax[1].set_xlim(-50,50)
    plt.title('u={}'.format(iu))
    plt.savefig("histo_u{}_Ty{}.pdf".format(iu,Ty))
    
def compa2(tabI,Tx,Ty,cc):
    fig2,ax2=plt.subplots(1,1,figsize=(6,6)) 
    # tabang=tabang[tabang>0]
    # tabang=np.log(tabang)
    ax2.hist(tabI,bins=100,density=True,log=True,range=(0,50),label=r'$T_1={}\, T_2={}$'.format(Tx,Ty),histtype='step')
    x=np.linspace(0,45,100)
    ax2.set_xlim(0,49)
    ax2.plot(x,PdeI(x,iu,Tx,Ty),'r-.',lw=4)
    ax2.legend()
        
    ax2.set_xlabel("I")
    ax2.set_ylabel("P(I)")
    ax2.set_ylim(0.00001,0.6)
    plt.title('u={}'.format(iu))
    plt.savefig("histoI_u{}_Ty{}.pdf".format(iu,Ty))


def display(r):
    plt.figure()
    plt.scatter(r[0::1000,0], r[0::1000,1])
    plt.show()

def expI(x,Tmoy):
    return(np.exp(-x/Tmoy)/Tmoy)


n2=4096*128*4
Deltat=0.01
T=Deltat*n2
Tx=1
Ty=5
#u=np.linspace(-0.8,0.8,40)
cc=['red','orange','green','brown','cyan','blue','black']

nrep=1


tabL=np.empty(0)
tabLbis=np.empty(0)
tabang=np.empty(0)
tabangbis=np.empty(0)
tabI=np.empty(0)
iu=0.0
start=time.time() 
for irep in np.arange(nrep): 
    r=np.zeros(shape=(n2,2))
    gnx_sample = np.random.normal(size=n2)
    gny_sample = np.random.normal(size=n2)
    r=traj(r,Tx,Ty,Deltat,iu,n2,gnx_sample,gny_sample)    
    deltatheta=np.diff(np.arctan2(r[1000:n2,1],r[1000:n2,0])) 
    deltatheta[deltatheta<-np.pi]+=2*np.pi
    deltatheta[deltatheta>np.pi]+=-2*np.pi
    deltatheta/=Deltat
    #deltatheta/=Delta
    
    r2=r[1000:n2,0]**2+r[1000:n2,1]**2
    r2b=(r[1000:n2,0]**2-r[1000:n2,1]**2)
    rtrunc=r[1000:n2]
    deltaangbis=0.5*(r2[:-1]+r2[1:])*deltatheta
    deltaang=iu*r2b+np.sqrt(2*Ty)*gny_sample[1000:n2]*rtrunc[:,0]-np.sqrt(2*Tx)*gnx_sample[1000:n2]*rtrunc[:,1]
    deltathetabis=deltaang/r2
    #deltaangbis=0.5*(r2[:-1]+r2[1:])*deltatheta
    

    tabL=np.append(tabL,deltaang)
    tabLbis=np.append(tabLbis,deltaangbis)
    tabI=np.append(tabI,r2)
    tabang=np.append(tabang,deltatheta)
    
    tabangbis=np.append(tabangbis,deltathetabis)
print(np.min(tabang),np.max(tabang))
print(np.min(tabangbis),np.max(tabangbis))

print("elapsed time",time.time()-start)
print(tabL.mean(),iu*(Tx-Ty))
print(tabLbis.mean()/np.sqrt(Deltat),iu*(Tx-Ty))
print("omega",tabang.mean()/np.sqrt(Deltat),omega(iu,Tx,Ty))
print("omegabis",tabangbis.mean())

display(r)
#compa(tabL,tabLbis,tabang,tabangbis,Tx,Ty,cc)  

#compa2(tabI,Tx,Ty,cc)
