# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:11:56 2017

@author: prierm
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

# some initial data
Uinf=3
rho=1.204
chord=.12
thickness=.006
minRad=thickness/2
majRad=chord/4
numPanels=32
xp=np.zeros((numPanels+1,1))
yp=np.zeros((numPanels+1,1))

# panel coordinates for a finite thickness plate with rounded leading edges
xStep=chord/(numPanels/2)
xp[0]=chord/2
yp[0]=0
i=1
while i<numPanels/8:
    xp[i]=chord/2-xStep*i
    yp[i]=-minRad/majRad*(majRad**2-(majRad-xStep*i)**2)**(1/2)
    i=i+1
    
while i<=numPanels/4+numPanels/8:
    xp[i]=chord/2-xStep*i
    yp[i]=-minRad
    i=i+1
    
while i<numPanels/2:
    xp[i]=chord/2-xStep*i
    yp[i]=-minRad/majRad*(majRad**2-(xp[i]+(chord/2-majRad))**2)**(1/2)
    i=i+1
xp[i]=chord/2-xStep*i
xp[i+1:]=xp[i-1::-1]
yp[i+1:]=-yp[i-1::-1]

# collocation points
xc=np.zeros((numPanels,1))
yc=np.zeros((numPanels,1))
for i in range(numPanels):
    xc[i]=(xp[i]+xp[i+1])/2
    yc[i]=(yp[i]+yp[i+1])/2

# time frames, panel positions, collocation positions, induced freestream velocities
numFrames=2  
tInterval=np.linspace(0,.05,numFrames)
f=2
omega=2*np.pi*f
h0=.05
theta0=60*np.pi/180
h_t=h0*np.cos(omega*tInterval)
h_t_dot=-h0*omega*np.sin(omega*tInterval)
theta_t=theta0*-np.sin(omega*tInterval)
theta_t_dot=theta0*omega*-np.cos(omega*tInterval)
numPoints=20
xLeft=-.1
xRight=.1
yLower=-.1
yUpper=.1
X,Y=np.meshgrid(np.linspace(xLeft,xRight,numPoints),np.linspace(yLower,yUpper,numPoints))

# things that don't change
# thetas for each panel
theta=np.zeros((numPanels,1))
for i in range(numPanels):
    theta[i]=np.arctan2(yp[i+1]-yp[i],xp[i+1]-xp[i])
    
# influence coefficients global x y
ups=np.zeros((numPanels,numPanels))
wps=np.zeros((numPanels,numPanels))  

#   local panel coor to global  
for i in range(numPanels):   # collocation points
    for j in range(numPanels):  # panels 
        r1=((xc[i]-xp[j])**2+(yc[i]-yp[j])**2)**(1/2)
        r2=((xc[i]-xp[j+1])**2+(yc[i]-yp[j+1])**2)**(1/2)
        nu2=np.arctan2(-(xc[i]-xp[j])*np.sin(theta[j])+(yc[i]-yp[j])*np.cos(theta[j])+(xp[j+1]-xp[j])*np.sin(theta[j])-(yp[j+1]-yp[j])*np.cos(theta[j]),\
                         (xc[i]-xp[j])*np.cos(theta[j])+(yc[i]-yp[j])*np.sin(theta[j])-(xp[j+1]-xp[j])*np.cos(theta[j])-(yp[j+1]-yp[j])*np.sin(theta[j]))
        nu1=np.arctan2(-(xc[i]-xp[j])*np.sin(theta[j])+(yc[i]-yp[j])*np.cos(theta[j]),(xc[i]-xp[j])*np.cos(theta[j])+(yc[i]-yp[j])*np.sin(theta[j]))
        ups[i,j]=1/(4*np.pi)*np.log(r1**2/r2**2)
        wps[i,j]=1/(2*np.pi)*(nu2-nu1)
np.fill_diagonal(ups,0)
np.fill_diagonal(wps,.5)
upv=copy.copy(wps)
wpv=-ups

#   calculate normal tangential influence coefficients
nSource=-np.sin(theta-np.transpose(theta))*(ups)+np.cos(theta-np.transpose(theta))*(wps)
tSource=np.cos(theta-np.transpose(theta))*(ups)+np.sin(theta-np.transpose(theta))*(wps)
nVortex=-np.sin(theta-np.transpose(theta))*(upv)+np.cos(theta-np.transpose(theta))*(wpv)
tVortex=np.cos(theta-np.transpose(theta))*(upv)+np.sin(theta-np.transpose(theta))*(wpv)

#   matrix A
A=np.zeros((numPanels+1,numPanels+1))
A[:numPanels,:numPanels]=nSource
A[:numPanels,-1]=np.sum(nVortex,axis=1)
A[-1,:numPanels]=tSource[0,:]+tSource[-1,:]
A[-1,-1]=np.sum(tVortex[0,:])+np.sum(tVortex[-1,:])

# data storage
tangVelSto=np.zeros((numPanels,numFrames))
ClKuttaSto=np.zeros((numFrames,1))
ClBernSto=np.zeros((numFrames,1))
 
# panel method for each snap shot in time interval
for t in range(numFrames):     
    # induced velocity and AoA at collocation point
    AoA=theta_t[t]-np.arctan2(h_t_dot[t]+xc*theta_t_dot[t]*np.cos(theta_t[t]),Uinf-theta_t_dot[t]*xc*np.sin(theta_t[t]))#!!!!!!!!!!!!!!
    normU=Uinf*np.sin(AoA-theta)
    tangU=Uinf*np.cos(AoA-theta)
    
    # vector b
    b=np.zeros((numPanels+1,1))
    b[:-1]=-normU
    b[-1]=-tangU[0]-tangU[-1]
    
    # solve exactly
    x=np.linalg.solve(A,b)
    
    #confirm that velocities on airfoil are tangential
#    normVelFoil=np.zeros((numPanels,1))
#    tangVelFoil=np.zeros((numPanels,1))
    normVelFoil=np.dot(nSource,x[:-1])+np.dot(nVortex,x[-1]*np.ones((numPanels,1)))+normU
    tangVelFoil=np.dot(tSource,x[:-1])+np.dot(tVortex,x[-1]*np.ones((numPanels,1)))+tangU
    
    # Lift from kutta
    si=np.zeros((numPanels,1))
    perimeter=0
    for i in range(numPanels):
        si[i]=((xp[i+1]-xp[i])**2+(yp[i+1]-yp[i])**2)**(1/2)
        perimeter=perimeter+si[i]
    gamma=x[-1]*perimeter
    Cl_kutta=2*gamma/(Uinf*chord)
    
    # Lift from conformal mapping
#    Cl_kutta=2*lift_kutta/(rho*Uinf**2*chord)
#    Cl_J=2*np.pi*np.sin(AoA)
    
    # lift from bernoulli
    dP=np.reshape(1/2*rho*(Uinf**2-tangVelFoil**2),(numPanels,1))
    yForceDist=-dP*si*np.cos(theta-AoA)
    lift_bern=np.sum(yForceDist)
    Cl_bern=2*lift_bern/(rho*Uinf**2*chord)
    
    # save data
    tangVelSto[:,t]=tangVelFoil[:,0]
    ClKuttaSto[t]=Cl_kutta
    ClBernSto[t]=Cl_bern
