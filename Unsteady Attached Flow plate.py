# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:11:56 2017

@author: prierm
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

# some initial data
pathName='E://Research//Scripts//Potential Flow//Panel Method//Git_Panel_Methods//frames'
Uinf=8.8
rho=1.204
chord=.12
thickness=.006
minRad=thickness/2
majRad=chord/4
numPanels=64
xp=np.zeros((numPanels+1,1))
yp=np.zeros((numPanels+1,1))

# panel coordinates for a finite thickness plate with rounded leading edges
xStep=chord/(numPanels/2)
xp[0]=chord/2
yp[0]=0
i=1
while i<numPanels//8:
    xp[i]=chord/2-xStep*i
    yp[i]=-minRad/majRad*(majRad**2-(majRad-xStep*i)**2)**(1/2)
    i=i+1
    
while i<=numPanels//4+numPanels//8:
    xp[i]=chord/2-xStep*i
    yp[i]=-minRad
    i=i+1
    
while i<numPanels//2:
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
numFrames=12
tEnd=.5
tInterval=np.linspace(0,tEnd,numFrames)
f=2
omega=2*np.pi*f
h0=.1
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
    
# influence coefficients panel coordinates
ups=np.zeros((numPanels,numPanels))
wps=np.zeros((numPanels,numPanels))  
  
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

#   airfoil perimeter
si=np.zeros((numPanels,1))
perimeter=0
for i in range(numPanels):
        si[i]=((xp[i+1]-xp[i])**2+(yp[i+1]-yp[i])**2)**(1/2)
        perimeter=perimeter+si[i]
 
#   calculate velocities in the flow field
upsField=np.zeros((numPoints*numPoints,numPanels))
wpsField=np.zeros((numPoints*numPoints,numPanels))

#   influence coefficients for points outside the foil
for i in range(numPoints): # each row of grid points
    for j in range(numPoints): # each grid point in each row
        for k in range(numPanels): # for each panel
            r1=((X[0,j]-xp[k])**2+(Y[i,0]-yp[k])**2)**(1/2)
            r2=((X[0,j]-xp[k+1])**2+(Y[i,0]-yp[k+1])**2)**(1/2)
            nu2=np.arctan2(-(X[0,j]-xp[k])*np.sin(theta[k])+(Y[i,0]-yp[k])*np.cos(theta[k])+(xp[k+1]-xp[k])*np.sin(theta[k])-(yp[k+1]-yp[k])*np.cos(theta[k]),\
                         (X[0,j]-xp[k])*np.cos(theta[k])+(Y[i,0]-yp[k])*np.sin(theta[k])-(xp[k+1]-xp[k])*np.cos(theta[k])-(yp[k+1]-yp[k])*np.sin(theta[k]))
            nu1=np.arctan2(-(X[0,j]-xp[k])*np.sin(theta[k])+(Y[i,0]-yp[k])*np.cos(theta[k]),(X[0,j]-xp[k])*np.cos(theta[k])+(Y[i,0]-yp[k])*np.sin(theta[k]))
            upsField[i*numPoints+j,k]=1/(4*np.pi)*np.log(r1**2/r2**2)
            wpsField[i*numPoints+j,k]=1/(2*np.pi)*(nu2-nu1)
upvField=copy.copy(wpsField)
wpvField=-upsField

#   transform influence coefficients to cartesian
ugsField=upsField*np.cos(np.transpose(theta))-wpsField*np.sin(np.transpose(theta))
wgsField=upsField*np.sin(np.transpose(theta))+wpsField*np.cos(np.transpose(theta))
ugvField=upvField*np.cos(np.transpose(theta))-wpvField*np.sin(np.transpose(theta))
wgvField=upvField*np.sin(np.transpose(theta))+wpvField*np.cos(np.transpose(theta))

#   x and y influence coefficients for points outside the foil
AFieldx=np.zeros((numPoints*numPoints,numPanels+1))
AFieldy=np.zeros((numPoints*numPoints,numPanels+1))
AFieldx[:,:numPanels]=ugsField
AFieldx[:,-1]=np.sum(ugvField,axis=1)
AFieldy[:,:numPanels]=wgsField
AFieldy[:,-1]=np.sum(wgvField,axis=1)

xVelStream=np.zeros((numPoints,numPoints))
yVelStream=np.zeros((numPoints,numPoints))  
      
# data storage
tangVelSto=np.zeros((numPanels,numFrames))
ClKuttaSto=np.zeros((numFrames,1))
CYKuttaSto=np.zeros((numFrames,1))
ClBernSto=np.zeros((numFrames,1))
CYBernSto=np.zeros((numFrames,1))
xSto=np.zeros((numPanels+1,numFrames))

# panel method for each snap shot in time interval
for t in range(numFrames):     
    # induced velocity and AoA at collocation point
    AoA=theta_t[t]-np.arctan2(h_t_dot[t]-xc*theta_t_dot[t]*np.cos(theta_t[t]),Uinf+theta_t_dot[t]*xc*np.sin(theta_t[t]))#!!!!!!!!!!!!!!
    normU=Uinf*np.sin(AoA-theta)
    tangU=Uinf*np.cos(AoA-theta)
    
    # vector b
    b=np.zeros((numPanels+1,1))
    b[:-1]=-normU
    b[-1]=-tangU[0]-tangU[-1]
    
    # solve exactly
    x=np.linalg.solve(A,b)
    
    #confirm that velocities on airfoil are tangential
    normVelFoil=np.dot(nSource,x[:-1])+np.dot(nVortex,x[-1]*np.ones((numPanels,1)))+normU
    tangVelFoil=np.dot(tSource,x[:-1])+np.dot(tVortex,x[-1]*np.ones((numPanels,1)))+tangU
    
    # Lift from kutta
    gamma=x[-1]*perimeter
    Cl_kutta=2*gamma/(Uinf*chord)
    CY_kutta=Cl_kutta*np.cos(np.average(AoA))
    
    # lift from bernoulli
    dP=np.reshape(1/2*rho*(Uinf**2-tangVelFoil**2),(numPanels,1))
    yForceDist=-dP*si*np.cos(theta-AoA)
    lift_bern=np.sum(yForceDist)
    Cl_bern=2*lift_bern/(rho*Uinf**2*chord)
    CY_bern=Cl_bern*np.cos(np.average(AoA))
    
    # moment about pitching axis
    
    
    # adjust freestream velocity in domain to account for heaving and pitching
    xModU=(Uinf+theta_t_dot[t]*X[0,:]*np.sin(theta_t[t]))*np.cos(theta_t[t])+(-h_t_dot[t]+theta_t_dot[t]*X[0,:]*np.cos(theta_t[t]))*-np.sin(theta_t[t])
    yModU=(Uinf+theta_t_dot[t]*X[0,:]*np.sin(theta_t[t]))*np.sin(theta_t[t])+(-h_t_dot[t]+theta_t_dot[t]*X[0,:]*np.cos(theta_t[t]))*np.cos(theta_t[t])
    
    # calculate velocity stream vectors 
    for i in range(numPoints): # each row of points
        for j in range(numPoints): # each point in row
            xVelStream[i,j]=np.dot(AFieldx[i*numPoints+j,:],x)+xModU[j]
            yVelStream[i,j]=np.dot(AFieldy[i*numPoints+j,:],x)+yModU[j]
    
    # plot velocity vectors
    title='t/T = ' + '{:3f}'.format(tInterval[t]/tEnd)+ r' , $\theta_p = $' + '{:3f}'.format(theta_t[t]/np.pi*180) +' deg , ' + r'$C_L = $' + '{:3f}'.format(Cl_kutta[0]) +\
                      r' , $C_Y = $' + '{:3f}'.format(CY_kutta[0])
    fig1=plt.figure(figsize=(12,8))
    fig1.suptitle(title,fontsize=14)
    fig1.add_subplot(111)
    plt.ylim(yLower,yUpper)
    plt.xlim(xLeft,xRight)
    plt.plot(xp,yp)
    plt.quiver(X,Y,xVelStream,yVelStream,color='r')
    plt.plot(xc,yc,'*',color='m')
    plt.savefig(pathName + '//frame' + str(t))
    plt.close()
    
    # save data
    tangVelSto[:,t]=tangVelFoil[:,0]
    ClKuttaSto[t]=Cl_kutta
    CYKuttaSto[t]=Cl_kutta*np.cos(np.average(AoA))
    ClBernSto[t]=Cl_bern
    CYBernSto[t]=Cl_bern*np.cos(np.average(AoA))
    xSto[:,t]=x[:,0]

plt.plot(tInterval,ClKuttaSto)