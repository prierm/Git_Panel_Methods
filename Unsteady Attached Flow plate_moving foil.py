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
Uinf=3
rho=1.204
chord=.12
thickness=.006
minRad=thickness/2
majRad=chord/4
numPanels=64
xp=np.zeros((numPanels+1,1))
yp=np.zeros((numPanels+1,1))

# panel coordinates in terms of global x and y
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

xpOld=copy.copy(xp)
ypOld=copy.copy(yp)
xcOld=copy.copy(xc)
ycOld=copy.copy(yc)

# time frames, panel positions, collocation positions, induced freestream velocities
numFrames=3
tEnd=.1
tInterval=np.linspace(0,tEnd,numFrames)
f=2
omega=2*np.pi*f
h0=.0375
theta0=60*np.pi/180
h_t=h0*np.cos(omega*tInterval)
h_t_dot=-h0*omega*np.sin(omega*tInterval)
theta_t=theta0*-np.sin(omega*tInterval)
theta_t_dot=theta0*omega*-np.cos(omega*tInterval)
numPoints=20
xLeft=-.15
xRight=.15
yLower=-.15
yUpper=.15
X,Y=np.meshgrid(np.linspace(xLeft,xRight,numPoints),np.linspace(yLower,yUpper,numPoints))

# things that don't change
# thetas for each panel relative to the foil
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

#   matrix A (everything in normal tangential coordinates)
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

xVelStream=np.zeros((numPoints,numPoints))
yVelStream=np.zeros((numPoints,numPoints)) 

# data storage
tangVelSto=np.zeros((numPanels,numFrames))
ClKuttaSto=np.zeros((numFrames,1))
CYKuttaSto=np.zeros((numFrames,1))
CYBernSto=np.zeros((numFrames,1))
xSto=np.zeros((numPanels+1,numFrames))
momSto=np.zeros((numFrames,1))

# panel method for each snap shot in time interval
for t in range(numFrames):
    # freestream in terms of normal tangential coordinates
    normU=(-np.sin(theta)*np.cos(theta_t[t])+np.cos(theta)*np.sin(theta_t[t]))*Uinf#!!!!!!!!!!!!
    tangU=(np.cos(theta)*np.cos(theta_t[t])+np.sin(theta_t[t])*np.sin(theta))*Uinf
    
    #foil contribution
    foilVelNorm=-h_t_dot[t]*np.sin(theta_t[t])*np.sin(theta)+(-h_t_dot[t]*np.cos(theta_t[t])-theta_t_dot[t]*-xc)*np.cos(theta)
    foilVelTang=h_t_dot[t]*np.sin(theta_t[t])*np.cos(theta)+(-h_t_dot[t]*np.cos(theta_t[t])-theta_t_dot[t]*-xc)*np.sin(theta)
    
    # vector b
    b=np.zeros((numPanels+1,1))
    b[:-1]=-(normU+foilVelNorm)
    b[-1]=(-tangU[0]-tangU[-1])+(-foilVelTang[0]-foilVelTang[-1])
    
    # solve exactly
    x=np.linalg.solve(A,b)
    
    #confirm that velocities on airfoil are tangential
    normVelFoil=np.dot(nSource,x[:-1])+np.dot(nVortex,x[-1]*np.ones((numPanels,1)))+(normU+foilVelNorm)
    tangVelFoil=np.dot(tSource,x[:-1])+np.dot(tVortex,x[-1]*np.ones((numPanels,1)))+(tangU+foilVelTang)
    
    # Lift from kutta
    gamma=x[-1]*perimeter
    Cl_kutta=2*gamma/(Uinf*chord)
    CY_kutta=Cl_kutta*np.cos(-np.arctan2(h_t_dot[t],Uinf))#!!!!!!!!!!!!!!!!!!!!
    
    # lift from bernoulli
    dP=np.reshape(1/2*rho*(Uinf**2-tangVelFoil**2),(numPanels,1))
    forceDist=-dP*si#!!!!!!!!!!!!!!!!!!!
    Fy_bern=np.sum(forceDist*(np.sin(theta_t[t])*np.sin(theta)+np.cos(theta_t[t])*np.cos(theta)))
    CY_bern=2*Fy_bern/(rho*Uinf**2*chord)
    
    # moment from bernoulli
    momentBern=np.sum(forceDist*np.cos(theta)*-xc)#!!!!!!!!!!!!!!!!
    
    # coefficient of pressure over chord
    
    # update panel coordinates 
    xp=xpOld+xpOld*np.cos(theta_t[t])
    yp=ypOld+h_t[t]-xpOld*np.sin(theta_t[t])
    
    #   influence coefficients for points outside the foil in panel coor sys
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
    
    #   transform influence coefficients to foil frame
    ugsField=upsField*np.cos(np.transpose(theta))-wpsField*np.sin(np.transpose(theta))
    wgsField=upsField*np.sin(np.transpose(theta))+wpsField*np.cos(np.transpose(theta))
    ugvField=upvField*np.cos(np.transpose(theta))-wpvField*np.sin(np.transpose(theta))
    wgvField=upvField*np.sin(np.transpose(theta))+wpvField*np.cos(np.transpose(theta))
    
    #   transform to global X Y
    upsField=copy.copy(ugsField)
    wpsField=copy.copy(wgsField)
    upvField=copy.copy(ugvField)
    wpvField=copy.copy(wgvField)
    
    ugsField=upsField*np.cos(theta_t[t])+wpsField*np.sin(theta_t[t])
    wgsField=-upsField*np.sin(theta_t[t])+wpsField*np.cos(theta_t[t])
    ugvField=upvField*np.cos(theta_t[t])+wpvField*np.sin(theta_t[t])
    wgvField=-upvField*np.sin(theta_t[t])+wpvField*np.cos(theta_t[t])
    
    #   x and y influence coefficients for points outside the foil global X Y
    AFieldx=np.zeros((numPoints*numPoints,numPanels+1))
    AFieldy=np.zeros((numPoints*numPoints,numPanels+1))
    AFieldx[:,:numPanels]=ugsField
    AFieldx[:,-1]=np.sum(ugvField,axis=1)
    AFieldy[:,:numPanels]=wgsField
    AFieldy[:,-1]=np.sum(wgvField,axis=1)
    
    # adjust freestream velocity in domain to account for heaving and pitching (global X Y)
    xModU=Uinf-theta_t_dot[t]*-X*np.cos(np.arctan2(Y-h_t[t],-X)-theta_t[t])*np.sin(theta_t[t])
    yModU=-theta_t_dot[t]*-X*np.cos(np.arctan2(Y-h_t[t],-X)-theta_t[t])*np.cos(theta_t[t])-h_t_dot[t]
    
    # calculate velocity stream vectors in global X Y
    for i in range(numPoints): # each row of points
        for j in range(numPoints): # each point in row
            xVelStream[i,j]=np.dot(AFieldx[i*numPoints+j,:],x)+xModU[i,j]
            yVelStream[i,j]=np.dot(AFieldy[i*numPoints+j,:],x)+yModU[i,j]
    
    # plot velocity vectors
    title='t/T = ' + '{:3f}'.format(tEnd/.5)+ r' , $\theta_p = $' + '{:3f}'.format(theta_t[t]/np.pi*180) +' deg , ' + r'$C_L = $' + '{:3f}'.format(Cl_kutta[0]) +\
                      r' , $C_Y = $' + '{:3f}'.format(CY_kutta[0])
    fig1=plt.figure(figsize=(12,8))
    fig1.suptitle(title,fontsize=14)
    fig1.add_subplot(111)
    plt.ylim(yLower,yUpper)
    plt.xlim(xLeft,xRight)
    plt.plot(xp,yp)
    plt.quiver(X,Y,xVelStream,yVelStream,color='r')
    plt.savefig(pathName + '//frame' + str(t))
    plt.close()
    
    # save data
    tangVelSto[:,t]=tangVelFoil[:,0]
    ClKuttaSto[t]=Cl_kutta
    CYKuttaSto[t]=CY_kutta
    CYBernSto[t]=CY_bern
    xSto[:,t]=x[:,0]
    momSto[t]=momentBern

# plot force in heaving direction
fig2=plt.figure(figsize=(12,8))
fig2.add_subplot(111)
plt.plot(tInterval/tInterval[-1],CYKuttaSto)
plt.plot(tInterval/tInterval[-1],h_t/h0)
plt.legend([r'$C_Y$',r'$h/h_0$'],loc='lower right',fontsize=14)
plt.xlabel('t/T',fontsize=18)
plt.ylabel(r'$C_Y$',fontsize=18)
plt.title('From Kutta')
#plt.savefig('quasi_steady.png')

fig3=plt.figure(figsize=(12,8))
fig3.add_subplot(111)
plt.plot(tInterval/tInterval[-1],CYBernSto)
plt.plot(tInterval/tInterval[-1],h_t/h0)
plt.legend([r'$C_Y$',r'$h/h_0$'],loc='lower right',fontsize=14)
plt.xlabel('t/T',fontsize=18)
plt.ylabel(r'$C_Y$',fontsize=18)
plt.title('From Bernoulli')
#plt.savefig('quasi_steady.png')

