# -*- coding: utf-8 -*-
"""
steady panel method from HSPM
@author: prierm
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

# setup panel coordinates and control point coordinates
AoA=5*np.pi/180
Uinf=3
rho=1.204
chord=.12
thickness=.006
minRad=thickness/2
majRad=chord/4
numPanels=32
panel_coor=np.zeros((numPanels+1,2))
xp=np.zeros((numPanels+1,1))
yp=np.zeros((numPanels+1,1))

# panel coordinates for a finite thickness plate with rounded leading edges
xp[:,0]=chord/2*(np.cos(np.linspace(0,2*np.pi,numPanels+1))+1)
xp[:,0]=xp[:,0]-chord/2
yp[:numPanels//2+1]=-thickness*.385*(1-2*xp[:numPanels//2+1]/chord)*(1-(2*xp[:numPanels//2+1]/chord)**2)**(1/2)
yp[numPanels//2+1:]=thickness*.385*(1-2*xp[numPanels//2+1:]/chord)*(1-(2*xp[numPanels//2+1:]/chord)**2)**(1/2)

# collocation points
xc=np.zeros((numPanels,1))
yc=np.zeros((numPanels,1))
for i in range(numPanels):
    xc[i]=(xp[i]+xp[i+1])/2
    yc[i]=(yp[i]+yp[i+1])/2

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

#   velocity vector
normU=Uinf*np.sin(AoA-theta)
tangU=Uinf*np.cos(AoA-theta)

#   matrix A
A=np.zeros((numPanels+1,numPanels+1))
A[:numPanels,:numPanels]=nSource
A[:numPanels,-1]=np.sum(nVortex,axis=1)
A[-1,:numPanels]=tSource[0,:]+tSource[-1,:]
A[-1,-1]=np.sum(tVortex[0,:])+np.sum(tVortex[-1,:])

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
si=np.zeros((numPanels,1))
perimeter=0
for i in range(numPanels):
    si[i]=((xp[i+1]-xp[i])**2+(yp[i+1]-yp[i])**2)**(1/2)
    perimeter=perimeter+si[i]
gamma=x[-1]*perimeter
Cl_kutta=2*gamma/(Uinf*chord)

# Lift from conformal mapping
Cl_J=2*np.pi*(1+.77*thickness/chord)*np.sin(AoA)

# lift from bernoulli
dP=np.reshape(1/2*rho*(Uinf**2-tangVelFoil**2),(numPanels,1))
yForceDist=-dP*si*np.cos(theta-AoA)
lift_bern=np.sum(yForceDist)
Cl_bern=2*lift_bern/(rho*Uinf**2*chord)

# moment from bernoulli
momentBern=np.sum(yForceDist*np.cos(AoA)*-xc)

# coefficient of pressure over chord, 1st column collocation point, 2nd column Cp
Cp=dP/(1/2*rho*Uinf**2)

# calculate velocities in the flow field
numPoints=20
xLeft=-.075
xRight=.075
yLower=-.075
yUpper=.075
X,Y=np.meshgrid(np.linspace(xLeft,xRight,numPoints),np.linspace(yLower,yUpper,numPoints))
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

#   x and y velocities for points outside the foil
AFieldx=np.zeros((numPoints*numPoints,numPanels+1))
AFieldy=np.zeros((numPoints*numPoints,numPanels+1))
AFieldx[:,:numPanels]=ugsField
AFieldx[:,-1]=np.sum(ugvField,axis=1)
AFieldy[:,:numPanels]=wgsField
AFieldy[:,-1]=np.sum(wgvField,axis=1)

xVelStream=np.zeros((numPoints,numPoints))
yVelStream=np.zeros((numPoints,numPoints))
for i in range(numPoints): # each row of points
    for j in range(numPoints): # each point in row
        xVelStream[i,j]=np.dot(AFieldx[i*numPoints+j,:],x)+Uinf*np.cos(AoA)
        yVelStream[i,j]=np.dot(AFieldy[i*numPoints+j,:],x)+Uinf*np.sin(AoA)

# plot velocity vectors
fig1=plt.figure(figsize=(12,8))
fig1.add_subplot(111)
plt.ylim(yLower,yUpper)
plt.xlim(xLeft,xRight)
plt.plot(xp,yp)
plt.quiver(X,Y,xVelStream,yVelStream,color='r')
plt.plot(xc,yc,'*',color='m')
#plt.savefig('steady_plate_5_AoA_.png')

# plot pressure coefficient
fig2=plt.figure(figsize=(12,8))
fig2.add_subplot(111)
plt.plot(xc[:numPanels//2+1],Cp[:numPanels//2+1])
plt.plot(xc[numPanels//2:],Cp[numPanels//2:])
plt.xlabel('chord',fontsize=14)
plt.ylabel(r'$C_P$',fontsize=14)

print('C_L from kutta:',Cl_kutta)
print('C_L from bernoulli:',Cl_bern)
print('C_L from conformal mapping:',Cl_J)
print('moment about pivot: ',momentBern)
