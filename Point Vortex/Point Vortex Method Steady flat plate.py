# -*- coding: utf-8 -*-
"""
steady panel method from HSPM
@author: prierm
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import copy

# setup panel coordinates and control point coordinates
AoA=5*np.pi/180
Uinf=8.8
rho=1.204
chord=.12
thickness=.01
minRad=thickness/2
majRad=chord/4
numPanels=64
xp=np.zeros((numPanels+1,1))
yp=np.zeros((numPanels+1,1))

# panel coordinates for a finite thickness plate with rounded leading edges
xStep=chord/(numPanels/2)
xp=np.linspace(-chord/2,chord/2,numPanels+1)
yp=np.zeros((numPanels+1,1))
plt.plot(xp,yp)

# vortex points and collocation points
xv=np.zeros((numPanels,1))
yv=np.zeros((numPanels,1))
xc=np.zeros((numPanels,1))
yc=np.zeros((numPanels,1))
for i in range(numPanels):
    xv[i]=xp[i]+(xp[i+1]-xp[i])*1/4
    yv[i]=yp[i]+(yp[i+1]-yp[i])*1/4
    xc[i]=xp[i]+(xp[i+1]-xp[i])*3/4
    yc[i]=yp[i]+(yp[i+1]-yp[i])*3/4

# thetas for each panel
theta=np.zeros((numPanels,1))
for i in range(numPanels):
    theta[i]=np.arctan2(yp[i+1]-yp[i],xp[i+1]-xp[i])

# influence coefficients global foil x y
rj_sqrd=(xc-np.transpose(xv))**2+(yc-np.transpose(yv))**2
up=1/(2*np.pi*rj_sqrd)*(yc-np.transpose(yv))
vp=1/(2*np.pi*rj_sqrd)*-(xc-np.transpose(xv))

# transform to normal and tangential coordinates
nInf=-up*np.sin(theta)+vp*np.cos(theta)
tInf=up*np.cos(theta)+vp*np.sin(theta)

#   velocity vector
normU=Uinf*np.sin(AoA-theta)
tangU=Uinf*np.cos(AoA-theta)

# Solve system
x=sc.linalg.solve(nInf,-normU)
x=np.reshape(x,(numPanels,1))

#confirm that velocities on airfoil are tangential
normVelFoil=np.dot(nInf,x)+normU
tangVelFoil=np.dot(tInf,x)+tangU

# Lift from kutta
si=np.zeros((numPanels,1))
for i in range(numPanels):
    si[i]=((xp[i+1]-xp[i])**2+(yp[i+1]-yp[i])**2)**(1/2)
gamma=np.sum(x)
Cl_kutta=2*gamma/(Uinf*chord)

# lift from bernoulli
dP=np.reshape(1/2*rho*(Uinf**2-tangVelFoil**2),(numPanels,1))
yForceDist=-dP*si*np.cos(theta-AoA)
lift_bern=np.sum(yForceDist)
Cl_bern=2*lift_bern/(rho*Uinf**2*chord)

# Lift from conformal mapping
Cl_J=2*np.pi*np.sin(AoA)

# moment about pitching axis from Bernoulli
momentBern=np.sum(yForceDist*np.cos(AoA)*-xc)

# coefficient of pressure over chord
Cp=1-(tangVelFoil/Uinf)**2
dCp=x/si/(.5*Uinf)

# calculate velocities in the flow field
numPoints=30
xLeft=-.075
xRight=.075
yLower=-.075
yUpper=.075
X,Y=np.meshgrid(np.linspace(xLeft,xRight,numPoints),np.linspace(yLower,yUpper,numPoints))

#   influence coefficients for points outside the foil
upField=np.zeros((numPoints*numPoints,numPanels))
vpField=np.zeros((numPoints*numPoints,numPanels))
for i in range(numPoints):
    for j in range(numPoints):
        for k in range(numPanels):
            rjField_sqrd=(X[0,j]-xv[k])**2+(Y[i,0]-yv[k])**2
            upField[i*numPoints+j,k]=1/(2*np.pi*rjField_sqrd)*(X[0,j]-xv[k])
            vpField[i*numPoints+j,k]=1/(2*np.pi*rjField_sqrd)*-(Y[i,0]-yv[k])

# calculate far field velocities
xVelStream=np.zeros((numPoints,numPoints))
yVelStream=np.zeros((numPoints,numPoints))
for i in range(numPoints):
    for j in range(numPoints):
        xVelStream[i,j]=np.dot(np.reshape(upField[i*numPoints+j,:],(1,numPanels)),x)+Uinf*np.cos(AoA)
        yVelStream[i,j]=np.dot(np.reshape(vpField[i*numPoints+j,:],(1,numPanels)),x)+Uinf*np.sin(AoA)

# plot velocity vectors
fig1=plt.figure(figsize=(12,8))
fig1.add_subplot(111)
plt.ylim(yLower,yUpper)
plt.xlim(xLeft,xRight)
plt.plot(xp,yp)
plt.quiver(X,Y,xVelStream,yVelStream,color='r')
plt.plot(xc,yc,'*',color='m')
plt.savefig('point_vortex_steady.png')

# plot pressure coefficient
fig2=plt.figure(figsize=(12,8))
fig2.add_subplot(111)
plt.plot(xc,Cp,'*')
plt.xlabel('chord',fontsize=18)
plt.ylabel(r'$C_P$',fontsize=18)
plt.savefig('pressure_point_vortex_steady.png')

# plot pressure coefficient
fig3=plt.figure(figsize=(12,8))
fig3.add_subplot(111)
plt.plot(xc,dCp,'*')
plt.xlabel('chord',fontsize=18)
plt.ylabel(r'$\Delta C_P$',fontsize=18)
plt.savefig('delta_pressure_point_vortex_steady.png')

print('C_L from conformal:',Cl_J)
print('C_L from kutta:',Cl_kutta)
print('C_L from bernoulli:',Cl_bern)
print('moment about pivot: ',momentBern)
