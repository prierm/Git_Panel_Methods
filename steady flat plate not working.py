# -*- coding: utf-8 -*-
"""
steady panel method from HSPM
@author: prierm
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

# setup panel coordinates and control point coordinates
AoA=5*np.pi/180
Uinf=3
chord=.12
thickness=.006
minRad=thickness/2
majRad=chord/4
numPanels=32
panel_coor=np.zeros((numPanels+1,2))

# panel coordinates for a finite thickness plate with rounded leading edges
xStep=chord/(numPanels/2)
panel_coor[0,0]=chord/2
panel_coor[0,1]=0
i=1
while i<numPanels/8:
    panel_coor[i,0]=chord/2-xStep*i
    panel_coor[i,1]=-minRad/majRad*(majRad**2-(majRad-xStep*i)**2)**(1/2)
    i=i+1
    
while i<=numPanels/4+numPanels/8:
    panel_coor[i,0]=chord/2-xStep*i
    panel_coor[i,1]=-minRad
    i=i+1
    
while i<numPanels/2:
    panel_coor[i,0]=chord/2-xStep*i
    panel_coor[i,1]=-minRad/majRad*(majRad**2-(panel_coor[i,0]+(chord/2-majRad))**2)**(1/2)
    i=i+1
panel_coor[i,0]=chord/2-xStep*i
panel_coor[i+1:,0]=panel_coor[i-1::-1,0]
panel_coor[i+1:,1]=-panel_coor[i-1::-1,1]

# collocation points 
col_coor=np.zeros((numPanels,2))
i=0
for panel in range(numPanels):
    col_coor[i,0]=(panel_coor[i,0]+panel_coor[i+1,0])/2
    col_coor[i,1]=(panel_coor[i,1]+panel_coor[i+1,1])/2
    i=i+1
    
# thetas and position vectors
theta=np.zeros((numPanels,1))
r=np.zeros((numPanels,numPanels+1))
beta=np.zeros((numPanels,numPanels))
for i in range(numPanels): # collocation points
    theta[i]=np.arctan2(panel_coor[i+1,1]-panel_coor[i,1],panel_coor[i+1,0]-panel_coor[i,0]) 
    for j in range(numPanels+1): # panel endpoints
        r[i,j]=((col_coor[i,0]-panel_coor[j,0])**2+(col_coor[i,1]-panel_coor[j,1])**2)**(1/2)
        if j>0:
            beta[i,j-1]=np.arctan2((col_coor[i,1]-panel_coor[j,1]),(col_coor[i,0]-panel_coor[j,0]))-\
            np.arctan2((col_coor[i,1]-panel_coor[j-1,1]),(col_coor[i,0]-panel_coor[j-1,0]))

# influence coefficients
infCoef=np.zeros((numPanels+1,numPanels+1))
A=np.zeros((numPanels,numPanels,2))
B=np.zeros((numPanels,numPanels,2))
for i in range(numPanels): # collocation points
    for j in range(numPanels): # panel influence
        if i!=j:
            A[i,j,0]=1/(2*np.pi)*(np.sin(theta[i]-theta[j])*np.log(r[i,j+1]/r[i,j])+np.cos(theta[i]-theta[j])*beta[i,j])
            A[i,j,1]=1/(2*np.pi)*(-np.cos(theta[i]-theta[j])*np.log(r[i,j+1]/r[i,j])+np.sin(theta[i]-theta[j])*beta[i,j]) #!!!!!!!!!!!!!!!
        else:
            A[i,j,0]=1/2
            A[i,j,1]=0       
B[:,:,0]=-A[:,:,1]
B[:,:,1]=A[:,:,0]

infCoef[:numPanels,:numPanels]=A[:,:,0]
infCoef[:numPanels,-1]=np.sum(B[:,:,0],axis=1)
infCoef[-1,:numPanels]=A[0,:,1]+A[-1,:,1]
infCoef[-1,-1]=np.sum(B[0,:,1])+np.sum(B[-1,:,1])

# freestream velocity vector
U=np.zeros((numPanels,2))
U[:,0]=np.transpose(Uinf*np.sin(AoA-theta))
U[:,1]=np.transpose(Uinf*np.cos(AoA-theta))
rho=1.204

# b vector
b=np.zeros((numPanels+1,1))
b[:numPanels,0]=-U[:,0]
b[-1]=-U[0,1]-U[-1,1]

# solve exactly
x=np.linalg.solve(infCoef,b)

#confirm that velocities on airfoil are tangential
vel_foil=np.zeros((numPanels,2))
vel_foil[:,0]=np.reshape(np.dot(A[:,:,0],x[:-1])+np.dot(B[:,:,0],x[-1]*np.ones((numPanels,1)))+np.reshape(U[:,0],(numPanels,1)),(numPanels,))
vel_foil[:,1]=np.reshape(np.dot(A[:,:,1],x[:-1])+np.dot(B[:,:,1],x[-1]*np.ones((numPanels,1)))+np.reshape(U[:,1],(numPanels,1)),(numPanels,))
vel_foil_xy=np.zeros((numPanels,2))
vel_foil_xy[:,0]=-vel_foil[:,0]*np.transpose(np.sin(theta))+vel_foil[:,1]*np.transpose(np.cos(theta))
vel_foil_xy[:,1]=vel_foil[:,0]*np.transpose(np.cos(theta))+vel_foil[:,1]*np.transpose(np.sin(theta))

# Lift and Cl
si=np.zeros((numPanels,1))
perimeter=0
for i in range(numPanels):
    si[i]=((panel_coor[i+1,0]-panel_coor[i,0])**2+(panel_coor[i+1,1]-panel_coor[i,1])**2)**(1/2)
    perimeter=perimeter+si[i]
gamma=x[-1]*perimeter
lift_kutta=rho*Uinf*gamma
Cl_kutta=2*lift_kutta/(rho*Uinf**2*chord)
lift_J=4*np.pi*rho*Uinf**2*chord/4*np.sin(AoA)
Cl_J=2*lift_J/(rho*Uinf**2*chord)

# lift from bernoulli
dP=np.reshape(1/2*rho*(Uinf**2-vel_foil[:,1]**2),(numPanels,1))
yForceDist=-dP*si*np.cos(theta-AoA)
lift_bern=np.sum(yForceDist)
Cl_bern=2*lift_bern/(rho*Uinf**2*chord)

# calculate velocities in the flow field
numPoints=30
xLeft=-.1
xRight=.1
yLower=-.1
yUpper=.1
X,Y=np.meshgrid(np.linspace(xLeft,xRight,numPoints),np.linspace(yLower,yUpper,numPoints))
r_grid=np.zeros((numPoints*numPoints,numPanels+1))
beta_grid=np.zeros((numPoints*numPoints,numPanels))
for i in range(numPoints): # each row of grid points
    for j in range(numPoints): # each grid point in each row
        for k in range(numPanels+1): # each panel endpoint
            r_grid[i*numPoints+j,k]=((X[0,j]-panel_coor[k,0])**2+(Y[i,0]-panel_coor[k,1])**2)**(1/2)
            if k>0:
                beta_grid[i*numPoints+j,k-1]=np.arctan2((Y[i,0]-panel_coor[k,1]),(X[0,j]-panel_coor[k,0]))-\
                np.arctan2((Y[i,0]-panel_coor[k-1,1]),(X[0,j]-panel_coor[k-1,0]))

#   influence coefficients for points outside the foil
A_grid=np.zeros((numPoints*numPoints,numPanels,2))
B_grid=np.zeros((numPoints*numPoints,numPanels,2))
for i in range(numPoints): # each row of grid points
    for j in range(numPoints): # each grid point in each row
        for k in range(numPanels): # each panel
            A_grid[i*numPoints+j,k,0]=1/(2*np.pi)*(np.sin(-theta[k])*np.log(r_grid[i*numPoints+j,k+1]/r_grid[i*numPoints+j,k])+\
                  np.cos(-theta[k])*beta_grid[i*numPoints+j,k])
            A_grid[i*numPoints+j,k,1]=1/(2*np.pi)*(-np.cos(-theta[k])*np.log(r_grid[i*numPoints+j,k+1]/r_grid[i*numPoints+j,k])+\
                  np.sin(-theta[k])*beta_grid[i*numPoints+j,k])    
B_grid[:,:,0]=-A_grid[:,:,1]
B_grid[:,:,1]=A_grid[:,:,0]

#   x and y velocities for points outside the foil
inf_coef_grid=np.zeros((numPoints*numPoints,numPanels+1,2))
inf_coef_grid[:,:-1,0]=A_grid[:,:,0]
inf_coef_grid[:,:-1,1]=A_grid[:,:,1]
inf_coef_grid[:,-1,0]=np.sum(B_grid[:,:,0],axis=1)
inf_coef_grid[:,-1,1]=np.sum(B_grid[:,:,1],axis=1)

velStream=np.zeros((numPoints,numPoints,2))
for i in range(numPoints): # each row of points
    for j in range(numPoints): # each point in row
        velStream[i,j,0]=np.dot(inf_coef_grid[numPoints*i+j,:,0],x)+Uinf*np.sin(AoA)
        velStream[i,j,1]=np.dot(inf_coef_grid[numPoints*i+j,:,1],x)+Uinf*np.cos(AoA)

# plot velocity vectors
fig1=plt.figure(figsize=(12,8))
fig1.add_subplot(111)
plt.ylim(yLower,yUpper)
plt.xlim(xLeft,xRight)
plt.plot(panel_coor[:,0],panel_coor[:,1])
plt.quiver(X,Y,velStream[:,:,1],velStream[:,:,0],color='r')
plt.plot(col_coor[:,0],col_coor[:,1],'*',color='m')
#plt.quiver(col_coor[:,0],col_coor[:,1],vel_foil_xy[:,0],vel_foil_xy[:,1],color='g')
plt.savefig('steady_plate_15_AoA.png')

print('C_L from kutta:',Cl_kutta)
print('C_L from bernoulli:',Cl_bern)
print('C_L from joukowski:',Cl_J)
