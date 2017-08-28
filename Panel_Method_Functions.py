# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:01:19 2017
Panel_Method_Functions
@author: prierm
"""

def Airfoil_Setup(functUpper=None,functLower=None,xInterval=None,dataInput=None):
    """Sets up panel coordinates and collocation points.
    3 ways to use the function: 
    no input (default)m
    math function
        pass functUpper and functLower a string expression with variables x and y
        pass xInterval as numpy array
    input for upper and lower surface, and data input."""
    
    from sympy import lambdify
    from sympy.parsing.sympy_parser import parse_expr
    from sympy import symbols
    import numpy as np
    import matplotlib.pyplot as plt
    if functUpper != None: # THIS DOES NOT WORK!!!
        x,y=symbols('x y')
        functUpper=parse_expr(functUpper,local_dict={'x':x,'y':y})
        functLower=parse_expr(functLower,local_dict={'x':x,'y':y})
        numPanels=(len(xInterval)-1)*2
        panel_coor=np.zeros((numPanels+1,2))
        panel_coor[:int(numPanels/2)+1,0]=xInterval[::-1]
        panel_coor[int(numPanels/2):,0]=xInterval
        lambFunctUpper=lambdify(x,functUpper,'numpy')
        lambFunctLower=lambdify(x,functLower,'numpy')
        panel_coor[:int(numPanels/2)+1,1]=lambFunctUpper(panel_coor[:int(numPanels/2)+1,0])
        panel_coor[int(numPanels/2):,1]=lambFunctLower(panel_coor[:int(numPanels/2)+1,0])
        plt.plot(panel_coor[:,0],panel_coor[:,1])
        
    elif dataInput!=None:
        panel_coor=dataInput
        numPanels=len(dataInput)-1
        col_coor=np.zeros((numPanels,2))
        for i in range(numPanels):
            col_coor[i,0]=(panel_coor[i,0]+panel_coor[i+1,0])/2
            col_coor[i,1]=(panel_coor[i,1]+panel_coor[i+1,1])/2
            
    else:
        # setup panel coordinates and control point coordinatesTHIS IS RIGHT
        chord=.12
        thickness=.006
        minRad=thickness/2
        majRad=chord/4
        numPanels=128
        panel_coor=np.zeros((numPanels+1,2))
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
        plt.plot(panel_coor[:,0],panel_coor[:,1])
        
    return panel_coor,col_coor,numPanels

def Compute_Coeff(panel_coor,col_coor,numPanels,Uinf,AoA):
    """This function computes the influence coefficients based on impermability andkutta condition"""
    import numpy as np
    # thetas and position vectors
    theta=np.zeros((numPanels,1))
    r=np.zeros((numPanels,numPanels+1))
    beta=np.zeros((numPanels,numPanels))
    
    for i in range(numPanels):
        theta[i]=np.arctan2(panel_coor[i+1,1]-panel_coor[i,1],panel_coor[i+1,0]-panel_coor[i,0]) 
        for j in range(numPanels+1): 
            r[i,j]=((col_coor[i,0]-panel_coor[j,0])**2+(col_coor[i,1]-panel_coor[j,1])**2)**(1/2)
        for j in range(numPanels): 
            beta[i,j]=np.arctan2((col_coor[i,1]-panel_coor[j+1,1]),(col_coor[i,0]-panel_coor[j+1,0]))-np.arctan2((col_coor[i,1]-panel_coor[j,1]),(col_coor[i,0]-panel_coor[j,0]))

    # crazy shit I AM NOT SURE THIS IS RIGHT BUT I AM KEEPING IT FOR NOW...
    beta[:,int(numPanels/2):]=beta[::-1,int(numPanels/2)-1::-1]
    
    # influence coefficients
    infCoef=np.zeros((numPanels+1,numPanels+1))
    A=np.zeros((numPanels,numPanels,2))
    B=np.zeros((numPanels,numPanels,2))
    for i in range(numPanels):
        for j in range(numPanels):
            if i!=j:
                A[i,j,0]=1/(2*np.pi)*(np.sin(theta[i]-theta[j])*np.log(r[i,j+1]/r[i,j])+np.cos(theta[i]-theta[j])*beta[i,j])
                A[i,j,1]=1/(2*np.pi)*(-np.cos(theta[i]-theta[j])*np.log(r[i,j+1]/r[i,j])+np.sin(theta[i]-theta[j])*beta[i,j])
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
    
    # b vector
    b=np.zeros((numPanels+1,1))
    b[:numPanels,0]=-U[:,0]
    b[-1]=-U[0,1]-U[-1,1]
    
    # solve exactly
    x=np.linalg.solve(infCoef,b)
    
    return x,b,A,B,infCoef,U,theta,r,beta

def Comp_Lift(numPanels,panel_coor,x,Uinf,chord,U,AoA,rho):
    import numpy as np
    # Lift and Cl
    rho=1.204
    si=np.zeros((numPanels,1))
    perimeter=0
    i=0
    for panel in range(numPanels):
        si[i]=((panel_coor[i+1,0]-panel_coor[i,0])**2+(panel_coor[i+1,1]-panel_coor[i,1])**2)**(1/2)
        perimeter=perimeter+si[i]
        i=i+1
    gamma=x[-1]*perimeter
    lift=rho*Uinf*gamma
    lift_flat_plate_j=4*np.pi*rho*Uinf**2*chord/4*np.sin(AoA)
    
    # lift from bernoulli
    vel_t=np.zeros((numPanels,1))
    vel_t[:,0]=np.reshape(np.dot(A[:,:,1],x[:-1])+np.dot(B[:,:,1],x[-1]*np.ones((numPanels,1)))+np.reshape(U[:,1],(numPanels,1)),(numPanels,))
    dP=np.reshape(1/2*rho*(Uinf**2-vel_t[:,-1]**2),(numPanels,1))
    lift_bern=np.sum(dP*si*-np.cos(theta))
    
    return lift,lift_flat_plate_j,lift_bern,vel_t


import numpy as np
Uinf=3
AoA=5*np.pi/180
rho=1.204

(panel_coor,col_coor,numPanels)=Airfoil_Setup()
(x,b,A,B,infCoef,U,theta,r,beta)=Compute_Coeff(panel_coor,col_coor,numPanels,Uinf,AoA)
=Comp_Lift(numPanels,panel_coor,x,Uinf,chord,U,AoA,rho):
