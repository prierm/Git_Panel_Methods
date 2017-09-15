# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:11:56 2017

@author: prierm
"""

import numpy as np
import scipy as sc
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
numPanels=32
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
numFrames=1
tEnd=.01
tInterval=np.linspace(0,tEnd,numFrames+1)#!!!!!!!!!!!!
tStep=tEnd/numFrames
f=2
omega=2*np.pi*f
h0=.0375
theta0=60*np.pi/180
h_t=h0*np.cos(omega*tInterval)
h_t_dot=-h0*omega*np.sin(omega*tInterval)
theta_t=theta0*-np.sin(omega*tInterval)
theta_t_dot=theta0*omega*-np.cos(omega*tInterval)

# thetas for each panel relative to foil frame
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

#   matrix A, source inf coef normal direction
A=copy.copy(nSource)

#   airfoil perimeter
si=np.zeros((numPanels,1))
perimeter=0
for i in range(numPanels):
        si[i]=((xp[i+1]-xp[i])**2+(yp[i+1]-yp[i])**2)**(1/2)
        perimeter=perimeter+si[i] 

# data storage
xpSto=np.zeros((numPanels+1,numFrames))
ypSto=np.zeros((numPanels+1,numFrames))
tangVelSto=np.zeros((numPanels,numFrames))
normVelSto=np.zeros((numPanels,numFrames))

# free vorticies
xVorPos=np.zeros((numFrames,1))
yVorPos=np.zeros((numFrames,1))
vortStrength=np.zeros((numFrames,1))

# previous bound vorticity
prevGamma=0
currGamma=0

# initial wake panel parameters
wakePanelTheta=0
wakePanelLen=.01
wakePanelStr=0
xpWake1=xp[0]
xpWake2=xpWake1+wakePanelLen
ypWake1=yp[0]
ypWake2=yp[0]
xcWake=(xpWake1+xpWake2)/2
ycWake=(ypWake1+ypWake2)/2

# test storage
wakeThetaSto=np.zeros((20,numFrames))
wakeStrSto=np.zeros((20,numFrames))
wakeLenSto=np.zeros((20,numFrames))

for t in range(numFrames):   # for each time frame
    # freestream in terms of normal tangential coordinates
    normU=(-np.sin(theta)*np.cos(theta_t[t])+np.cos(theta)*np.sin(theta_t[t]))*Uinf#!!!!!!!!!!!!
    tangU=(np.cos(theta)*np.cos(theta_t[t])+np.sin(theta_t[t])*np.sin(theta))*Uinf
    
    #foil contribution
    foilContNorm=-h_t_dot[t]*np.sin(theta_t[t])*np.sin(theta)+(-h_t_dot[t]*np.cos(theta_t[t])-theta_t_dot[t]*-xc)*np.cos(theta)
    foilContTang=h_t_dot[t]*np.sin(theta_t[t])*np.cos(theta)+(-h_t_dot[t]*np.cos(theta_t[t])-theta_t_dot[t]*-xc)*np.sin(theta)
    
    # source foil panels on free vortices
    if t>0:
        xTemp=np.zeros((t,numPanels))
        yTemp=np.zeros((t,numPanels))
        for i in range(t):
            for j in range(numPanels):
                r1=((xVorPos[i]-xp[j])**2+(yVorPos[i]-yp[j])**2)**(1/2)
                r2=((xVorPos[i]-xp[j+1])**2+(yVorPos[i]-yp[j+1])**2)**(1/2)
                nu2=np.arctan2(-(xVorPos[i]-xp[j])*np.sin(theta[j])+(yVorPos[i]-yp[j])*np.cos(theta[j])+(xp[j+1]-xp[j])*np.sin(theta[j])-(yp[j+1]-yp[j])*np.cos(theta[j]),\
                             (xVorPos[i]-xp[j])*np.cos(theta[j])+(yVorPos[i]-yp[j])*np.sin(theta[j])-(xp[j+1]-xp[j])*np.cos(theta[j])-(yp[j+1]-yp[j])*np.sin(theta[j]))
                nu1=np.arctan2(-(xVorPos[i]-xp[j])*np.sin(theta[j])+(yVorPos[i]-yp[j])*np.cos(theta[j]),(xVorPos[i]-xp[j])*np.cos(theta[j])+(yVorPos[i]-yp[j])*np.sin(theta[j]))
                yTemp[i,j]=1/(2*np.pi)*(nu2-nu1)
                xTemp[i,j]=1/(4*np.pi)*np.log(r1**2/r2**2)
        ySPV=-np.sin(-np.reshape(theta,(1,numPanels)))*xTemp+np.cos(-np.reshape(theta,(1,numPanels)))*yTemp
        xSPV=np.cos(-np.reshape(theta,(1,numPanels)))*xTemp+np.sin(-np.reshape(theta,(1,numPanels)))*yTemp # foil coordinate system
        # vortex foil panels on free vorticies
        xVPV=copy.copy(ySPV)
        yVPV=-xSPV # GOOD
    
    # free vortices on collocation points
    if t>0:
        tVVP=np.zeros((numPanels,t))
        nVVP=np.zeros((numPanels,t))
        for i in range(numPanels):
            for j in range(t):
                theta_im=np.arctan2(yc[i]-yVorPos[j],xc[i]-xVorPos[j])
                r1=((xVorPos[j]-xc[i])**2+(yVorPos[j]-yc[i])**2)**(1/2)
                tVVP[i,j]=-np.sin(theta[i]-theta_im)/(2*np.pi*r1)
                nVVP[i,j]=-np.cos(theta[i]-theta_im)/(2*np.pi*r1)  # GOOD 
        
    # free vorticies on free vorticies
    if t>0:
        xVV=np.zeros((t,t))
        yVV=np.zeros((t,t))
        for i in range(t):
            for j in range(t):
                if i==j:
                    xVV[i,j]=yVV[i,j]=0
                else:
                    theta_im=np.arctan2(yVorPos[i]-yVorPos[j],xVorPos[i]-xVorPos[j])
                    r1=((xVorPos[i]-xVorPos[j])**2+(yVorPos[i]-yVorPos[j])**2)**(1/2)
                    xVV[i,j]=-np.sin(-theta_im)/(2*np.pi*r1)
                    yVV[i,j]=-np.cos(-theta_im)/(2*np.pi*r1)    # GOOD
    
    # previous gamma
    prevGamma=copy.copy(currGamma)
    # iterate until wake panel theta , wake panel length, and wake panel strength converge
    k=5
    prevThetaW=wakePanelTheta
    prevPanelLen=wakePanelLen
    while k<1:
        wakeThetaSto[k,t]=wakePanelTheta    # test code
        wakeLenSto[k,t]=wakePanelLen
        wakeStrSto[k,t]=wakePanelStr
        
        # free vorticies on wake panel
        if t>0:
            tVVW=np.zeros((1,t))
            nVVW=np.zeros((1,t))
            for i in range(t):
                theta_im=np.arctan2(ycWake-yVorPos[i],xcWake-xVorPos[i])
                r1=((xVorPos[i]-xcWake)**2+(yVorPos[i]-ycWake)**2)**(1/2)
                tVVW[0,i]=-np.sin(wakePanelTheta-theta_im)/(2*np.pi*r1)
                nVVW[0,i]=-np.cos(wakePanelTheta-theta_im)/(2*np.pi*r1) # GOOD
        
        # wake panel on collocation points
        xTemp=np.zeros((numPanels,1))
        yTemp=np.zeros((numPanels,1))
        for i in range(numPanels):
            r1=((xc[i]-xpWake1)**2+(yc[i]-ypWake1)**2)**(1/2)
            r2=((xc[i]-xpWake2)**2+(yc[i]-ypWake2)**2)**(1/2)
            nu2=np.arctan2(-(xc[i]-xpWake1)*np.sin(wakePanelTheta)+(yc[i]-ypWake1)*np.cos(wakePanelTheta)+(xpWake2-xpWake1)*np.sin(wakePanelTheta)-(ypWake2-ypWake1)*np.cos(wakePanelTheta),\
                             (xc[i]-xpWake1)*np.cos(wakePanelTheta)+(yc[i]-ypWake1)*np.sin(wakePanelTheta)-(xpWake2-xpWake1)*np.cos(wakePanelTheta)-(ypWake2-ypWake1)*np.sin(wakePanelTheta))
            nu1=np.arctan2(-(xc[i]-xpWake1)*np.sin(wakePanelTheta)+(yc[i]-ypWake1)*np.cos(wakePanelTheta),(xc[i]-xpWake1)*np.cos(wakePanelTheta)+(yc[i]-ypWake1)*np.sin(wakePanelTheta))
            xTemp[i]=1/(2*np.pi)*(nu2-nu1)
            yTemp[i]=-1/(4*np.pi)*np.log(r1**2/r2**2) # GOOD
            
        nWP=-np.sin(theta-wakePanelTheta)*xTemp+np.cos(theta-wakePanelTheta)*yTemp
        tWP=np.cos(theta-wakePanelTheta)*xTemp+np.sin(theta-wakePanelTheta)*yTemp   # GOOD
            
        # vortex foil panels on wake panel
        xTemp=np.zeros((1,numPanels))
        yTemp=np.zeros((1,numPanels))
        for j in range(numPanels):
            r1=((xcWake-xp[j])**2+(ycWake-yp[j])**2)**(1/2)
            r2=((xcWake-xp[j+1])**2+(ycWake-yp[j+1])**2)**(1/2)
            nu2=np.arctan2(-(xcWake-xp[j])*np.sin(theta[j])+(ycWake-yp[j])*np.cos(theta[j])+(xp[j+1]-xp[j])*np.sin(theta[j])-(yp[j+1]-yp[j])*np.cos(theta[j]),\
                             (xcWake-xp[j])*np.cos(theta[j])+(ycWake-yp[j])*np.sin(theta[j])-(xp[j+1]-xp[j])*np.cos(theta[j])-(yp[j+1]-yp[j])*np.sin(theta[j]))
            nu1=np.arctan2(-(xcWake-xp[j])*np.sin(theta[j])+(ycWake-yp[j])*np.cos(theta[j]),(xcWake-xp[j])*np.cos(theta[j])+(ycWake-yp[j])*np.sin(theta[j]))
            xTemp[0,j]=1/(2*np.pi)*(nu2-nu1)
            yTemp[0,j]=-1/(4*np.pi)*np.log(r1**2/r2**2)
        nVPW=-xTemp*np.sin(wakePanelTheta-np.reshape(theta,(1,numPanels)))+yTemp*np.cos(wakePanelTheta-np.reshape(theta,(1,numPanels)))
        tVPW=xTemp*np.cos(wakePanelTheta-np.reshape(theta,(1,numPanels)))+yTemp*np.sin(wakePanelTheta-np.reshape(theta,(1,numPanels)))   
        
        # source foil panels on wake panel
        nSPW=copy.copy(tVPW)
        tSPW=-nVPW # THIS LOOKS GOOD
        
        # wake panel on free vorticies
        if t>0:
            xTemp=np.zeros((t,1))
            yTemp=np.zeros((t,1))
            for i in range(t):
                r1=((xVorPos[i]-xpWake1)**2+(yVorPos[i]-ypWake1)**2)**(1/2)
                r2=((xVorPos[i]-xpWake2)**2+(yVorPos[i]-ypWake2)**2)**(1/2)
                nu2=np.arctan2(-(xVorPos[i]-xpWake1)*np.sin(wakePanelTheta)+(yVorPos[i]-ypWake1)*np.cos(wakePanelTheta)+(xpWake2-xpWake1)*np.sin(wakePanelTheta)-(ypWake2-ypWake1)*np.cos(wakePanelTheta),\
                                 (xVorPos[i]-xpWake1)*np.cos(wakePanelTheta)+(yVorPos[i]-ypWake1)*np.sin(wakePanelTheta)-(xpWake2-xpWake1)*np.cos(wakePanelTheta)-(ypWake2-ypWake1)*np.sin(wakePanelTheta))
                nu1=np.arctan2(-(xVorPos[i]-xpWake1)*np.sin(wakePanelTheta)+(yVorPos[i]-ypWake1)*np.cos(wakePanelTheta),(xVorPos[i]-xpWake1)*np.cos(wakePanelTheta)+(yVorPos[i]-ypWake1)*np.sin(wakePanelTheta))
                xTemp[i]=1/(2*np.pi)*(nu2-nu1)
                yTemp[i]=-1/(4*np.pi)*np.log(r1**2/r2**2) # THIS LOOKS GOOD
                
                yWV=-np.sin(-wakePanelTheta)*xTemp+np.cos(-wakePanelTheta)*yTemp
                xWV=np.cos(-wakePanelTheta)*xTemp+np.sin(-wakePanelTheta)*yTemp
        
        # kutta condition, find bound circulation
        currGamma=prevGamma+tStep*(tangVelSto[0,t]**2-tangVelSto[-1,t]**2)/(2*perimeter) #MAYBE CHANGE THIS...
        
         # vector b
        b=perimeter/wakePanelLen*nWP-np.reshape(np.sum(nVortex,axis=1),(numPanels,1))
        
        # vector c
        if t>0:
            c=-normU-foilContNorm-perimeter/wakePanelLen*prevGamma*nWP-np.dot(nVVP,np.reshape(vortStrength[:t],(t,1)))
        else:
            c=-normU-foilContNorm-perimeter/wakePanelLen*prevGamma*nWP
        
        # solve for source strengths
        x=sc.linalg.solve(A,currGamma*b+c) 
        
        #confirm that velocities on airfoil are tangential
        if t>0:
            normVelFoil=np.dot(nSource,np.reshape(x,(numPanels,1)))+np.reshape(currGamma*np.sum(nVortex,axis=1),(numPanels,1))+nWP*wakePanelStr+np.reshape(np.dot(nVVP,vortStrength[:t]),(numPanels,1))+normU+foilContNorm
            tangVelFoil=np.dot(tSource,np.reshape(x,(numPanels,1)))+np.reshape(currGamma*np.sum(tVortex,axis=1),(numPanels,1))+tWP*wakePanelStr+np.reshape(np.dot(tVVP,vortStrength[:t]),(numPanels,1))+tangU+foilContTang
        else:
            normVelFoil=np.dot(nSource,np.reshape(x,(numPanels,1)))+np.reshape(currGamma*np.sum(nVortex,axis=1),(numPanels,1))+nWP*wakePanelStr+normU+foilContNorm
            tangVelFoil=np.dot(tSource,np.reshape(x,(numPanels,1)))+np.reshape(currGamma*np.sum(tVortex,axis=1),(numPanels,1))+tWP*wakePanelStr+tangU+foilContTang
        tangVelSto[:,t]=np.reshape(tangVelFoil,(numPanels,))
        normVelSto[:,t]=np.reshape(normVelFoil,(numPanels,))
        
        # calculate velocity at wake panel
        if t>0:
            tangInf=np.sum(tVPW)*currGamma+np.dot(tSPW,x)+np.dot(tVVW,vortStrength[:t])
            normInf=np.sum(nVPW)*currGamma+np.dot(nSPW,x)+np.dot(nVVW,vortStrength[:t])
        else:
            tangInf=np.sum(tVPW)*currGamma+np.dot(tSPW,x)
            normInf=np.sum(nVPW)*currGamma+np.dot(nSPW,x)
        uWake=tangInf*np.cos(wakePanelTheta)-normInf*np.sin(wakePanelTheta)+Uinf
        vWake=tangInf*np.sin(wakePanelTheta)+normInf*np.cos(wakePanelTheta) 
        
        # update panel parameters
        wakePanelTheta=np.arctan2(vWake,uWake)
        wakePanelLen=tStep*(uWake**2+vWake**2)**(1/2)
        wakePanelStr=perimeter/wakePanelLen*(prevGamma-currGamma)  # KELVIN THRM
        xpWake1=xpWake1
        ypWake1=ypWake1
        xpWake2=xpWake1+wakePanelLen*np.cos(wakePanelTheta)
        ypWake2=ypWake1+wakePanelLen*np.sin(wakePanelTheta)
        xcWake=(xpWake1+xpWake2)/2
        ycWake=(ypWake1+ypWake2)/2
        
        k=k+1
        
    # lift from bernoulli
    
    # moment from bernoulli
    
    # coefficient of pressure over chord
    
    # detach and convect wake panel
    xVorPos[t]=xcWake+uWake*tStep
    yVorPos[t]=ycWake+vWake*tStep
    vortStrength[t]=wakePanelLen*wakePanelStr   # THIS LOOKS GOOD
    
    # convect free vorticies
    if t>0:
        xVorPos[:t]=np.reshape(np.dot(xSPV,x),(t,1))+np.reshape(np.sum(xVPV*currGamma,axis=1),(t,1))+xWV*wakePanelStr+np.dot(xVV,vortStrength[:t])+Uinf
        yVorPos[:t]=np.reshape(np.dot(ySPV,x),(t,1))+np.reshape(np.sum(yVPV*currGamma,axis=1),(t,1))+yWV*wakePanelStr+np.dot(yVV,vortStrength[:t])
        
    # transform free vorticies to next time frame foil coordinate system
    xVorPosOld=copy.copy(xVorPos)
    yVorPosOld=copy.copy(yVorPos)
    xVorPos=xVorPosOld*np.cos(theta_t[t+1]-theta_t[t])-yVorPosOld*np.sin(theta_t[t+1]-theta_t[t])+h_t[t]*np.sin(theta_t[t+1])
    yVorPos=xVorPosOld*np.sin(theta_t[t+1]-theta_t[t])+yVorPosOld*np.cos(theta_t[t+1]-theta_t[t])-h_t[t]*np.cos(theta_t[t+1])
    
    
        
    
    
    
    
    
    