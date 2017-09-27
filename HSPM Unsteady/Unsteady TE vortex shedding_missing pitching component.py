# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:11:56 2017

@author: prierm
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

def SourceDistOnPt(xc,yc,xp,yp,theta):
    ups=np.zeros((len(xc),len(xp)-1))
    wps=np.zeros((len(xc),len(xp)-1)) 
    for i in range(len(xc)):
        for j in range(len(xp)-1):
            r1=((xc[i]-xp[j])**2+(yc[i]-yp[j])**2)**(1/2)
            r2=((xc[i]-xp[j+1])**2+(yc[i]-yp[j+1])**2)**(1/2)
            nu2=np.arctan2(-(xc[i]-xp[j])*np.sin(theta[j])+(yc[i]-yp[j])*np.cos(theta[j])+(xp[j+1]-xp[j])*np.sin(theta[j])-(yp[j+1]-yp[j])*np.cos(theta[j]),\
                             (xc[i]-xp[j])*np.cos(theta[j])+(yc[i]-yp[j])*np.sin(theta[j])-(xp[j+1]-xp[j])*np.cos(theta[j])-(yp[j+1]-yp[j])*np.sin(theta[j]))
            nu1=np.arctan2(-(xc[i]-xp[j])*np.sin(theta[j])+(yc[i]-yp[j])*np.cos(theta[j]),(xc[i]-xp[j])*np.cos(theta[j])+(yc[i]-yp[j])*np.sin(theta[j]))
            ups[i,j]=1/(4*np.pi)*np.log(r1**2/r2**2)
            wps[i,j]=1/(2*np.pi)*(nu2-nu1)
    return ups,wps

def VortexDistOnPt(xc,yc,xp,yp,theta):
    ups,wps=SourceDistOnPt(xc,yc,xp,yp,theta)
    upv=wps
    wpv=-ups
    return upv,wpv

def VortexPointOnPt(xc,yc,xVorPos,yVorPos,theta):
    theta_im=np.arctan2(yc-yVorPos,xc-xVorPos)
    r1=((xVorPos-xc)**2+(yVorPos-yc)**2)**(1/2)
    tVP=-np.sin(theta-theta_im)/(2*np.pi*r1)
    nVP=-np.cos(theta-theta_im)/(2*np.pi*r1)
    return tVP,nVP

def PanelCoorToNTCoor(up,wp,theta_i,theta_j):
    theta_i=np.reshape(theta_i,(len(theta_i),1))
    theta_j=np.reshape(theta_j,(len(theta_j),1))
    normal=-np.sin(theta_i-np.transpose(theta_j))*(up)+np.cos(theta_i-np.transpose(theta_j))*(wp)
    tangent=np.cos(theta_i-np.transpose(theta_j))*(up)+np.sin(theta_i-np.transpose(theta_j))*(wp)
    return tangent,normal

    
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
numFrames=180
tEnd=1.5
tInterval=np.linspace(0,tEnd,numFrames+1)
tStep=tEnd/numFrames
f=2
omega=2*np.pi*f
h0=.0375
theta0=60*np.pi/180
h_t=h0*np.cos(omega*tInterval)
h_t_dot=-h0*omega*np.sin(omega*tInterval)
theta_t=theta0*-np.sin(omega*tInterval)
theta_t_dot=theta0*omega*-np.cos(omega*tInterval)

# panel coordinates in terms of foil x and y
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

# thetas for each panel relative to foil frame
theta=np.zeros((numPanels,1))
for i in range(numPanels):
    theta[i]=np.arctan2(yp[i+1]-yp[i],xp[i+1]-xp[i])
    
# influence coefficients of foil on foil
ups,wps=SourceDistOnPt(xc,yc,xp,yp,theta)
np.fill_diagonal(ups,0)
np.fill_diagonal(wps,.5)
upv=copy.copy(wps)
wpv=-ups
tSource,nSource=PanelCoorToNTCoor(ups,wps,theta,theta)
tVortex,nVortex=PanelCoorToNTCoor(upv,wpv,theta,theta)

#   matrix A, source inf coef normal direction
A=np.zeros((numPanels+1,numPanels+1))
A[:numPanels,:numPanels]=nSource
A[:numPanels,-1]=np.sum(nVortex,axis=1)
A[-1,:numPanels]=tSource[0,:]+tSource[-1,:]
A[-1,-1]=np.sum(tVortex[0,:])+np.sum(tVortex[-1,:])

# lu decomp and inversion
Ainv=np.linalg.inv(A)

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
xVorPos=np.zeros((numFrames,1))
yVorPos=np.zeros((numFrames,1))
vortStrength=np.zeros((numFrames,1))
dPSto1=np.zeros((numPanels,2*numFrames//3//5))
dPSto2=np.zeros((numPanels,2*numFrames//3//5))
CySto=np.zeros((2*numFrames//3//5,1))
momentSto=np.zeros((2*numFrames//3//5,1))
CpSto=np.zeros((numPanels,2*numFrames//3//5))
timeSto=np.zeros(((numFrames-(numFrames*1//3))//5,1))

# initial wake panel parameters
wakePanelStr=0
xWake=0
yWake=0

# bound circulation
oldGammaB=0

# storage to check convergence
iterations=5
wakeStrSto=np.zeros((iterations,numFrames))
wakeThetaSto=np.zeros((numFrames,1))
wakeLenSto=np.zeros((numFrames,1))
n=0
for t in range(numFrames):   # for each time frame
    # freestream in terms of normal tangential coordinates
    normU=(-np.sin(theta)*np.cos(theta_t[t])+np.cos(theta)*np.sin(theta_t[t]))*Uinf
    tangU=(np.cos(theta)*np.cos(theta_t[t])+np.sin(theta_t[t])*np.sin(theta))*Uinf
    
    #foil contribution
    foilContNorm=-h_t_dot[t]*np.sin(theta_t[t])*np.sin(theta)+(-h_t_dot[t]*np.cos(theta_t[t])-theta_t_dot[t]*-xc)*np.cos(theta)
    foilContTang=h_t_dot[t]*np.sin(theta_t[t])*np.cos(theta)+(-h_t_dot[t]*np.cos(theta_t[t])-theta_t_dot[t]*-xc)*np.sin(theta)
    
    # source foil panels on free vortices
    if t>0:
        xTemp,yTemp=SourceDistOnPt(xVorPos[:t],yVorPos[:t],xp,yp,theta)
        xSPV,ySPV=PanelCoorToNTCoor(xTemp,yTemp,np.zeros((t,1)),theta)
        
        # vortex foil panels on free vorticies
        xVPV=copy.copy(ySPV)
        yVPV=-xSPV
    
    # free vortices on collocation points
    if t>0:
        tVVP=np.zeros((numPanels,t))
        nVVP=np.zeros((numPanels,t))
        for i in range(numPanels):
            for j in range(t):
                tVVP[i,j],nVVP[i,j]=VortexPointOnPt(xc[i],yc[i],xVorPos[j],yVorPos[j],theta[i])
        
    # free vorticies on free vorticies
    if t>0:
        xVV=np.zeros((t,t))
        yVV=np.zeros((t,t))
        for i in range(t):
            for j in range(t):
                if i==j:
                    xVV[i,j]=yVV[i,j]=0
                else:
                    xVV[i,j],yVV[i,j]=VortexPointOnPt(xVorPos[i],yVorPos[i],xVorPos[j],yVorPos[j],0)
    
    # previous circulation sum
    prevCircSum=0
    currCircSum=1
    
    # iterate until wake panel theta , wake panel length, and wake panel strength converge
    wakePanelLen=tStep*((Uinf*np.sin(theta_t[t])+foilContNorm[-1]*np.cos(theta_t[t])+foilContTang[-1]*np.sin(theta_t[t]))**2\
                        +(Uinf*np.cos(theta_t[t])+foilContNorm[-1]*-np.sin(theta_t[t])+foilContTang[-1]*np.cos(theta_t[t]))**2)**(1/2)*.25
    wakePanelTheta=np.arctan2(Uinf*np.sin(theta_t[t])+foilContNorm[-1]*np.cos(theta_t[t])+foilContTang[-1]*np.sin(theta_t[t]),\
                              Uinf*np.cos(theta_t[t])+foilContNorm[-1]*-np.sin(theta_t[t])+foilContTang[-1]*np.cos(theta_t[t]))
    wakePanelStr=.1
    xpWake1=copy.copy(xp[-1])
    ypWake1=copy.copy(yp[-1])
    xcWake=xpWake1+wakePanelLen*np.cos(theta_t[t])
    ycWake=ypWake1+wakePanelLen*np.sin(theta_t[t])
    k=0
    while k<iterations and abs(currCircSum)>10**(-14):
        wakeThetaSto[t,0]=wakePanelTheta
        wakeLenSto[t,0]=wakePanelLen
        wakeStrSto[k,t]=wakePanelStr
        
        # update circulation sum for next iteration
        if k>0:
            prevCircSum=copy.copy(currCircSum)
        
        # free vorticies on wake panel
        if t>0:
            xVVW=np.zeros((1,t))
            yVVW=np.zeros((1,t))
            for j in range(t):
                xVVW[0,j],yVVW[0,j]=VortexPointOnPt(xcWake,ycWake,xVorPos[j],yVorPos[j],0)
        
        # wake panel on collocation points
        tWP=np.zeros((numPanels,1))
        nWP=np.zeros((numPanels,1))
        for i in range(numPanels):
            tWP[i],nWP[i]=VortexPointOnPt(xc[i],yc[i],xcWake,ycWake,theta[i])
            
        # vortex foil panels on wake panel
        tTemp,nTemp=VortexDistOnPt(xcWake,ycWake,xp,yp,theta)
        xVPW,yVPW=PanelCoorToNTCoor(tTemp,nTemp,np.zeros((1,1)),theta)
        
        # source foil panels on wake panel
        ySPW=copy.copy(xVPW)
        xSPW=-yVPW
        
        # wake panel on free vorticies
        if t>0:
            xWV=np.zeros((t,1))
            yWV=np.zeros((t,1))
            for i in range(t):
                xWV[i],yWV[i]=VortexPointOnPt(xVorPos[t],yVorPos[t],xcWake,ycWake,0)
        
         # vector b
        b=np.zeros((numPanels+1,1))
        if t==0:
            b[:-1]=-(normU+foilContNorm+wakePanelStr*nWP)
            b[-1]=-(tangU[0]+tangU[-1])-(foilContTang[0]+foilContTang[-1])-(tWP[0]+tWP[-1])*wakePanelStr
        else:
            b[:-1]=-(normU+foilContNorm+wakePanelStr*nWP+np.dot(nVVP,np.reshape(vortStrength[:t],(t,1))))
            b[-1]=-(tangU[0]+tangU[-1])-(foilContTang[0]+foilContTang[-1])-(tWP[0]+tWP[-1])*wakePanelStr-np.dot(tVVP[0,:]+tVVP[-1,:],np.reshape(vortStrength[:t],(t,1)))
        
        # solve
        x=np.dot(Ainv,b)
        
        
        # kelvin circulation thrm
        if t==0:
            currCircSum=x[-1]*perimeter+wakePanelStr
        else:
            currCircSum=x[-1]*perimeter+wakePanelStr+np.sum(vortStrength[:t])
        
        # calculate wake panel strength
        if k==0:
            if t==0:
                wakePanelStr=-x[-1]*perimeter
            else:
                wakePanelStr=-x[-1]*perimeter-np.sum(vortStrength[:t])
        elif abs(currCircSum)>10**(-14):
            wakePanelStr=wakePanelStr-currCircSum/((currCircSum-prevCircSum)/(wakeStrSto[k,t]-wakeStrSto[k-1,t]))
        
        #confirm that velocities on airfoil are tangential
        if t>0:
            normVelFoil=np.dot(nSource,np.reshape(x[:-1],(numPanels,1)))+np.reshape(x[-1]*np.sum(nVortex,axis=1),(numPanels,1))+nWP*wakePanelStr+np.reshape(np.dot(nVVP,vortStrength[:t]),(numPanels,1))+normU+foilContNorm
            tangVelFoil=np.dot(tSource,np.reshape(x[:-1],(numPanels,1)))+np.reshape(x[-1]*np.sum(tVortex,axis=1),(numPanels,1))+tWP*wakePanelStr+np.reshape(np.dot(tVVP,vortStrength[:t]),(numPanels,1))+tangU+foilContTang
        else:
            normVelFoil=np.dot(nSource,np.reshape(x[:-1],(numPanels,1)))+np.reshape(x[-1]*np.sum(nVortex,axis=1),(numPanels,1))+nWP*wakePanelStr+normU+foilContNorm
            tangVelFoil=np.dot(tSource,np.reshape(x[:-1],(numPanels,1)))+np.reshape(x[-1]*np.sum(tVortex,axis=1),(numPanels,1))+tWP*wakePanelStr+tangU+foilContTang
        tangVelSto[:,t]=np.reshape(tangVelFoil,(numPanels,))
        normVelSto[:,t]=np.reshape(normVelFoil,(numPanels,))
        
        # calculate velocity at wake panel, include heaving motion
        if t>0:
            uWake=np.sum(xVPW)*x[-1]+np.dot(xSPW,x[:-1])+np.dot(xVVW,vortStrength[:t])+Uinf*np.cos(theta_t[t])+h_t_dot[t]*np.sin(theta_t[t])
            vWake=np.sum(yVPW)*x[-1]+np.dot(ySPW,x[:-1])+np.dot(yVVW,vortStrength[:t])+Uinf*np.sin(theta_t[t])-h_t_dot[t]*np.cos(theta_t[t])
        else:
            uWake=np.sum(xVPW)*x[-1]+np.dot(xSPW,x[:-1])+Uinf*np.cos(theta_t[t])+h_t_dot[t]*np.sin(theta_t[t])
            vWake=np.sum(yVPW)*x[-1]+np.dot(ySPW,x[:-1])+Uinf*np.sin(theta_t[t])-h_t_dot[t]*np.cos(theta_t[t])
        
        k=k+1
     
    #Lift from Bernoulli
    currGammaB=x[-1]
    if t>numFrames*1//3-1 and t%5==0:
        timeSto[n]=tInterval[t]
        dPSto1[:,n]=1/2*rho*(np.reshape((Uinf**2-tangVelFoil**2),(numPanels,))-2*np.reshape(si,(numPanels,))*(tangVelSto[:,t]-tangVelSto[:,t-1])/tStep)
        dPSto2[:,n]=1/2*rho*(np.reshape((Uinf**2-tangVelFoil**2),(numPanels,))-(x[-1]-oldGammaB)*np.reshape(si,(numPanels,))/tStep)
        forceDist=-dPSto1[:,n]*np.reshape(si,(numPanels,))
        Fy_bern=np.sum(np.reshape(forceDist,(numPanels,1))*(np.sin(theta_t[t])*np.sin(theta)+np.cos(theta_t[t])*np.cos(theta)))
        CySto[n]=2*Fy_bern/(rho*Uinf**2*chord)
    
        # moment from bernoulli
        momentSto[n]=np.sum(forceDist*np.cos(theta)*-xc)
        
        # coefficient of pressure over chord
        CpSto[:,n]=dPSto1[:,n]/(1/2*rho*Uinf**2)
        
        title='t/T = ' + '{:.3f}'.format((tInterval[t]-1)/.5) + r'   $\theta_p$ = ' + '{:.3f}'.format(theta_t[t]*180/np.pi) + r'$^\circ$' + r'   $C_Y = $' + '{:.3f}'.format(float(CySto[n]))
        fig,(ax0,ax1)=plt.subplots(2,1)
        fig.set_size_inches(16,14)
        ax1.plot(xc[:numPanels//2],CpSto[:numPanels//2,n],'*')
        ax1.plot(xc[numPanels//2:],CpSto[numPanels//2:,n],'o')
        plt.legend(['bottom surface','top surface'],loc='lower right',fontsize=16)
        plt.xlabel('chord',fontsize=20)
        plt.ylabel(r'$C_P$',fontsize=20)
        ax1.set_title(title,fontsize=24)   
    
        # transform everything to global X Y for plotting
        xpGlobal=xp*np.cos(theta_t[t])+yp*np.sin(theta_t[t])
        ypGlobal=-xp*np.sin(theta_t[t])+yp*np.cos(theta_t[t])+h_t[t]
        xcWakeGlobal=xcWake*np.cos(theta_t[t])+ycWake*np.sin(theta_t[t])
        ycWakeGlobal=-xcWake*np.sin(theta_t[t])+ycWake*np.cos(theta_t[t])+h_t[t]
        xVorPosGlobal=xVorPos*np.cos(theta_t[t])+yVorPos*np.sin(theta_t[t])
        yVorPosGlobal=-xVorPos*np.sin(theta_t[t])+yVorPos*np.cos(theta_t[t])+h_t[t]
        
        # plot foil and vortex positions in foil frame
        ax0.set_xlim(-.1,1.5)
        ax0.set_ylim(-.8,.8)
        ax0.plot(xpGlobal,ypGlobal)
        ax0.plot(xcWakeGlobal,ycWakeGlobal,'o')
        ax0.plot(xVorPosGlobal[:t],yVorPosGlobal[:t],'*')
        fig.savefig(pathName + '//frame' + str(n) + '.png')
        plt.close()
        n=n+1
        
    # detach and convect wake panel in current foil frame, heaving included
    xVorPos[t]=xcWake+uWake*tStep
    yVorPos[t]=ycWake+vWake*tStep
    vortStrength[t]=wakePanelStr
    
    # convect free vorticies in current foil frame, include heaving
    if t>0:
        xVorPos[:t]=xVorPos[:t]+(np.reshape(np.dot(xSPV,x[:-1]),(t,1))+np.reshape(np.sum(xVPV*x[-1],axis=1),(t,1))+xWV*wakePanelStr+np.dot(xVV,vortStrength[:t])+Uinf*np.cos(theta_t[t]))*tStep+\
        (h_t[t+1]-h_t[t])*np.sin(theta_t[t])
        yVorPos[:t]=yVorPos[:t]+(np.reshape(np.dot(ySPV,x[:-1]),(t,1))+np.reshape(np.sum(yVPV*x[-1],axis=1),(t,1))+yWV*wakePanelStr+np.dot(yVV,vortStrength[:t])+Uinf*np.sin(theta_t[t]))*tStep-\
        (h_t[t+1]-h_t[t])*np.cos(theta_t[t])
        
    # transform free vorticies to next time frame foil coordinate system, heaving included prior
    xVorPosOld=copy.copy(xVorPos)
    yVorPosOld=copy.copy(yVorPos)
    xVorPos=xVorPosOld*np.cos(theta_t[t+1]-theta_t[t])-yVorPosOld*np.sin(theta_t[t+1]-theta_t[t])     #UNSURE ABOUT h_t...
    yVorPos=xVorPosOld*np.sin(theta_t[t+1]-theta_t[t])+yVorPosOld*np.cos(theta_t[t+1]-theta_t[t])
    
    oldGammaB=x[-1]
  
fig2,ax2=plt.subplots(1,1)
fig2.set_size_inches(12,8)
ax2.plot((timeSto-.5)/(.5),CySto)
plt.xlabel('t/T',fontsize=20)
plt.ylabel(r'$C_Y$',fontsize=20)
fig2.savefig(pathName + '//force_history' + '.png')
plt.close()
    
    

