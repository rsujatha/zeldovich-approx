#~ from __future__ import division
import numpy as np
import math
import pylab
import matplotlib.pyplot as plt
import sys
import math
import library as lb
import mathtools as mt


Xsize = 64
GridSize=64
dx = Xsize/float(GridSize)
dk=2*np.pi/(dx*GridSize)

kspace =np.concatenate([range(0,GridSize/2),range(-GridSize/2,0)])*dk
xspace = np.array(range(0,GridSize))*dx
XSize=GridSize*dx
ScaleFactor0=0.0001
H_0=70.
mean=0
Omega_matter=0.3

def H(a):
	Omega_matter=0.3
	Omega_lambda=0.7
	H_0=70.
	H=np.ones(np.size(a))
	H=H_0*(Omega_matter*(a**(-3.))+Omega_lambda)**(1/2.)
	return H



def P(k):
	kc=0.2
	P=10*np.exp(-k/kc)*(k+1e-15)
	return P

def P_single(k):
	tiny=1e-15
	k_fundamental=0.2
	P=np.zeros(np.size(k))
	P=P+tiny
	index=np.where((k<0.2+tiny) & (k>0.2-tiny))[0]
	P[index]=2
	return P
	

#~ variance = P_single(np.abs(kspace))
variance = P(np.abs(kspace))
#~ print variance
#~ variance_single= P(np.abs(kspace))/2.
#~ print variance_single
###############                                                                                  ###############################
############### Initialising delta k as random gaussian variable taking care to make its IFFT real###############################
###############                                                                                  ###############################
deltak = np.random.normal(mean, variance , GridSize) + np.random.normal(mean, variance , GridSize)*1j
deltak[GridSize/2+1:] = deltak[1:GridSize/2][::-1].conjugate()
deltak[0] = 0
deltak[GridSize/2] = np.real(deltak[GridSize/2])
deltax = GridSize *np.fft.ifft(deltak)


psik = -(1j)*deltak/(kspace+1e-15)
index = np.where(kspace==0)
psik[index] = 0
psix0 = GridSize*np.fft.ifft(psik)
N_a=100
Position = np.zeros([N_a,GridSize])
Position_PERIODIC = np.zeros([N_a,GridSize])

i=0
a_value = np.linspace(0.00000001,0.0001,N_a)
da = a_value[2]-a_value[1]

for ScaleFactor in a_value:	
	
	psix = lb.GrowthFunctionAnalytic(ScaleFactor)/lb.GrowthFunctionAnalytic(ScaleFactor0)*psix0.real
	#~ Position[i] = xspace - psix
	#~ print Position
	Position[i] = xspace- psix
	Position_PERIODIC[i] = np.mod(xspace - psix.real,XSize)
	i=i+1


j=0
Velocity = np.gradient(Position ,da, axis=0) 
VelocityA =np.zeros(Velocity.shape)
VelocityN =np.zeros(Velocity.shape)

for ScaleFactor in a_value:
	VelocityA[j] = (psix0.real)/lb.GrowthFunctionAnalytic(ScaleFactor0)*H_0**2/H(ScaleFactor)*(3/2.*Omega_matter**1*ScaleFactor**(-2.)*lb.GrowthFunctionAnalytic(ScaleFactor) - 5/2.*Omega_matter/ScaleFactor)
	plt.plot(Position[j],Velocity[j]*H(ScaleFactor)*ScaleFactor**2,c='r')
	plt.scatter(Position[j],VelocityA[j])
	plt.xlabel('Position')
	plt.ylabel('Velocity')
	plt.xlim([-20,70])
	plt.ylim([-200000,200000])
	plt.xticks([])
	plt.yticks([])
	#~ string =  'phaseSingleKmode' + str(j) +'.png'
	string =  'phaseRandom' + str(j) +'.png'
	plt.savefig(string)
 	plt.clf()
	j=j+1
sys.exit()
