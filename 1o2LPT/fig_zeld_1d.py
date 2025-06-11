import numpy as np
import math
import pylab
import matplotlib.pyplot as plt
import sys
import math
import library as lb
import mathtools as mt

GridSize=64
dk=0.1
dx=2*np.pi/(dk*GridSize)
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

def P_single(k):
	tiny=1e-15
	k_fundamental=0.2
	P=np.zeros(np.size(k))
	P=P+tiny
	index=np.where((k<0.2+tiny) & (k>0.2-tiny))[0]
	P[index]=2
	return P
	
variance = P_single(np.abs(kspace))


###############                                                                                  ###############################
############### Initialising delta k as random gaussian variable taking care to make its IFFT real###############################
###############                                                                                  ###############################
deltak = np.random.normal(mean, variance , GridSize) + np.random.normal(mean, variance , GridSize)*1j
deltak[GridSize/2+1:] = deltak[1:GridSize/2][::-1].conjugate()
deltak[0] = 0
deltak[GridSize/2] = np.real(deltak[GridSize/2])
deltax = GridSize *np.fft.ifft(deltak)	

psik = -deltak/abs(kspace+1e-15)
index = np.where(kspace==0)
psik[index] = 0
psix0 = GridSize*np.fft.ifft(psik)
N_a=20
Position = np.zeros([N_a,GridSize])
Position_PERIODIC = np.zeros([N_a,GridSize])

i=0
a_value = np.linspace(0.00000001,0.0001,N_a)
print a_value.shape
da = a_value[2]-a_value[1]

for ScaleFactor in a_value:	
	
	psix = lb.GrowthFunctionAnalytic(ScaleFactor)/lb.GrowthFunctionAnalytic(ScaleFactor0)*psix0.real
	Position[i] = xspace- psix
	i=i+1


print Position_PERIODIC.shape
VelocityA =np.zeros(Position.shape)



f, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True,figsize=(15,5))
j=1
ScaleFactor = a_value[j]
VelocityA[j] = (psix0.real)/lb.GrowthFunctionAnalytic(ScaleFactor0)*H_0**2/H(ScaleFactor)*(3/2.*Omega_matter**1*ScaleFactor**(-2.)*lb.GrowthFunctionAnalytic(ScaleFactor) - 5/2.*Omega_matter/ScaleFactor)
ax1.plot(Position[j],VelocityA[j])
ax1.scatter(Position[j],VelocityA[j])
ax1.set_xlabel('Position')
ax1.set_ylabel('Velocity')
ax1.set_xlim([-20,70])
ax1.set_ylim([-200000,200000])
ax1.legend()

j=7
ScaleFactor = a_value[j]
VelocityA[j] = (psix0.real)/lb.GrowthFunctionAnalytic(ScaleFactor0)*H_0**2/H(ScaleFactor)*(3/2.*Omega_matter**1*ScaleFactor**(-2.)*lb.GrowthFunctionAnalytic(ScaleFactor) - 5/2.*Omega_matter/ScaleFactor)
ax2.plot(Position[j],VelocityA[j])
ax2.scatter(Position[j],VelocityA[j])
ax2.set_xlabel('Position')
ax2.set_ylabel('Velocity')
ax2.set_xlim([-20,70])
ax2.set_ylim([-200000,200000])
ax2.legend()

j=17
ScaleFactor = a_value[j]
VelocityA[j] = (psix0.real)/lb.GrowthFunctionAnalytic(ScaleFactor0)*H_0**2/H(ScaleFactor)*(3/2.*Omega_matter**1*ScaleFactor**(-2.)*lb.GrowthFunctionAnalytic(ScaleFactor) - 5/2.*Omega_matter/ScaleFactor)
ax3.plot(Position[j],VelocityA[j])
ax3.scatter(Position[j],VelocityA[j])
ax3.set_xlabel('Position')
ax3.set_ylabel('Velocity')
ax3.set_xlim([-20,70])
ax3.set_ylim([-200000,200000])
ax3.legend()
plt.savefig('fig_zeld_1d.pdf',bbox_inches='tight', pad_inches=0)
plt.show()
plt.clf()

