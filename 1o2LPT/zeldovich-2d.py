import numpy as np
import math
import pylab
import matplotlib.pyplot as plt
import sys
import math
import library as lb
import mathtools as mt
from matplotlib.colors import LogNorm
#############################################
##Scale factor of the Power Spectrum used####
ScaleFactor0 = 0.0001
#############################################
XSize = 4.
GridSize=32
dx = XSize/float(GridSize)
dk = 2*np.pi/(dx*GridSize)
kspace =np.concatenate([range(0,int(GridSize/2)),range(-int(GridSize/2),0)])*dk
print (kspace)

xspace = np.array(range(0,int(GridSize)))*dx
k_x, k_y = np.meshgrid(kspace,kspace, sparse=True)
x, y = np.meshgrid(xspace,xspace, sparse=True)	

mean=0
def P(k):
	kc=20
	P=0.1*(dk)**2/(2*np.pi)**2*np.exp(-k/kc)*(k+1e-15)
	return P


plt.plot(kspace,P(kspace))
ksquare  =(k_x**2+k_y**2)
variance = P(np.sqrt(ksquare))/2


###############                                                                                  ###############################
############### Initialising delta k as random gaussian variable taking care to make its DFT real###############################
################                                                                                  ###############################

deltak 									= np.random.normal(mean,np.sqrt(variance), [int(GridSize),int(GridSize)]) + np.random.normal(mean,np.sqrt(variance), [int(GridSize),int(GridSize)])*1j
deltak[0,0]								= 0
deltak[0,int(GridSize/2)]					= deltak[0,int(GridSize/2)].real
deltak[int(GridSize/2),0]					= deltak[int(GridSize/2),0].real
deltak[int(GridSize/2),int(GridSize/2)]			= deltak[int(GridSize/2),int(GridSize/2)].real
deltak[int(GridSize/2)+1:,0]					= deltak[1:int(GridSize/2),0][::-1].conjugate()
deltak[0,int(GridSize/2)+1:]					= deltak[0,1:int(GridSize/2)][::-1].conjugate()
 
deltak[int(GridSize/2+1):,int(GridSize/2)]		= deltak[1:int(GridSize/2),int(GridSize/2)][::-1].conjugate()
deltak[1:int(GridSize),int(GridSize/2+1):] = deltak[1:int(GridSize),1:int(GridSize/2)][::-1,::-1].conjugate()



deltax_fast=np.fft.ifft2(deltak)
fig, ax = plt.subplots(nrows=1,ncols=2)
plt.subplot(1,2,1)
plt.title('FFT2')
plt.imshow(np.real(deltax_fast),cmap='binary')
plt.colorbar()


plt.subplot(1,2,2)
deltax=mt.IDFT2 (deltak)
plt.imshow(np.real(deltax),cmap='binary')
plt.colorbar()
plt.title('DFT2')
plt.savefig('ift.pdf')
plt.show()



psik_scalar = -deltak/(ksquare + 1e-15)
psik_scalar[0,0] = 0

psik_vector_x = psik_scalar * k_x * 1j
psik_vector_y = psik_scalar * k_y * 1j


psi_vector_x0 = mt.IDFT2(psik_vector_x)
psi_vector_y0 = mt.IDFT2(psik_vector_y)
print (psi_vector_x0)
k=0
plt.gca().invert_yaxis()
for ScaleFactor in np.linspace(0.00,0.01,100):
	plt.clf()
	psi_vector_x = lb.GrowthFunctionAnalytic(ScaleFactor)/lb.GrowthFunctionAnalytic(ScaleFactor0)*psi_vector_x0
	psi_vector_y = lb.GrowthFunctionAnalytic(ScaleFactor)/lb.GrowthFunctionAnalytic(ScaleFactor0)*psi_vector_y0
	i = range(0,GridSize)
	j = range(0,GridSize)
	ii ,jj = np.meshgrid(i ,j ,sparse=True)
	Position = np.zeros([3,GridSize,GridSize])
	Position[0,:,:] = np.mod(ii* dx - psi_vector_x.real,XSize)
	Position[1,:,:] = np.mod(jj* dx - psi_vector_y.real,XSize)
	
	
	Xposition=Position[0,:,:].reshape((GridSize*GridSize))
	Yposition=Position[1,:,:].reshape((GridSize*GridSize))
	
	plt.scatter(Xposition,Yposition,30)
	titleString = 'ScaleFactor = '+ str(ScaleFactor)
	plt.title=(titleString)
	string='pltno-'+ str(k)+'.png'
	plt.xticks([])
	plt.yticks([])
	plt.savefig(string,dpi=100,bbox_inches='tight', pad_inches=0)
	k=k+1
	

sys.exit()
