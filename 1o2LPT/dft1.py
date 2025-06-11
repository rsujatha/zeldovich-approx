import numpy as np
import math
import pylab
import matplotlib.pyplot as plt
import sys
import math
import library as lb


GridSize=100
dk=0.1
dx=2*np.pi/(dk*GridSize)

kspace = np.array(range(-GridSize/2,GridSize/2))*dk
#~ print kspace
xspace = np.array(range(-GridSize/2,GridSize/2))*dx
k_x, k_y = np.meshgrid(kspace,kspace, sparse=True)
x, y = np.meshgrid(xspace,xspace, sparse=True)

def IDFT2 (fnc):


###############################################################################
	#~ omega=2*np.pi*k
	fnc=np.matrix(fnc)    ### Make function suitable for matrix operations
	N_kx=np.size(fnc)
	#~ delta_k=k[1]-k[0]
	nx=np.arange(-N_kx/2,N_kx/2,1)

	#~ T=delta_t*N
	#~ omega=2*np.pi*n/T

	
##################################################################################
############################   THIS PART
############################   COMPRISES OF THE
############################   DFT ALGORITHM

	
	ma_x=np.matrix(nx)
	
	matx=ma_x.transpose()*ma_x
	px=np.exp(2*np.pi*matx*1j/N_kx)
	fxy=fnc*px
	#~ fmm=fm*delta_t
###########################
###########################
###########################
##############################################################################
			
	#~ plt.xlabel('omega')
	#~ plt.ylabel('f_tilde*dt')
	#~ plt.plot(omega,fm.imag*delta_t)
	#~ plt.title('Fourier Transform')
	#~ plt.show()		
	return fxy
	
	
	

mean=0
def P(k):
	rows=np.size(k)
	index=np.where(k!=0)
	kc=3.0
	P=np.ones(rows)
	P[index]=np.exp(-k[index]/kc)/(1e4*np.abs(k[index])**3)
	#~ P[index]=np.exp(-k[index]/kc)
	return P

kx      = np.array(range(-GridSize/2,1))*dk


#~ xx, yy = np.meshgrid(kx,ky, sparse=True)


#~ abs_k  = np.sqrt(xx**2+yy**2)
variance=P(kx)/2
#~ print variance
#~ print np.shape(variance)
#~ print variance.shape
#~ print variance
#~ print GridSize/2
np.random.seed(10000)
deltaki  = np.random.normal(mean,np.sqrt(variance), GridSize/2+1) + np.random.normal(mean,np.sqrt(variance), GridSize/2+1)*1j
#~ print deltaki.shape
deltakf  = np.zeros(GridSize)+1j*0
#~ print deltakf.shape
#~ print deltaki.shape
deltakf[0:GridSize/2+1]=deltaki[0:]
deltakf[GridSize/2]=0
deltakf[GridSize/2+1:]=deltaki[1:-1][::-1].conjugate()
deltakf[0]=np.real(deltakf[0])


ksquare=kspace**2
deltax=IDFT2 (deltakf)








psik=deltakf/ksquare
index=np.where(ksquare==0)

psik[index]=0

psix=IDFT2(psik)

#~ plt.imshow(np.real(psix))
#~ plt.colorbar()
#~ 
#~ plt.imshow(np.imag(psix))
#~ plt.colorbar()
psix=np.array(psix)
#~ print psix
grad_psix=np.gradient(np.real(psix),dx,axis=1)


#~ print grad_psix
#~ plt.imshow(np.imag(grad_psix))
#~ plt.show()
a_value=np.linspace(0.0001,0.5,100)


deltax=np.array(deltax)

p=0
for a in a_value:
	deltax_final=np.zeros(GridSize)
	D = lb.GrowthFunctionAnalytic(a)
	gradx=np.array(D*grad_psix/dx)
	
	for i in range(0,GridSize):
		if (0<=i-int(gradx[0][i])<GridSize):
			deltax_final[i-int(gradx[0][i])]=deltax_final[i-int(gradx[0][i])]+np.real(deltax[0][i])
	deltax_final=np.matrix(deltax_final)		
	plt.imshow(deltax_final,vmin=-2, vmax=2, cmap='binary', aspect='auto')
	string='plot'+str(p)+'.png'
	plt.savefig(string)
	p=p+1
	plt.clf()
	
	

#~ f=np.fft.irfft(deltak)
#~ print np.size(f),np.size(deltak)
#~ plt.imshow(np.abs(deltakf),cmap='binary')
#~ plt.imshow(np.abs(deltakf),aspect='auto')
#~ plt.colorbar()
#~ plt.imshow(np.real(f),cmap='binary')
#~ plt.imshow(np.real(deltax))


#~ plt.colorbar()
#~ plt.imshow(np.abs(f))

#~ plt.show()
#~ plt.draw()
#~ plt.pause(1) # <-------
#~ raw_input("<Hit Enter To Close>")
#~ plt.close()

