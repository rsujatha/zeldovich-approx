import numpy as np
import math
import pylab
import matplotlib.pyplot as plt


def DFT (fnc,time):


###############################################################################
	N=np.size(fnc)
	delta_t=time[1]-time[0]
	n=np.arange(-N/2,N/2,1)
	T=delta_t*N
	omega=2*np.pi*n/T

	
##################################################################################
############################   THIS PART
############################   COMPRISES OF THE
############################   DFT ALGORITHM

	fnc=np.matrix(fnc)
	ma=np.matrix(n)
	mat=ma.transpose()*ma
	p=np.exp(-2*np.pi*mat*1j/N)
	fm=p*fnc.transpose()
	fmm=fm*delta_t
###########################
###########################   THIS PART DOES THE PLOTTING WITH APPROPRIATE 
###########################   X AXIS AND Y AXIS
##############################################################################
			
	plt.xlabel('omega')
	plt.ylabel('f_tilde*dt')
	plt.plot(omega,fm.imag*delta_t)
	plt.title('Fourier Transform')
	plt.show()		
	return 

def autocorr(fnc,time):
	N=np.size(fnc)
	delta_t=time[1]-time[0]
### tau should be chosen to be an integral multiple of dt	
	tau=delta_t*np.arange(-(N-1),(N),1)
### intitialise autocorrelation function
	r=np.zeros(N)	
	for i in range(0,N):
		r[0:N-i]=r[0:N-i]+fnc[i:]*fnc[i]*delta_t
### making use of the fact that autocorrelation function is even and reversing and attaching this array for negative values
	r=np.append(r[:0:-1],r)
	plt.plot(tau,r)
	plt.show()
	return tau,r

def IDFT2 (fnc):
	"""
	The input (must be even) should have the term for zero frequency
	in the low-order corner of the two axes, the positive frequency terms in
	the first half of these axes, the term for the Nyquist frequency in the
	middle of the axes and the negative frequency terms in the second half of
	both axes, in order of decreasingly negative frequency. Note that the Negative
	frequency limit is larger than the positve frequency.

	"""

###############################################################################
	
	fnc=np.matrix(fnc)    ### Make function suitable for matrix operations
	N_kx,N_ky=fnc.shape
	nx=np.arange(0,N_kx,1)
	ny=np.arange(0,N_ky,1)
	
	
##################################################################################
############################   THIS PART
############################   COMPRISES OF THE
############################   DFT ALGORITHM

	
	ma_x=np.matrix(nx)
	ma_y=np.matrix(ny)
	matx=ma_x.transpose()*ma_x
	maty=ma_y.transpose()*ma_y
	px=np.exp(2*np.pi*matx*1j/N_kx)
	py=np.exp(2*np.pi*maty*1j/N_ky)
	fxy=px*fnc*py
	
###########################
###########################
###########################
##############################################################################
			
		
	return fxy
	
def IDFT2_version2 (fnc):
	"""
	Differs from the IDFT2 in the order of input.
	The input (must be even) should have negative frequencies in the 
	lower order of both the axis and arranged in increasing order of frequency till the maximum positive frequency.
	Note that the Negative frequency limit is larger than the positve frequency.

	"""

###############################################################################
	
	fnc=np.matrix(fnc)    ### Make function suitable for matrix operations
	N_kx,N_ky=fnc.shape
	nx=np.arange(-N_kx/2,N_kx/2,1)
	ny=np.arange(-N_ky/2,N_ky/2,1)
	
	
##################################################################################
############################   THIS PART
############################   COMPRISES OF THE
############################   DFT ALGORITHM

	
	ma_x=np.matrix(nx)
	ma_y=np.matrix(ny)
	matx=ma_x.transpose()*ma_x
	maty=ma_y.transpose()*ma_y
	px=np.exp(2*np.pi*matx*1j/N_kx)
	py=np.exp(2*np.pi*maty*1j/N_ky)
	fxy=px*fnc*py
	
###########################
###########################
###########################
##############################################################################
			
		
	return fxy
	
