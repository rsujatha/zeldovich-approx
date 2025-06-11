from __future__ import division
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy import integrate

class cosmology(object):
	def __init__(self,Omega_matter = 0.3,Omega_lambda = 0.7,H_0=70.,ns=0.96,sigma_8 = 0.8 ,Omega_baryon = 0.045 ):
		self.Omega_matter=Omega_matter
		self.Omega_lambda=Omega_lambda
		self.H_0=H_0
		self.hubble=self.H_0/100.
		self.rho_c = 9.5e-27    ## in SI units ?? why do i need this here
		self.ns = ns
		self.sigma_8 = sigma_8
		self.Omega_baryon = Omega_baryon
	def H(self,a):
		return self.H_0*(self.Omega_matter*a**(-3) + self.Omega_lambda + (1-self.Omega_lambda-self.Omega_matter)*a**(-4))**(1/2)

	def GrowthFunctionAnalytic(self,a):
		a=np.array(a)+1e-15
		D=np.ones(np.size(a))
		H=np.ones(np.size(a))
		H=self.H_0*(self.Omega_matter*(a**(-3))+self.Omega_lambda)**(1/2)
		#print sp.hyp2f1(5/6,3/2,11/6,-a**3*self.Omega_lambda/self.Omega_matter)
		D=(H/self.H_0)*a**(5/2)/np.sqrt(self.Omega_matter)*sp.hyp2f1(5/6,3/2,11/6,-a**3*self.Omega_lambda/self.Omega_matter)
		return D


####### BBKS TRANSFER FUNCTION: Dodelson Chapter: Inhomogeneities , Page 205  ######################
	def BBKS_tf(self,k):
		Gamma = 	self.Omega_matter*self.hubble  #from Dodelson
		# ~ Gamma = 	self.Omega_matter*self.hubble * np.exp(-self.Omega_baryon-self.Omega_baryon/self.Omega_matter)  ## correction from sugiyama1994
		q = k/(Gamma)
		return np.log(1+2.34*q)/(2.34*q)*(1+3.89*q+(16.2*q)**2+(5.47*q)**3+(6.71*q)**4)**(-0.25)

	def Wk(self,k,R):
		return 3/(k*R)**3*(np.sin(k*R)-(k*R)*np.cos(k*R))
		
	def PS(self,k,z,T):
		"""
		Input
		k in h Mpc^1
		z redshift
		T the tranfer function
		
		Outputs 
		Pk the power spectrum
		"""
		R=8.
		integrand = 1/(2*np.pi**2)*k**(self.ns+2.)*T**2*cosmology.Wk(self,k,R)**2
		igrate = np.trapz(integrand,k)
		# ~ print self.ns
		# ~ print igrate
		# ~ print self.sigma_8
		SigmaSquare=self.sigma_8**2
		NormConst = SigmaSquare/igrate
		# ~ print NormConst
		return NormConst*k**self.ns*(T)**2 *cosmology.GrowthFunctionAnalytic(self,1/(1+z))**2/cosmology.GrowthFunctionAnalytic(self,1)**2 

	def GaussianWk(self,k,R):
		### R is the standard deviation of the Gaussian.
		return np.exp(-k**2*R**2/2)


	def PS_calc(self,BoxSize,deltax,NofBins=100,kmin=None,kmax=None):
		"""
		Calculates power spectrum P(k) 
		Input: 
		BoxSize - Size of the box 
		deltax  - real space density fluctuation 
		NofBins - number of k bins
		Output:
		Pk   -  P(k)  
		k_array -  k in h Mpc^{-1}
		"""
		
		dk = 2*np.pi/BoxSize
		GridSize = deltax.shape[0]
		kspace = np.concatenate([range(0,int(GridSize/2)),range(-int(GridSize/2),0)])*dk	
		
		if kmin==None:
			kmin = dk
		if kmax==None:
			kmax = GridSize/2*dk
		
		k_bin = np.logspace(np.log10(kmin),np.log10(kmax),NofBins)
		
		k_x, k_y ,k_z = np.meshgrid(kspace,kspace,kspace[0:int(GridSize/2)+1], indexing='ij')
		k = np.sqrt(k_x**2 + k_y**2 + k_z**2)
		deltak = np.fft.rfftn(deltax)/(GridSize)**3   ### deltak_{our convention} = 1/GridSize**3  deltak_{numpy}(See Aseem's Math Methods Notes and numpy fft documentation)
		Pk = deltak*deltak.conjugate()*BoxSize**3 ## P(k) = BoxSize**3 |deltak_{our convention}|**2
		Pk = Pk.flatten()
		k = k.flatten()
		### The following line will bin the P(k) logarithmically and average over values belonging to the same bin
		Pk_avg = np.histogram(k , bins = k_bin , weights = Pk)[0]/ (np.histogram(k , bins = k_bin )[0]+1e-15)
		return Pk_avg, np.sqrt(k_bin[:-1]*k_bin[1:])
		

