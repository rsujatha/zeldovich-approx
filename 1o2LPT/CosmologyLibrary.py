from __future__ import division
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import matplotlib.image as mpimg
from scipy import integrate

class cosmology(object):
	def __init__(self,Omega_matter = 0.3,Omega_lambda = 0.7,H_0=70.):
		self.Omega_matter=Omega_matter
		self.Omega_lambda=0.7
		self.H_0=70.
		self.h=self.H_0/100.
		self.rho_c = 9.5e-27    ## in SI units
Omega_matter=0.3
Omega_lambda=0.7
H_0=70.
h=0.7


def GrowthFunctionAnalytic(a,Omega_matter=0.3,Omega_lambda=0.7):
	a=np.array(a)+1e-15
	D=np.ones(np.size(a))
	H=np.ones(np.size(a))
	H=H_0*(Omega_matter*(a**(-3))+Omega_lambda)**(1/2)
	D=(H/H_0)*a**(5/2)/np.sqrt(Omega_matter)*sp.hyp2f1(5/6,3/2,11/6,-a**3*Omega_lambda/Omega_matter)
	return D
	
		
def GrowthFunctionNumerical(arg,Omega_matter,Omega_lambda):   ##### Less Efficient than GrowthFunctionAnalytic 
	i=0;
	D=np.zeros(np.size(arg))
	for a in arg:
		def integrand(x):
			return 1/(x*(Omega_matter*x**(-3) + Omega_lambda + (1-Omega_lambda-Omega_matter))**(1/2))**3
		H=H_0*(Omega_matter*a**(-3) + Omega_lambda + (1-Omega_lambda-Omega_matter)**2*a**(-2))**(1/2)
		D[i]=5./2.*Omega_matter*(H/H_0)*integrate.quad(integrand,0,a)[0]
		i+=1 
	return D


def H(a,Omega_matter=0.3,Omega_lambda=0.7):
	return H_0*(Omega_matter*a**(-3) + Omega_lambda + (1-Omega_lambda-Omega_matter)**2*a**(-2))**(1/2)






####### BBKS TRANSFER FUNCTION: Dodelson Chapter: Inhomogeneities , Page 205  ######################
def BBKS_tf(k,Omega_matter=0.3,Omega_lambda=0.7):
	Gamma = 	Omega_matter 
	q = k/Gamma
	return np.log(1+2.34*q)/(2.34*q)*(1+3.89*q+(1.62*q)**2+(5.47*q)**3+(6.71*q)**4)**(-0.25)

def Wk(k,R):
	return 3/(k*R)**3*(np.sin(k*R)-(k*R)*np.cos(k*R))
	

def integrand(k,Omega_matter=0.3,Omega_lambda=0.7):
	## This integrand is specifically for bbks transfer function.
	ns=0.96
	R=8.
	return 1/(2*np.pi**2)*k**(ns+2)*BBKS_tf(k,Omega_matter,Omega_lambda)**2*Wk(k,R)**2
		

	
	
			
def P(k,z,Omega_matter=0.3,Omega_lambda=0.7):
	ns=0.96
	igrate=integrate.quad(lambda k:integrand(k,Omega_matter,Omega_lambda),0,np.inf)[0]	
	SigmaSquare=0.8**2
	NormConst = SigmaSquare/igrate
	return NormConst*k**ns*(BBKS_tf(k))**2 *GrowthFunctionAnalytic(1/(1+z),Omega_matter=0.3,Omega_lambda=0.7)**2/GrowthFunctionAnalytic(1,Omega_matter=0.3,Omega_lambda=0.7)**2 

def PS(k,z,T,Omega_matter=0.3,Omega_lambda=0.7):
	ns=0.96
	R=8.
	integrand = 1/(2*np.pi**2)*k**(ns+2)*T**2*Wk(k,R)**2
	igrate = np.trapz(integrand,k)
	SigmaSquare=0.8**2
	NormConst = SigmaSquare/igrate
	return NormConst*k**ns*(T)**2 *GrowthFunctionAnalytic(1/(1+z),Omega_matter,Omega_lambda)**2/GrowthFunctionAnalytic(1,Omega_matter,Omega_lambda)**2 

def GaussianWk(k,R):
	### A normalised gaussian 1/sqrt(2pi)R exp(-x^2/2R^2) can be used to smooth a field by taking its convolution with the gaussian
	### In fourier space this amounts to multiplying Wk with fourier space field 
	### R is the standard deviation of the Gaussian.
	"""
	The function gives the fourier space normalised gaussian of standard deviation R
	"""
	return np.exp(-k**2*R**2/2)



#~ 
#~ fig=plt.figure(figsize=(10,8), dpi= 100, facecolor='w', edgecolor='k')
#~ plt.subplot(1,2,20
#~ logk=np.linspace(-3,1)
#~ k=10**(logk)
#~ plt.loglog(k,k**3*P(k,100)/(2*np.pi**2),label='z=0')
#~ plt.title('Dimensionless Power Spectrum $\Delta^2$')
#~ plt.xlabel('k  (h$Mpc^{-1}$)',fontsize=20)
#~ plt.ylabel('$\Delta^2 \equiv k^3$ P(k)/(2 $\Pi^2$)',fontsize=25)
#~ plt.savefig('DPowerSpectrum.svg', bbox_inches='tight', pad_inches=0)

#~ plt.clf()
#~ plt.subplot(1,2,1)
#~ logk=np.linspace(-3,1)
#~ k=10**(logk)
#~ plt.loglog(k,P(k,100),label='z=0')
#~ plt.title(' Power Spectrum ')
#~ plt.xlabel('k  (h$Mpc^{-1}$)',fontsize=20)
#~ plt.ylabel('P(k)  ($h^{-1}Mpc)^3$',fontsize=25)
#~ plt.savefig('PowerSpectrum.svg', bbox_inches='tight', pad_inches=0)
#~ 
#~ plt.show()
