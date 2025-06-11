###Class containing most of the codes required for zeldovich approximation and 2lpt

from __future__ import division
import numpy as np
import CosmologyLibraryM as lb
import intpol
		
class initial_density_field(lb.cosmology):
	
	def __init__(self,GridSize=128, XSize = 1.28,Seed = 300000,Omega_matter = 0.3,Omega_lambda = 0.7,H_0=70.,ns=0.96, sigma_8 = 0.8 ,Omega_baryon = 0.045):
		super(initial_density_field,self).__init__(Omega_matter = Omega_matter,Omega_lambda = Omega_lambda ,H_0=H_0,ns=ns ,sigma_8 = sigma_8 ,Omega_baryon = Omega_baryon )
		#~ lb.cosmology.__init__(self)
		self.RedShift0 = 100.
		self.ScaleFactor0 = 1./(1+self.RedShift0)
		self.GridSize = GridSize
		self.XSize =  XSize
		self.dx = XSize/float(GridSize)
		self.dk=2*np.pi/(self.dx*self.GridSize)
		np.random.seed(Seed) 
		self.rand_state = np.random.get_state()
		
		
	def x_range(self):
		xspace = np.array(range(0,self.GridSize))*self.dx
		return xspace	
		
	def k_array(self):
		kspace = np.concatenate([range(0,int(self.GridSize/2)),range(-int(self.GridSize/2),0)])*self.dk	
		k_x, k_y ,k_z = np.meshgrid(kspace,kspace,kspace, indexing='ij')
		return k_x, k_y, k_z
		
	def ksquare(self):
		kx, ky, kz = initial_density_field.k_array(self)
		ksquare = kx**2 + ky**2 + kz**2
		return ksquare

	def PS_initial(self):
		"""
		returns 
		1)unique k_values (in h Mpc^-1)
		2)the power spectrum (for unique k values) used to generate the initial density field 
		3)the reverse indices which can be used to reconstruct P(k) for the three dimensional k space. 
		"""
		tiny=1e-15
		ksquare  = initial_density_field.ksquare(self)
		k_unique ,ind = np.unique(np.sqrt(ksquare.flatten()),return_inverse = True)
		T_bbks = lb.cosmology.BBKS_tf(self , k_unique+tiny)
		PS = lb.cosmology.PS(self,k_unique+tiny,self.RedShift0,T_bbks)
		return k_unique,PS,ind


	def var(self):
		"""
		returns the variance for the imaginary and real random numbers used to generate initial density in fourier space. 
		"""
		
		k_unique ,Pk, rev_ind = initial_density_field.PS_initial(self) 
		#~ ksquare  = initial_density_field.ksquare(self)
		#~ k_unique ,ind = np.unique(np.sqrt(ksquare.flatten()),return_inverse = True)
		#~ T_bbks = lb.cosmology.BBKS_tf(self , k_unique+tiny)
		Pk_discrete_half = (self.dk)**3/(2*np.pi)**3*Pk/2
		return Pk_discrete_half[rev_ind].reshape([self.GridSize,self.GridSize,self.GridSize])
		
	def initial_deltak(self):
		mean=0
		np.random.set_state(self.rand_state)
		variance = initial_density_field.var(self)
		deltak = np.random.normal(mean,np.sqrt(variance), [self.GridSize,self.GridSize,self.GridSize]) + np.random.normal(mean,np.sqrt(variance), [self.GridSize,self.GridSize,self.GridSize])*1j
		deltak[1:,1:,int(self.GridSize/2+1):] = deltak[1:,1:,1:int(self.GridSize/2)][::-1,::-1,::-1].conjugate()
		deltak[0,0,0]=0
		deltak[0,0,int(self.GridSize/2)] = deltak[0,0,int(self.GridSize/2)].real*np.sqrt(2) ## this multiplication by sqrt(2) is since the random number is purely real the variance must be twice
		deltak[int(self.GridSize/2),0,0] = deltak[int(self.GridSize/2),0,0].real*np.sqrt(2)
		deltak[0,int(self.GridSize/2),0] = deltak[0,int(self.GridSize/2),0].real*np.sqrt(2)
		deltak[0,int(self.GridSize/2),int(self.GridSize/2)] = deltak[0,int(self.GridSize/2),int(self.GridSize/2)].real*np.sqrt(2)
		deltak[int(self.GridSize/2),int(self.GridSize/2),0] = deltak[int(self.GridSize/2),int(self.GridSize/2),0].real*np.sqrt(2)
		deltak[int(self.GridSize/2),0,int(self.GridSize/2)] = deltak[int(self.GridSize/2),0,int(self.GridSize/2)].real*np.sqrt(2)
		deltak[int(self.GridSize/2),int(self.GridSize/2),int(self.GridSize/2)] = deltak[int(self.GridSize/2),int(self.GridSize/2),int(self.GridSize/2)].real*np.sqrt(2)
		deltak[0,0,int(self.GridSize/2+1):] = deltak[0,0,1:int(self.GridSize/2)][::-1].conjugate()
		deltak[0,int(self.GridSize/2+1):,0] = deltak[0,1:int(self.GridSize/2),0][::-1].conjugate()
		deltak[int(self.GridSize/2+1):,0,0] = deltak[1:int(self.GridSize/2),0,0][::-1].conjugate()
		deltak[0,1:int(self.GridSize),int(self.GridSize/2+1):] = deltak[0,1:int(self.GridSize),1:int(self.GridSize/2)][::-1,::-1].conjugate()
		deltak[1:int(self.GridSize),0,int(self.GridSize/2+1):] = deltak[1:int(self.GridSize),0,1:int(self.GridSize/2)][::-1,::-1].conjugate()
		deltak[1:int(self.GridSize),int(self.GridSize/2+1):,0] = deltak[1:int(self.GridSize),1:int(self.GridSize/2),0][::-1,::-1].conjugate()
		deltak[1:int(self.GridSize),int(self.GridSize/2+1):,int(self.GridSize/2)] = deltak[1:int(self.GridSize),1:int(self.GridSize/2),int(self.GridSize/2)][::-1,::-1].conjugate()
		return deltak
		
	def initial_deltax(self):
		deltak = initial_density_field.initial_deltak(self)
		return self.GridSize**3 *np.fft.ifftn(deltak)


class LPT(initial_density_field,intpol.pmInterpolation):
	"""
	Poisson_Solver 0 ----> Poor Mans Poisson Solver (Boris & Roberts(1969))	- Continous Greens Function Kernel
				1 ----> Using second order accurate Central Finite difference Fourier space Kernel	
	Gradient       0 ----> Continous Green Function Kernel
				1 ----> Using second order accurate Central Finite difference Fourier space Kernel	
				
	"""	
	
	def __init__(self,GridSize=128, XSize = 1.28,Seed = 300000,Omega_matter = 0.3,Omega_lambda = 0.7,H_0=70.,ns=0.96, sigma_8 = 0.8 ,Omega_baryon = 0.045,deltax = None,redshift = None,Poisson_Solver = 0,Gradient = 0):
		super(LPT,self).__init__(GridSize=GridSize, XSize = XSize, Seed = Seed, Omega_matter = Omega_matter,Omega_lambda = Omega_lambda ,H_0=H_0,ns=ns ,sigma_8 = sigma_8 ,Omega_baryon = Omega_baryon)
		#~ lb.cosmology.__init__(self)
		intpol.pmInterpolation.__init__(self)
		self.deltax = deltax
		self.rsofdeltax = redshift
		self.Poisson_Solver = Poisson_Solver   
		self.Gradient = Gradient
		
	def psik_scalar(self):
		ksquare = initial_density_field.ksquare(self)
		if self.deltax is None:
			deltak = initial_density_field.initial_deltak(self)
			if self.Poisson_Solver ==0:
				kernel = -1/(ksquare + 1e-15)
				psik_scalar = kernel * deltak
				psik_scalar[0,0,0] = 0
			elif self.Poisson_Solver ==1:
				kx,ky,kz = initial_density_field.k_array(self)
				kernel = - (self.dx/2.)**2/((np.sin(kx*self.dx/2))**2 + (np.sin(ky*self.dx/2))**2 + (np.sin(kz*self.dx/2))**2)
				psik_scalar = kernel * deltak
				psik_scalar[0,0,0] = 0
				
		else:
			Scalefactor = 1./(1.+ self.rsofdeltax)
			deltax0 = self.deltax*lb.cosmology.GrowthFunctionAnalytic(self,self.ScaleFactor0)/lb.cosmology.GrowthFunctionAnalytic(self,Scalefactor)
			deltak = 1/self.GridSize**3 * np.fft.fftn(deltax0)
			if self.Poisson_Solver ==0:
				kernel = -1/(ksquare + 1e-15)
				psik_scalar = kernel * deltak
				psik_scalar[0,0,0] = 0
			elif self.Poisson_Solver ==1:
				kx,ky,kz = initial_density_field.k_array(self)
				kernel = - (self.dx/2.)**2/((np.sin(kx*self.dx/2))**2 + (np.sin(ky*self.dx/2))**2 + (np.sin(kz*self.dx/2))**2)
				psik_scalar = kernel * deltak
				psik_scalar[0,0,0] = 0
		return psik_scalar
	
	
	def tidal_tensor(self,psik_scalar=None,k_x=None,k_y=None,k_z=None,flag=None):
		"""
		If the flag is 0 then reshape the tidal tensor as a N^3 x 3 x 3 array
		else tidal tensor is N x N x N x 3 x 3 array
		"""
		if psik_scalar is None:
			psik_scalar = LPT.psik_scalar(self)
		if k_x is None:
			k_x=initial_density_field.k_array(self)[0]
		if k_y is None:
			k_y=initial_density_field.k_array(self)[1]
		if k_z is None:
			k_z=initial_density_field.k_array(self)[2]
		if flag is None:
			flag= 0
		if flag==0:
			TidalTensor = np.zeros([self.GridSize**3,3,3])
			TidalTensor[:,0,0] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_x * k_x )).reshape([self.GridSize**3])
			TidalTensor[:,0,1] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_x * k_y )).reshape([self.GridSize**3])
			TidalTensor[:,0,2] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_x * k_z )).reshape([self.GridSize**3])
			TidalTensor[:,1,0] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_y * k_x )).reshape([self.GridSize**3])
			TidalTensor[:,1,1] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_y * k_y )).reshape([self.GridSize**3])
			TidalTensor[:,1,2] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_z * k_z )).reshape([self.GridSize**3])
			TidalTensor[:,2,0] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_z * k_x )).reshape([self.GridSize**3])
			TidalTensor[:,2,1] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_z * k_y )).reshape([self.GridSize**3])
			TidalTensor[:,2,2] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_z * k_z )).reshape([self.GridSize**3])
		else:
			TidalTensor = np.zeros([self.GridSize,self.GridSize,self.GridSize,3,3])
			TidalTensor[:,:,:,0,0] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_x * k_x ))
			TidalTensor[:,:,:,0,1] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_x * k_y ))
			TidalTensor[:,:,:,0,2] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_x * k_z ))
			TidalTensor[:,:,:,1,0] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_y * k_x ))
			TidalTensor[:,:,:,1,1] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_y * k_y ))
			TidalTensor[:,:,:,1,2] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_z * k_z ))
			TidalTensor[:,:,:,2,0] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_z * k_x ))
			TidalTensor[:,:,:,2,1] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_z * k_y ))
			TidalTensor[:,:,:,2,2] = (self.GridSize**3 * np.fft.ifftn( - psik_scalar * k_z * k_z ))
		return TidalTensor
		
		
	def eigval_tidal(self,TidalTensor):
		eigenValue = np.linalg.eigvalsh(TidalTensor)     ####returns eigen values in ascending order
		return eigenValue
		
	def deltax_final(self,eigenValue,RedShift): 
		Scalefactor = 1./(1.+RedShift)
		D = lb.cosmology.GrowthFunctionAnalytic(Scalefactor)
		deltax_final = 1./((1.-D*eigenValue[:,0])*(1.-D*eigenValue[:,1])*(1.-D*eigenValue[:,2]))-1.
		return deltax_final
		 
	def web_classification(self,psik_scalar,kx,ky,kz):
		
		#############Web Classification#################################
		########Void       =0 
		########Sheet   =1
		########Filament      =2
		########Node       =3
		
		WebClassification = np.zeros([self.GridSize**3])
		TidalTensor = LPT.tidal_tensor(self,psik_scalar,kx,ky,kz)
		eigenValue  = LPT.eigval_tidal(self,TidalTensor)
		WebClassification[np.where((eigenValue[:,2]>0)&(eigenValue[:,1]<0))[0]] = 1       
		WebClassification[np.where((eigenValue[:,1]>0)&(eigenValue[:,0]<0))[0]] = 2
		WebClassification[np.where((eigenValue[:,0]>0))[0]] = 3
		WebClassification = WebClassification.reshape([self.GridSize,self.GridSize,self.GridSize])
		return WebClassification
	
	def psik_vector(self):
		k_x,k_y,k_z = initial_density_field.k_array(self)
		psiksc = LPT.psik_scalar(self)
		if self.Gradient ==0:
				kernelx = 1j * k_x
				kernely = 1j * k_y
				kernelz = 1j * k_z
				psik_vector_x = psiksc * kernelx
				psik_vector_y = psiksc * kernely
				psik_vector_z = psiksc * kernelz
		elif self.Gradient ==1:
				kernelx = 1j * np.sin(k_x * self.dx ) / self.dx
				kernely = 1j * np.sin(k_y * self.dx ) / self.dx
				kernelz = 1j * np.sin(k_z * self.dx ) / self.dx
				psik_vector_x = psiksc * kernelx
				psik_vector_y = psiksc * kernely
				psik_vector_z = psiksc * kernelz
		return psik_vector_x, psik_vector_y, psik_vector_z
		
	def psi_vector0(self):
		psik_vector_x, psik_vector_y, psik_vector_z = LPT.psik_vector(self)
		psi_vector_x0 = self.GridSize**3 *  np.fft.ifftn(psik_vector_x)
		psi_vector_y0 = self.GridSize**3 *  np.fft.ifftn(psik_vector_y)
		psi_vector_z0 = self.GridSize**3 *  np.fft.ifftn(psik_vector_z)
		return psi_vector_x0, psi_vector_y0, psi_vector_z0

	def psi_vector(self,RedShift,psi_vector_x0=None,psi_vector_y0=None,psi_vector_z0=None):
		"""
		Input psi_vector_x0,y_0,z0 when dealing with multiple redshift of same instance for speed.
		
		"""
		psi_vector0 = LPT.psi_vector0(self)
		if psi_vector_x0 is None:
			psi_vector_x0 = psi_vector0[0]
		if psi_vector_y0 is None:
			psi_vector_y0 = psi_vector0[1]
		if psi_vector_z0 is None:
			psi_vector_z0 = psi_vector0[2]
			
		Scalefactor = 1./(1+RedShift)
		psi_vector_x = psi_vector_x0*lb.cosmology.GrowthFunctionAnalytic(self,Scalefactor)/lb.cosmology.GrowthFunctionAnalytic(self,self.ScaleFactor0)
		psi_vector_y = psi_vector_y0*lb.cosmology.GrowthFunctionAnalytic(self,Scalefactor)/lb.cosmology.GrowthFunctionAnalytic(self,self.ScaleFactor0)
		psi_vector_z = psi_vector_z0*lb.cosmology.GrowthFunctionAnalytic(self,Scalefactor)/lb.cosmology.GrowthFunctionAnalytic(self,self.ScaleFactor0)
		return psi_vector_x, psi_vector_y, psi_vector_z

			
	def Position_zeld(self,RedShift,psi_vector_x=None,psi_vector_y=None,psi_vector_z=None):
		"""		
		o/p is a 4 dimensional array whose first dimension gives x, y, z zeldovich position
		second third and fourth dimension corresponds to the i,j,k grid cell the particle originally belonged to before 
		zeldovich approximation was applied.
		
		If multiple redshifts are required it is faster to call the psi vectors first and then input them here.

		"""
		
		
		if psi_vector_x is None:
			psi_vector = LPT.psi_vector(self,RedShift)
			psi_vector_x = psi_vector[0].real
		if psi_vector_y is None:
			psi_vector_y = psi_vector[1].real
		if psi_vector_z is None:
			psi_vector_z = psi_vector[2].real
		
		i = range(0,self.GridSize)
		j = range(0,self.GridSize)
		k = range(0,self.GridSize)
		ii ,jj ,kk = np.meshgrid(i ,j ,k, indexing='ij' )
		Position = np.zeros([3,self.GridSize,self.GridSize,self.GridSize])          
		Position[0,:,:,:] = np.mod((ii + 1/2.)* self.dx - psi_vector_x,self.XSize)   ## Particles are initially placed at the center of the boxes whose vertices are the grid location.
		Position[1,:,:,:] = np.mod((jj + 1/2.)* self.dx - psi_vector_y,self.XSize)
		Position[2,:,:,:] = np.mod((kk + 1/2.)* self.dx - psi_vector_z,self.XSize)
		return Position

	def deltax_zeld_cic(self,RedShift,GridSize_cic=None):
		if GridSize_cic is None:
			GridSize_cic = self.GridSize
		Position = LPT.Position_zeld(self,RedShift)
		PositionX = Position[0,:,:,:].flatten()
		PositionY = Position[1,:,:,:].flatten()
		PositionZ = Position[2,:,:,:].flatten()
		deltaxf = LPT.cic(self,PositionX,PositionY,PositionZ,self.XSize,GridSize_cic)*(GridSize_cic/self.GridSize)**3-1
		return deltaxf
	
	def PS_zeld_cic(self,RedShift,GridSize_cic=None,NofBins = 100,kmin=None, kmax=None):
		"""
		returns Pk , k in h Mpc^-1
		"""
		deltax = LPT.deltax_zeld_cic(self,RedShift,GridSize_cic)
		Pk,k = LPT.PS_calc(self,self.XSize,deltax,NofBins , kmin, kmax)
		return Pk,k
		
	
	def psi2_vector0(self):
		"""
		The 2 attached to variables indicates second order terms of the Lagrangian perturbation theory.
		"""
		k_x,k_y,k_z = initial_density_field.k_array(self)
		ksquare = initial_density_field.ksquare(self)
		TT = LPT.tidal_tensor(self,flag=1)
		Lapsi2 = -3./7.*self.Omega_matter**(-1/143.)*(TT[:,:,:,0,0]*TT[:,:,:,1,1]+TT[:,:,:,1,1]*TT[:,:,:,2,2]+TT[:,:,:,2,2]*TT[:,:,:,0,0]-TT[:,:,:,0,1]**2-TT[:,:,:,1,2]**2-TT[:,:,:,2,0]**2)
		psi2k_scalar = - np.fft.fftn(Lapsi2)/(ksquare+1e-15)
		psi2_vector_x0 = np.fft.ifftn(1j*k_x*psi2k_scalar) 
		psi2_vector_y0 = np.fft.ifftn(1j*k_y*psi2k_scalar) 
		psi2_vector_z0 = np.fft.ifftn(1j*k_z*psi2k_scalar) 
		return psi2_vector_x0 , psi2_vector_y0 , psi2_vector_z0
	
	
	def psi2_vector(self,RedShift,psi2_vector_x0=None,psi2_vector_y0=None,psi2_vector_z0=None):
		"""
		The 2 attached to variables indicates second order terms of the Lagrangian perturbation theory.
		"""
		psi2_vector0 = LPT.psi2_vector0(self)
		if psi2_vector_x0 is None:
			psi2_vector_x0 = psi2_vector0[0]
		if psi2_vector_y0 is None:
			psi2_vector_y0 = psi2_vector0[1]
		if psi2_vector_z0 is None:
			psi2_vector_z0 = psi2_vector0[2]
		Scalefactor = 1./(1.+RedShift)
		psi2_vector_x = psi2_vector_x0 * (lb.cosmology.GrowthFunctionAnalytic(self,Scalefactor)/lb.cosmology.GrowthFunctionAnalytic(self,self.ScaleFactor0))**2
		psi2_vector_y = psi2_vector_y0 * (lb.cosmology.GrowthFunctionAnalytic(self,Scalefactor)/lb.cosmology.GrowthFunctionAnalytic(self,self.ScaleFactor0))**2
		psi2_vector_z = psi2_vector_z0 * (lb.cosmology.GrowthFunctionAnalytic(self,Scalefactor)/lb.cosmology.GrowthFunctionAnalytic(self,self.ScaleFactor0))**2
		return psi2_vector_x,psi2_vector_y,psi2_vector_z
	
	def Position_2lpt(self,RedShift,psi_vector_x=None,psi_vector_y=None,psi_vector_z=None,psi2_vector_x=None,psi2_vector_y=None,psi2_vector_z=None):
		"""
		Gives the 2lpt position of the particles at the particular redshift. o/p is a 4 dimensional array whose first dimension gives x, y, z 2lpt position
		second third and fourth dimension corresponds to the i,j,k grid cell the particle originally belonged to before 
		2lpt was applied.
		If multiple redshifts are required it is faster to call the psi vectors first and then input them here.
		"""
		
		psi_vector = LPT.psi_vector(self,RedShift)
		if psi_vector_x is None:
			psi_vector_x = psi_vector[0]
		if psi_vector_y is None:
			psi_vector_y = psi_vector[1]
		if psi_vector_z is None:
			psi_vector_z = psi_vector[2]
		
		psi2_vector = LPT.psi2_vector(self,RedShift)
		if psi2_vector_x is None:
			psi2_vector_x = psi2_vector[0]
		if psi2_vector_y is None:
			psi2_vector_y = psi2_vector[1]
		if psi2_vector_z is None:
			psi2_vector_z = psi2_vector[2]
		
		i = range(0,self.GridSize)
		j = range(0,self.GridSize)
		k = range(0,self.GridSize)
		ii ,jj ,kk = np.meshgrid(i ,j ,k, indexing='ij' )
		Position = np.zeros([3,self.GridSize,self.GridSize,self.GridSize])
		Position[0,:,:,:] = np.mod((ii + 1/2.)* self.dx - psi_vector_x + psi2_vector_x,self.XSize)
		Position[1,:,:,:] = np.mod((jj + 1/2.)* self.dx - psi_vector_y + psi2_vector_y,self.XSize)
		Position[2,:,:,:] = np.mod((kk + 1/2.)* self.dx - psi_vector_z + psi2_vector_z,self.XSize)
		return Position


	def deltax_2lpt_cic(self,RedShift,GridSize_cic=None):
		if GridSize_cic is None:
			GridSize_cic = self.GridSize
		Position = LPT.Position_2lpt(self,RedShift)
		PositionX = Position[0,:,:,:].flatten()
		PositionY = Position[1,:,:,:].flatten()
		PositionZ = Position[2,:,:,:].flatten()
		deltaxf = LPT.cic(self,PositionX,PositionY,PositionZ,self.XSize,GridSize_cic)*(GridSize_cic/self.GridSize)**3-1
		return deltaxf
	
	def PS_2lpt_cic(self,RedShift,GridSize_cic=None,NofBins = 100,kmin=None, kmax=None):
		"""
		returns Pk , k in h Mpc^-1
		"""
		deltax = LPT.deltax_2lpt_cic(self,RedShift,GridSize_cic)
		Pk,k = LPT.PS_calc(self,self.XSize,deltax,NofBins , kmin, kmax)
		return Pk,k
