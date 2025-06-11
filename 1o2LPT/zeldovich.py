###Class containing most of the codes required for zeldovich approximation

from __future__ import division
import numpy as np
import library as lb

		
class initial_density_field(object):
	
	def __init__(self,GridSize=128, XSize = 1.28,Seed = 300000):
		self.RedShift0 = 100.
		self.ScaleFactor0 = 1./(1+self.RedShift0)
		self.GridSize = GridSize
		self.XSize =  XSize
		self.dx = XSize/float(GridSize)
		self.seed = Seed
		self.dk=2*np.pi/(self.dx*self.GridSize)
		
	def x_range(self):
		xspace = np.array(range(0,self.GridSize))*self.dx
		return xspace	
		
	def k_array(self):
		kspace = np.concatenate([range(0,int(self.GridSize/2)),range(-int(self.GridSize/2),0)])*self.dk	
		k_x, k_y ,k_z = np.meshgrid(kspace,kspace,kspace[0:int(self.GridSize/2)+1], indexing='ij')
		return k_x, k_y, k_z
		
	def ksquare(self):
		kx, ky, kz = initial_density_field.k_array(self)
		ksquare = kx**2 + ky**2 + kz**2
		return ksquare

	def var(self,tiny=1e-15):
		ksquare  = initial_density_field.ksquare(self)
		return (self.dk)**3/(2*np.pi)**3*lb.P(np.sqrt(ksquare)+tiny,self.RedShift0)/2
		
	def initial_deltak(self):
		mean=0
		np.random.seed(self.seed)
		variance = initial_density_field.var(self)
		return np.random.normal(mean,np.sqrt(variance), [self.GridSize,self.GridSize,int(self.GridSize/2)+1]) + np.random.normal(mean,np.sqrt(variance), [self.GridSize,self.GridSize,int(self.GridSize/2)+1])*1j
	
	def initial_deltax(self):
		deltak = initial_density_field.initial_deltak(self)
		return self.GridSize**3 *np.fft.irfftn(deltak)


class zeldovich(initial_density_field):
	def __init__(self,GridSize=128, XSize = 1.28,Seed = 300000):
		super(zeldovich,self).__init__(GridSize=GridSize, XSize = XSize,Seed = Seed)
	def psik_scalar(self):
		deltak = initial_density_field.initial_deltak(self)
		ksquare = initial_density_field.ksquare(self)
		psik_scalar = -deltak/(ksquare + 1e-15)
		psik_scalar[0,0] = 0
		return psik_scalar
	
	
	def tidal_tensor(self,psik_scalar,k_x,k_y,k_z):
		TidalTensor = np.zeros([self.GridSize**3,3,3])
		TidalTensor[:,0,0] = (self.GridSize**3 * np.fft.irfftn( - psik_scalar * k_x * k_x )).reshape([self.GridSize**3])
		TidalTensor[:,0,1] = (self.GridSize**3 * np.fft.irfftn( - psik_scalar * k_x * k_y )).reshape([self.GridSize**3])
		TidalTensor[:,0,2] = (self.GridSize**3 * np.fft.irfftn( - psik_scalar * k_x * k_z )).reshape([self.GridSize**3])
		TidalTensor[:,1,0] = (self.GridSize**3 * np.fft.irfftn( - psik_scalar * k_y * k_x )).reshape([self.GridSize**3])
		TidalTensor[:,1,1] = (self.GridSize**3 * np.fft.irfftn( - psik_scalar * k_y * k_y )).reshape([self.GridSize**3])
		TidalTensor[:,1,2] = (self.GridSize**3 * np.fft.irfftn( - psik_scalar * k_z * k_z )).reshape([self.GridSize**3])
		TidalTensor[:,2,0] = (self.GridSize**3 * np.fft.irfftn( - psik_scalar * k_z * k_x )).reshape([self.GridSize**3])
		TidalTensor[:,2,1] = (self.GridSize**3 * np.fft.irfftn( - psik_scalar * k_z * k_y )).reshape([self.GridSize**3])
		TidalTensor[:,2,2] = (self.GridSize**3 * np.fft.irfftn( - psik_scalar * k_z * k_z )).reshape([self.GridSize**3])
		eigenValue = np.linalg.eigvalsh(TidalTensor)     ####returns eigen values in ascending order
		return TidalTensor,eigenValue
		
	def deltax_final(self,eigenValue,RedShift): 
		Scalefactor = 1./(1.+RedShift)
		D = lb.GrowthFunctionAnalytic(Scalefactor)
		deltax_final = 1./((1.-D*eigenValue[:,0])*(1.-D*eigenValue[:,1])*(1.-D*eigenValue[:,2]))-1.
		return deltax_final
		 
	def web_classification(self,psik_scalar,kx,ky,kz):
		
		#############Web Classification#################################
		########Void       =0 
		########Sheet   =1
		########Filament      =2
		########Node       =3
		
		WebClassification = np.zeros([self.GridSize**3])
		TidalTensor,eigenValue = zeldovich.tidal_tensor(self,psik_scalar,kx,ky,kz)
		WebClassification[np.where((eigenValue[:,2]>0)&(eigenValue[:,1]<0))[0]] = 1       
		WebClassification[np.where((eigenValue[:,1]>0)&(eigenValue[:,0]<0))[0]] = 2
		WebClassification[np.where((eigenValue[:,0]>0))[0]] = 3
		WebClassification = WebClassification.reshape([self.GridSize,self.GridSize,self.GridSize])
		return WebClassification
	
	def psik_vector(self):
		k_x,k_y,k_z = initial_density_field.k_array(self)
		psik_vector_x = zeldovich.psik_scalar(self) * k_x * 1j
		psik_vector_y = zeldovich.psik_scalar(self) * k_y * 1j
		psik_vector_z = zeldovich.psik_scalar(self) * k_z * 1j
		return psik_vector_x, psik_vector_y, psik_vector_z
		
	def psi_vector0(self):
		psik_vector_x, psik_vector_y, psik_vector_z = zeldovich.psik_vector(self)
		psi_vector_x0 = self.GridSize**3 *  np.fft.irfftn(psik_vector_x)
		psi_vector_y0 = self.GridSize**3 *  np.fft.irfftn(psik_vector_y)
		psi_vector_z0 = self.GridSize**3 *  np.fft.irfftn(psik_vector_z)
		return psi_vector_x0, psi_vector_y0, psi_vector_z0
		
	def psi_vector(self,RedShift):
		Scalefactor = 1./(1+RedShift)
		psi_vector_x0,psi_vector_y0,psi_vector_z0 = zeldovich.psi_vector0(self)
		psi_vector_x = psi_vector_x0*lb.GrowthFunctionAnalytic(Scalefactor)/lb.GrowthFunctionAnalytic(self.ScaleFactor0)
		psi_vector_y = psi_vector_y0*lb.GrowthFunctionAnalytic(Scalefactor)/lb.GrowthFunctionAnalytic(self.ScaleFactor0)
		psi_vector_z = psi_vector_z0*lb.GrowthFunctionAnalytic(Scalefactor)/lb.GrowthFunctionAnalytic(self.ScaleFactor0)
		return psi_vector_x, psi_vector_y, psi_vector_z

	def psi_vector_multiple(self,RedShift,psi_vector_x0,psi_vector_y0,psi_vector_z0):
		Scalefactor = 1./(1+RedShift)
		psi_vector_x = psi_vector_x0*lb.GrowthFunctionAnalytic(Scalefactor)/lb.GrowthFunctionAnalytic(self.ScaleFactor0)
		psi_vector_y = psi_vector_y0*lb.GrowthFunctionAnalytic(Scalefactor)/lb.GrowthFunctionAnalytic(self.ScaleFactor0)
		psi_vector_z = psi_vector_z0*lb.GrowthFunctionAnalytic(Scalefactor)/lb.GrowthFunctionAnalytic(self.ScaleFactor0)
		return psi_vector_x, psi_vector_y, psi_vector_z

	
	def Position(self,RedShift):
		"""
		o/p is a 4 dimensional array whose first dimension gives x, y, z position
		second third and fourth dimension corresponds to the i,j,k grid cell the particle originally belonged to before 
		zeldovich approximation was applied.
		
		"""
		i = range(0,self.GridSize)
		j = range(0,self.GridSize)
		k = range(0,self.GridSize)
		ii ,jj ,kk = np.meshgrid(i ,j ,k, indexing='ij' )
		Position = np.zeros([3,self.GridSize,self.GridSize,self.GridSize])
		psi_vector_x ,psi_vector_y, psi_vector_z = zeldovich.psi_vector(self,RedShift)
		Position[0,:,:,:] = np.mod(ii* self.dx - psi_vector_x,self.XSize)
		Position[1,:,:,:] = np.mod(jj* self.dx - psi_vector_y,self.XSize)
		Position[2,:,:,:] = np.mod(kk* self.dx - psi_vector_z,self.XSize)
		return Position
		
	def Position_multiple(self,psi_vector_x,psi_vector_y,psi_vector_z,RedShift):
		"""
		Use this in preference to Position function when dealing with multiple redshift of same instance.
		
		o/p is a 4 dimensional array whose first dimension gives x, y, z position
		second third and fourth dimension corresponds to the i,j,k grid cell the particle originally belonged to before 
		zeldovich approximation was applied.
		
		"""
		i = range(0,self.GridSize)
		j = range(0,self.GridSize)
		k = range(0,self.GridSize)
		ii ,jj ,kk = np.meshgrid(i ,j ,k, indexing='ij' )
		Position = np.zeros([3,self.GridSize,self.GridSize,self.GridSize])
		Position[0,:,:,:] = np.mod(ii* self.dx - psi_vector_x,self.XSize)
		Position[1,:,:,:] = np.mod(jj* self.dx - psi_vector_y,self.XSize)
		Position[2,:,:,:] = np.mod(kk* self.dx - psi_vector_z,self.XSize)
		return Position

class pmInterpolation(lb.cosmology):
	def __init__(self):
		lb.cosmology.__init__(self)
		
	def Counts(self,BinNumber,GridSize,Weight):
		if Weight is not None:
			Weightn = Weight.flatten(order='C')
		else: Weightn = Weight
		Counts = np.bincount((BinNumber.astype(int)).flatten(order='C'),Weightn)
		padcount = GridSize**3-Counts.shape[0] 
		Counts = np.pad(Counts,(0,padcount),'constant')
		return Counts.reshape([GridSize,GridSize,GridSize])
		
	def Deltax(self,TotalCounts):
		return TotalCounts-1 
		
	def FieldDensity(self,TotalCounts):
		return TotalCounts * self.rho_c *Omega_matter
		
	def ksquare(self,GridSize,LBox):
		dk = 2*np.pi/(LBox)
		kspace = np.concatenate([range(0,GridSize/2),range(-GridSize/2,0)])*dk	
		k_x, k_y ,k_z = np.meshgrid(kspace,kspace,kspace, indexing='ij')
		ksquare = k_x**2 + k_y**2 + k_z**2
		return ksquare
			
	def ngp(self,PositionX,PositionY,PositionZ,Lbox,GridSize):
		dx = Lbox / GridSize
		inew = np.floor(np.mod(PositionX + dx/2,Lbox)/dx)
		jnew = np.floor(np.mod(PositionY + dx/2,Lbox)/dx)
		knew = np.floor(np.mod(PositionZ + dx/2,Lbox)/dx)
		BinNumber = inew*GridSize**2 + GridSize *jnew + knew  
		return pmInterpolation.Counts(self,BinNumber,GridSize,Weight=None)
		
	def ngp_smooth(self,PositionX,PositionY,PositionZ,Lbox,GridSize,R):
		ksquare = pmInterpolation.ksquare(self,GridSize,Lbox)
		TotalCounts = pmInterpolation.ngp(self,PositionX,PositionY,PositionZ,Lbox,GridSize)     
		TotalCountsk = np.fft.fftn(TotalCounts)	
		TotalCounts_smooth = np.fft.ifftn(TotalCountsk*lb.GaussianWk(np.sqrt(ksquare),R)).real
		return TotalCounts_smooth
		
	def cic(self,PositionX,PositionY,PositionZ,Lbox,GridSize):
		dx = Lbox / GridSize
		#### Cell 0	
		inew = np.mod(np.floor(PositionX/dx),GridSize)
		jnew = np.mod(np.floor(PositionY/dx),GridSize)
		knew = np.mod(np.floor(PositionZ/dx),GridSize)
		BinNumber = inew*GridSize**2  + GridSize *jnew + knew
		delx = PositionX - inew*dx
		dely = PositionY - jnew*dx
		delz = PositionZ - knew*dx
		Weight = (dx-delx)*(dx-dely)*(dx-delz)/dx**3
		TotalCounts = pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
		
		#### Cell 1	
		BinNumber = np.mod(inew + 1,GridSize)*GridSize**2  + GridSize *jnew + knew
		Weight	   = delx * (dx - dely) * (dx - delz)/dx**3
		TotalCounts += pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
		
		
		#### Cell 2	
		BinNumber = np.mod(inew + 1,GridSize)*GridSize**2  + GridSize *np.mod(jnew + 1,GridSize) + knew
		Weight	   = delx * dely * (dx - delz)/dx**3
		TotalCounts += pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
	 
		#### Cell 3	
		BinNumber = inew*GridSize**2  + GridSize *np.mod(jnew + 1,GridSize) + knew
		Weight	  = (dx - delx)*dely * (dx-delz)/dx**3
		TotalCounts += pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
 
		#~ #### Cell 4	
		BinNumber = inew*GridSize**2  + GridSize *jnew + np.mod(knew + 1,GridSize)
		Weight	  =  (dx-delx) * (dx-dely) * delz /dx**3
		TotalCounts += pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
 
 
		#~ #### Cell 5	
		BinNumber = np.mod(inew + 1,GridSize)*GridSize**2  + GridSize *jnew + np.mod(knew + 1,GridSize)
		Weight	  = delx * (dx- dely) * delz /dx**3
		TotalCounts += pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
 
		#### Cell 6	
		BinNumber = np.mod(inew +1,GridSize)*GridSize**2  + GridSize *np.mod(jnew +1,GridSize) + np.mod(knew + 1,GridSize)
		Weight	  = delx*dely*delz/dx**3
		TotalCounts += pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
 
		#~ #### Cell 7	
		BinNumber = inew*GridSize**2  + GridSize *np.mod(jnew + 1,GridSize) + np.mod(knew + 1,GridSize)
		Weight	  = (dx - delx)*dely*delz/dx**3
		TotalCounts += pmInterpolation.Counts(self,BinNumber,GridSize,Weight)
		return TotalCounts
		

	def cicl(self,PositionX,PositionY,PositionZ,Lbox,GridSize):
		dx = Lbox / GridSize
		Counts = np.zeros([GridSize,GridSize,GridSize])
		for i in range(0,PositionX.shape[0]):
			inew  = int(np.mod(np.floor(PositionX[i]/dx),GridSize))
			inew1 = int(np.mod(inew + 1,GridSize))
			jnew  = int(np.mod(np.floor(PositionY[i]/dx),GridSize))
			jnew1 = int(np.mod(jnew + 1,GridSize))
			knew  = int(np.mod(np.floor(PositionZ[i]/dx),GridSize))
			knew1 = int(np.mod(knew + 1,GridSize))
			delx  = PositionX[i] - inew*dx
			dely  = PositionY[i] - jnew*dx
			delz  = PositionZ[i] - knew*dx
			
			Counts[inew,jnew,knew] += (dx-delx)*(dx-dely)*(dx-delz)/dx**3
			Counts[inew1,jnew,knew] += delx * (dx- dely) * (dx - delz) / dx**3 
			Counts[inew1,jnew1,knew] +=delx * dely * (dx - delz)/dx**3
			Counts[inew,jnew1,knew] += (dx - delx)*dely * (dx-delz)/dx**3
			Counts[inew,jnew,knew1] += (dx-delx) * (dx-dely) * delz /dx**3
			Counts[inew1,jnew,knew1] +=  delx * (dx- dely) * delz /dx**3
			Counts[inew1,jnew1,knew1] += delx*dely*delz/dx**3
			Counts[inew,jnew1,knew1] += (dx - delx)*dely*delz/dx**3
		return Counts
		
