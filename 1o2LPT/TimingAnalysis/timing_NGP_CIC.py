import numpy as np
import math
import pylab
import matplotlib.pyplot as plt
import sys
import math
import library as lb
import time
begin=time.time()
#############################################
##Scale factor of the Power Spectrum used####
RedShift=100.
ScaleFactor0 =1/(1+RedShift)
#############################################
s=range(16,256,8)
TA= np.zeros([3,np.size(s)])
ID=0
for GridSize in s:
	
	
	dx=0.01
	
	dk=2*np.pi/(dx*GridSize)
	XSize=GridSize*dx
	kspace =np.concatenate([range(0,GridSize/2),range(-GridSize/2,0)])*dk
	xspace = np.array(range(0,GridSize))*dx
	
	k_x, k_y ,k_z = np.meshgrid(kspace,kspace,kspace)
	k_x_half, k_y_half ,k_z_half = np.meshgrid(kspace,kspace,kspace[0:GridSize/2+1], sparse=True)
	x, y ,z = np.meshgrid(xspace,xspace,xspace, sparse=True)	
	
	mean=0
	#~ np.random.seed(1000)
	#~ def P(k):
		#~ kc=.1
		#~ P=10000000*(dk)**3/(2*np.pi)**3*np.exp(-k/kc)*(k+1e-15)
		#~ return P
	
	
	
	#~ k_half_square  =(k_x_half**2+k_y_half**2+k_z_half**2)
	ksquare  =(k_x**2+k_y**2+k_z**2)
	tiny=1e-15
	#~ variance_half = (dk)**3/(2*np.pi)**3*lb.P(np.sqrt(k_half_square)+tiny,RedShift)/2
	variance = (dk)**3/(2*np.pi)**3*lb.P(np.sqrt(ksquare)+tiny        ,RedShift)/2
	
	
	
	###############                                                  ###############################
	############### Initialising delta k as random gaussian variable ###############################
	###############                                                  ###############################
	
	#### Using iRFFT
	#~ deltak = np.random.normal(mean,np.sqrt(variance_half), [GridSize,GridSize,GridSize/2+1]) + np.random.normal(mean,np.sqrt(variance_half), [GridSize,GridSize,GridSize/2+1])*1j
	#~ deltax_fast=np.fft.irfftn(deltak)
	#~ 
	
	
	
	start=time.time()
	##### Using ifft and taking care to make the ift real
	deltakf = np.random.normal(mean,np.sqrt(variance), [GridSize,GridSize,GridSize]) + np.random.normal(mean,np.sqrt(variance), [GridSize,GridSize,GridSize])*1j
	deltakf[1:,1:,GridSize/2+1:] = deltakf[1:,1:,1:GridSize/2][::-1,::-1,::-1].conjugate()
	deltakf[0,0,0]=0
	deltakf[0,0,GridSize/2] = deltakf[0,0,GridSize/2].real*np.sqrt(2)
	deltakf[GridSize/2,0,0] = deltakf[GridSize/2,0,0].real*np.sqrt(2)
	deltakf[0,GridSize/2,0] = deltakf[0,GridSize/2,0].real*np.sqrt(2)
	deltakf[0,GridSize/2,GridSize/2] = deltakf[0,GridSize/2,GridSize/2].real*np.sqrt(2)
	deltakf[GridSize/2,GridSize/2,0] = deltakf[GridSize/2,GridSize/2,0].real*np.sqrt(2)
	deltakf[GridSize/2,0,GridSize/2] = deltakf[GridSize/2,0,GridSize/2].real*np.sqrt(2)
	deltakf[GridSize/2,GridSize/2,GridSize/2] = deltakf[GridSize/2,GridSize/2,GridSize/2].real*np.sqrt(2)
	deltakf[0,0,GridSize/2+1:] = deltakf[0,0,1:GridSize/2][::-1].conjugate()
	deltakf[0,GridSize/2+1:,0] = deltakf[0,1:GridSize/2,0][::-1].conjugate()
	deltakf[GridSize/2+1:,0,0] = deltakf[1:GridSize/2,0,0][::-1].conjugate()
	deltakf[0,1:GridSize,GridSize/2+1:] = deltakf[0,1:GridSize,1:GridSize/2][::-1,::-1].conjugate()
	deltakf[1:GridSize,0,GridSize/2+1:] = deltakf[1:GridSize,0,1:GridSize/2][::-1,::-1].conjugate()
	deltakf[1:GridSize,GridSize/2+1:,0] = deltakf[1:GridSize,1:GridSize/2,0][::-1,::-1].conjugate()
	deltakf[1:GridSize,GridSize/2+1:,GridSize/2] = deltakf[1:GridSize,1:GridSize/2,GridSize/2][::-1,::-1].conjugate()
	
	deltax =np.fft.ifftn(deltakf)
	deltax = GridSize**3 *deltax
	
	
	
	#~ plt.imshow(np.real(deltax)[:,:,3],cmap='jet')
	#~ plt.colorbar()
	#~ plt.savefig('InitialDensity.pdf')
	#~ plt.show()
	#~ plt.imshow(np.real(deltax)[:,:,3],cmap='binary')
	#~ plt.colorbar()
	#~ plt.savefig('InitialDensityBW.pdf')
	#~ plt.show()
	
	
	psik_scalar = -deltakf/(ksquare + 1e-15)
	psik_scalar[0,0] = 0
	
	psik_vector_x = psik_scalar * k_x * 1j
	psik_vector_y = psik_scalar * k_y * 1j
	psik_vector_z = psik_scalar * k_z * 1j
	
	
	psi_vector_x0 = GridSize**3 *  np.fft.ifftn(psik_vector_x)
	psi_vector_y0 = GridSize**3 * np.fft.ifftn(psik_vector_y)
	psi_vector_z0 = GridSize**3 *  np.fft.ifftn(psik_vector_z)
	
	
	k=0
	
	#~ plt.gca().invert_yaxis()
	

	iii = 0
	
	
	Redshift=np.linspace(5,30,2)[::-1]
	Scalefactor= 1/(1+Redshift)
	#~ psi_vector_x = np.outer(psi_vector_x0,Scalefactor).reshape(psi_vector_x0.shape[0],psi_vector_x0.shape[1],psi_vector_x0.shape[2],Scalefactor.shape[0])/lb.GrowthFunctionAnalytic(ScaleFactor0)
	#~ psi_vector_y = np.outer(psi_vector_y0,Scalefactor).reshape(psi_vector_y0.shape[0],psi_vector_y0.shape[1],psi_vector_y0.shape[2],Scalefactor.shape[0])/lb.GrowthFunctionAnalytic(ScaleFactor0)
	#~ psi_vector_z = np.outer(psi_vector_z0,Scalefactor).reshape(psi_vector_z0.shape[0],psi_vector_z0.shape[1],psi_vector_z0.shape[2],Scalefactor.shape[0])/lb.GrowthFunctionAnalytic(ScaleFactor0)
	psi_vector_x = np.outer(psi_vector_x0,lb.GrowthFunctionAnalytic(Scalefactor)).reshape(psi_vector_x0.shape[0],psi_vector_x0.shape[1],psi_vector_x0.shape[2],Scalefactor.shape[0])/lb.GrowthFunctionAnalytic(ScaleFactor0)
	psi_vector_y = np.outer(psi_vector_y0,lb.GrowthFunctionAnalytic(Scalefactor)).reshape(psi_vector_y0.shape[0],psi_vector_y0.shape[1],psi_vector_y0.shape[2],Scalefactor.shape[0])/lb.GrowthFunctionAnalytic(ScaleFactor0)
	psi_vector_z = np.outer(psi_vector_z0,lb.GrowthFunctionAnalytic(Scalefactor)).reshape(psi_vector_z0.shape[0],psi_vector_z0.shape[1],psi_vector_z0.shape[2],Scalefactor.shape[0])/lb.GrowthFunctionAnalytic(ScaleFactor0)
	i = range(0,GridSize)
	j = range(0,GridSize)
	k = range(0,GridSize)
	ii ,jj ,kk,hh= np.meshgrid(i ,j ,k,Scalefactor)
	Position = np.zeros([3,GridSize,GridSize,GridSize,Scalefactor.shape[0]])
	Position[0,:,:,:,:] = np.mod(ii* dx - psi_vector_x.real,XSize)
	Position[1,:,:,:,:] = np.mod(jj* dx - psi_vector_y.real,XSize)
	Position[2,:,:,:,:] = np.mod(kk* dx - psi_vector_z.real,XSize)
	print 'saving done'
	
	
	
	
	
	
	flag=1
	flag2=0
	flag3=0
	####  flag=0   ------>  Make 2d projections at different redshifts
	####  flag=1   ------>  Make 3d scatter plot at different redshifts
	####  flag=2   ------>  Nearest Grid Point
	####  flag=3   ------>  Compute the eigen values for tensor potential
	
	
		
		
	
	import numpy.ma as ma 
	
	if flag2==0:
		########################## Nearest Grid Point ############################################
		ngp_start = time.time()
		index=1
		rho_c = 9.5e-27    ## in SI units
		OmegaMatter = 0.3
		Volume = dx**3
		Mass = rho_c *OmegaMatter *dx**3
		
		
		inew = np.floor(np.mod(Position[0,:,:,:,index] + dx/2,XSize)/dx)
		jnew = np.floor(np.mod(Position[1,:,:,:,index]  + dx/2,XSize)/dx)
		knew = np.floor(np.mod(Position[2,:,:,:,index]  + dx/2,XSize)/dx)
		BinNumber = inew*GridSize  + GridSize**2 *jnew + knew      #### Throughout Cartesian indexing is used. Hence inew gets index 1 and jnew gets index 0 
	
		
		
		### note of flatten() / reshape- unravels in row major style order
		#Row-major order,
		#Address 	Access 		
		#0 			A[0][0] 	 
		#1 			A[0][1] 	
		#2 			A[0][2] 	
		#3 			A[1][0] 	 
		#4 			A[1][1] 	 
		#5 			A[1][2] 	
		
		#~ FieldDensity,edge = np.histogram(BinNumber,bins=np.arange(0,GridSize*GridSize+1))
		FieldDensity = np.bincount((BinNumber.astype(int)).reshape(GridSize**3))
		
	
		padcount = GridSize**3-FieldDensity.shape[0] 
		FieldDensity = np.pad(FieldDensity,(0,padcount),'constant')
		FieldDensity = FieldDensity * rho_c *OmegaMatter
		FieldDensity = FieldDensity.reshape([GridSize,GridSize,GridSize])
		plt.clf()
		DELTAX = (FieldDensity - rho_c *OmegaMatter)/(rho_c *OmegaMatter)
		#~ plt.imshow(np.log10(DELTAX[:,:,3]),cmap='jet')
		#~ plt.colorbar()
		#~ plt.show()
		print 'something needs to be printed ', ID
		TA[0,ID]=time.time()-ngp_start
		######  Smoothening  ##########
		
		R=0.004  
		DELTAXk = np.fft.fft(DELTAX)
		DELTAXsmooth = np.fft.ifft(DELTAXk*lb.GaussianWk(np.sqrt(ksquare),R)).real
		#~ plt.imshow(np.log10(DELTAXsmooth[:,:,3]),cmap='jet')
		#~ plt.colorbar()
		#~ plt.show()
		#~ 
		TA[1,ID] = time.time() - ngp_start
		
		
		
	
	
	########################## Cloud in a Cell ############################################	
	cic_time=time.time()
	if flag3==0:  
		
		rho_c = 9.5e-27    ## in SI units
		OmegaMatter = 0.3
		Volume = dx**3
		Mass = rho_c *OmegaMatter *dx**3
			
		#### Cell 0	
		
		inew = np.floor(Position[0,:,:,:,index]/dx)
		jnew = np.floor(Position[1,:,:,:,index]/dx)
		knew = np.floor(Position[2,:,:,:,index]/dx)
		
		BinNumber = inew*GridSize  + GridSize**2 *jnew + knew
		#~ BinNumber = inew*GridSize**2  + GridSize *jnew + knew
		delx = Position[0,:,:,:,index] - inew*dx
		dely = Position[1,:,:,:,index] - jnew*dx
		delz = Position[2,:,:,:,index] - knew*dx
		Weight = (dx-delx)*(dx-dely)*(dx-dely)/dx**3
		FieldDensity_CIC = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
		print 'First Cell done'
		
		padcount = GridSize**3-FieldDensity_CIC.shape[0] 
		FieldDensity_CIC = np.pad(FieldDensity_CIC,(0,padcount),'constant')
	
		
		
		
		
		
		#### Cell 1	
		inew = np.mod(inew + 1,GridSize)
		jnew = jnew 
		knew = knew
		BinNumber = inew*GridSize  + GridSize**2 *jnew + knew
		#~ BinNumber1 = inew1*GridSize**2  + GridSize *jnew1 + knew1
		Weight	   =  delx * (dx- dely) * (dx - delz) / dx**3
		FieldDensity_CIC1 = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
		padcount = GridSize**3-FieldDensity_CIC1.shape[0] 
		FieldDensity_CIC1 = np.pad(FieldDensity_CIC1,(0,padcount),'constant')
		FieldDensity_CIC = FieldDensity_CIC + FieldDensity_CIC1
		
		
		#### Cell 2	
		inew = np.mod(inew + 1,GridSize)
		jnew = np.mod(jnew + 1,GridSize)
		knew = knew
		BinNumber = inew*GridSize  + GridSize**2 *jnew + knew
		#~ BinNumber2 = inew2*GridSize**2  + GridSize *jnew2 + knew2
		Weight	   = delx * dely * (dx - delz)/dx**3
		FieldDensity_CIC1 = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
		padcount = GridSize**3-FieldDensity_CIC1.shape[0] 
		
		
		FieldDensity_CIC1 = np.pad(FieldDensity_CIC1,(0,padcount),'constant')
		FieldDensity_CIC = FieldDensity_CIC + FieldDensity_CIC1
		
		#~ #### Cell 3	
		inew = inew 
		jnew = np.mod(jnew + 1,GridSize)
		knew = knew
		BinNumber = inew*GridSize  + GridSize**2 *jnew + knew
		#~ BinNumber3 = inew3*GridSize**2  + GridSize *jnew3 + knew3
		Weight	  = (dx - delx)*dely * (dx-delz)/dx**3
		FieldDensity_CIC1 = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
		padcount = GridSize**3-FieldDensity_CIC1.shape[0]
		FieldDensity_CIC1 = np.pad(FieldDensity_CIC1,(0,padcount),'constant')
		FieldDensity_CIC = FieldDensity_CIC + FieldDensity_CIC1
		
		#~ #### Cell 4	
		inew = inew 
		jnew = jnew 
		knew = np.mod(knew + 1,GridSize)
		BinNumber = inew*GridSize  + GridSize**2 *jnew + knew
		#~ BinNumber4 = inew4*GridSize**2  + GridSize *jnew4 + knew4
		Weight	  =  (dx-delx) * (dx-dely) * delz /dx**3
		FieldDensity_CIC1 = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
		padcount = GridSize**3-FieldDensity_CIC1.shape[0]
		FieldDensity_CIC1 = np.pad(FieldDensity_CIC1,(0,padcount),'constant')
		FieldDensity_CIC = FieldDensity_CIC + FieldDensity_CIC1
		
		#~ #### Cell 5	
		inew = np.mod(inew + 1,GridSize) 
		jnew = jnew 
		knew = np.mod(knew + 1,GridSize)
		BinNumber = inew*GridSize  + GridSize**2 *jnew + knew
		#~ BinNumber5 = inew5*GridSize**2  + GridSize *jnew5 + knew5
		Weight	  = delx * (dx- dely) * delz /dx**3
		FieldDensity_CIC1 = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
		padcount = GridSize**3-FieldDensity_CIC1.shape[0]
		FieldDensity_CIC1 = np.pad(FieldDensity_CIC1,(0,padcount),'constant')
		FieldDensity_CIC = FieldDensity_CIC + FieldDensity_CIC1
		
		#~ #### Cell 6	
		inew = np.mod(inew +1,GridSize) 
		jnew = np.mod(jnew +1,GridSize)
		knew = np.mod(knew + 1,GridSize)
		#~ BinNumber6 = inew6*GridSize  + GridSize**2 *jnew6 + knew6
		BinNumber = inew*GridSize**2  + GridSize *jnew + knew
		Weight	  = delx*dely*delz/dx**3
		FieldDensity_CIC1 = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
		padcount = GridSize**3-FieldDensity_CIC1.shape[0]
		FieldDensity_CIC1 = np.pad(FieldDensity_CIC1,(0,padcount),'constant')
		FieldDensity_CIC = FieldDensity_CIC + FieldDensity_CIC1
		
		#~ #### Cell 7	
		inew = inew
		jnew = np.mod(jnew + 1,GridSize)
		knew = np.mod(knew + 1,GridSize)
		BinNumber = inew*GridSize  + GridSize**2 *jnew + knew
		#~ BinNumber7 = inew7*GridSize**2  + GridSize *jnew7 + knew7
		Weight	  = (dx - delx)*dely*delz/dx**3
		FieldDensity_CIC1 = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
		padcount = GridSize**3-FieldDensity_CIC1.shape[0]
		FieldDensity_CIC1 = np.pad(FieldDensity_CIC1,(0,padcount),'constant')
		FieldDensity_CIC = FieldDensity_CIC + FieldDensity_CIC1
		
		FieldDensity_CIC = FieldDensity_CIC * rho_c *OmegaMatter
		FieldDensity_CIC = FieldDensity_CIC.reshape([GridSize,GridSize,GridSize])
		DELTAX_CIC = (FieldDensity_CIC - rho_c *OmegaMatter)/(rho_c *OmegaMatter)
		TA[2,ID]=time.time()-cic_time
		
		ID=ID+1
np.savetxt('TIME.txt',np.transpose(TA))
	
	
	

		
	













	
	
