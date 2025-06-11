import numpy as np
import math
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import math
import library as lb
import mathtools as mt
import time
begin=time.time()
import gc
#############################################
##Scale factor of the Power Spectrum used####
RedShift=100.
ScaleFactor0 =1/(1+RedShift)
#############################################
GridSize=80	
dx=4
seed = 300000
dk=2*np.pi/(dx*GridSize)
XSize=GridSize*dx
print (XSize)
kspace = np.concatenate([range(0,int(GridSize/2)),range(-int(GridSize/2),0)])*dk
xspace = np.array(range(0,GridSize))*dx

k_x, k_y ,k_z = np.meshgrid(kspace,kspace,kspace,indexing='ij')
k_x_half, k_y_half ,k_z_half = np.meshgrid(kspace,kspace,kspace[0:int(GridSize/2)+1], sparse=True,indexing='ij')
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
deltakf[1:,1:,int(GridSize/2+1):] = deltakf[1:,1:,1:int(GridSize/2)][::-1,::-1,::-1].conjugate()
deltakf[0,0,0]=0
deltakf[0,0,int(GridSize/2)] = deltakf[0,0,int(GridSize/2)].real*np.sqrt(2)
deltakf[int(GridSize/2),0,0] = deltakf[int(GridSize/2),0,0].real*np.sqrt(2)
deltakf[0,int(GridSize/2),0] = deltakf[0,int(GridSize/2),0].real*np.sqrt(2)
deltakf[0,int(GridSize/2),int(GridSize/2)] = deltakf[0,int(GridSize/2),int(GridSize/2)].real*np.sqrt(2)
deltakf[int(GridSize/2),int(GridSize/2),0] = deltakf[int(GridSize/2),int(GridSize/2),0].real*np.sqrt(2)
deltakf[int(GridSize/2),0,int(GridSize/2)] = deltakf[int(GridSize/2),0,int(GridSize/2)].real*np.sqrt(2)
deltakf[int(GridSize/2),int(GridSize/2),int(GridSize/2)] = deltakf[int(GridSize/2),int(GridSize/2),int(GridSize/2)].real*np.sqrt(2)
deltakf[0,0,int(GridSize/2)+1:] = deltakf[0,0,1:int(GridSize/2)][::-1].conjugate()
deltakf[0,int(GridSize/2)+1:,0] = deltakf[0,1:int(GridSize/2),0][::-1].conjugate()
deltakf[int(GridSize/2)+1:,0,0] = deltakf[1:int(GridSize/2),0,0][::-1].conjugate()
deltakf[0,1:GridSize,int(GridSize/2)+1:] = deltakf[0,1:GridSize,1:int(GridSize/2)][::-1,::-1].conjugate()
deltakf[1:GridSize,0,int(GridSize/2)+1:] = deltakf[1:GridSize,0,1:int(GridSize/2)][::-1,::-1].conjugate()
deltakf[1:GridSize,int(GridSize/2)+1:,0] = deltakf[1:GridSize,1:int(GridSize/2),0][::-1,::-1].conjugate()
deltakf[1:GridSize,int(GridSize/2)+1:,int(GridSize/2)] = deltakf[1:GridSize,1:int(GridSize/2),int(GridSize/2)][::-1,::-1].conjugate()

deltax =np.fft.ifftn(deltakf)
deltax = GridSize**3 *deltax

##########Sanity cHECK ################
print ('Variance of  delta x' ,np.var(np.real(deltax)))
print ('Sum of Discrete Power Spectrum is ',np.sum(variance*2))
##################################

plt.imshow(np.real(deltax)[:,:, 3],cmap='binary')
plt.colorbar()
#~ plt.savefig('InitialDensity.pdf')
plt.show()
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

fig=plt.figure(figsize=(8,7))
iii = 0


Redshift=np.linspace(1,100,100)[::-1]
Scalefactor= 1/(1+Redshift)
print (Scalefactor.shape,'shape')
#~ psi_vector_x = np.outer(psi_vector_x0,Scalefactor).reshape(psi_vector_x0.shape[0],psi_vector_x0.shape[1],psi_vector_x0.shape[2],Scalefactor.shape[0])/lb.GrowthFunctionAnalytic(ScaleFactor0)
#~ psi_vector_y = np.outer(psi_vector_y0,Scalefactor).reshape(psi_vector_y0.shape[0],psi_vector_y0.shape[1],psi_vector_y0.shape[2],Scalefactor.shape[0])/lb.GrowthFunctionAnalytic(ScaleFactor0)
#~ psi_vector_z = np.outer(psi_vector_z0,Scalefactor).reshape(psi_vector_z0.shape[0],psi_vector_z0.shape[1],psi_vector_z0.shape[2],Scalefactor.shape[0])/lb.GrowthFunctionAnalytic(ScaleFactor0)
## this was a mistake I did earlier corrected below

psi_vector_x = np.outer(psi_vector_x0,lb.GrowthFunctionAnalytic(Scalefactor)).reshape(psi_vector_x0.shape[0],psi_vector_x0.shape[1],psi_vector_x0.shape[2],Scalefactor.shape[0])/lb.GrowthFunctionAnalytic(ScaleFactor0)
psi_vector_y = np.outer(psi_vector_y0,lb.GrowthFunctionAnalytic(Scalefactor)).reshape(psi_vector_y0.shape[0],psi_vector_y0.shape[1],psi_vector_y0.shape[2],Scalefactor.shape[0])/lb.GrowthFunctionAnalytic(ScaleFactor0)
psi_vector_z = np.outer(psi_vector_z0,lb.GrowthFunctionAnalytic(Scalefactor)).reshape(psi_vector_z0.shape[0],psi_vector_z0.shape[1],psi_vector_z0.shape[2],Scalefactor.shape[0])/lb.GrowthFunctionAnalytic(ScaleFactor0)
i = range(0,GridSize)
j = range(0,GridSize)
k = range(0,GridSize)
ii ,jj ,kk,hh= np.meshgrid(i ,j ,k,Scalefactor,indexing='ij')
Position = np.zeros([3,GridSize,GridSize,GridSize,Scalefactor.shape[0]])
Position[0,:,:,:,:] = np.mod(ii* dx - psi_vector_x.real,XSize)
Position[1,:,:,:,:] = np.mod(jj* dx - psi_vector_y.real,XSize)
Position[2,:,:,:,:] = np.mod(kk* dx - psi_vector_z.real,XSize)
print (np.amax(Position))
print ('saving done')





flag5=9
flag=0
flag2=8
flag3=8
####  flag=0   ------>  Make 2d projections at different redshifts
####  flag=1   ------>  Make 3d scatter plot at different redshifts
####  flag=2   ------>  Nearest Grid Point
####  flag=3   ------>  Compute the eigen values for tensor potential

# ~ if flag==0:
	# ~ for index in range(95,100):
		# ~ plt.gca().invert_yaxis()
		# ~ Xposition=Position[0,:,:,3,index].reshape((GridSize*GridSize))
		# ~ Yposition=Position[1,:,:,3,index].reshape((GridSize*GridSize))
		# ~ Zposition=Position[2,:,:,3,index].reshape((GridSize*GridSize))
		
		
		# ~ plt.scatter(Xposition,Yposition,s=0.09)
	
		
		# ~ st= '2dproj'+str(index)+'.png'
		# ~ stpdf= '2dproj'+str(index)+'.pdf'
		# ~ st_title='Box Size=' +'{0:.7f}'.format(dx*GridSize)+' Mpc/h' + '  RedShift=' +str(Redshift[index])[0:3]
		# ~ plt.title(st_title)
		# ~ plt.savefig(st,dpi=100,bbox_inches='tight', pad_inches=0)         ####saving figure takes the longest time##############
		# ~ plt.savefig(stpdf,dpi=100)         ####saving figure takes the longest time##############
		# ~ plt.clf()    ######This is very important for speed of the loop for some reason###############

	
	
	
	

for index in range(98,100):
	print (time.time())
	Xposition=Position[0,:,:,:,index].reshape((GridSize*GridSize*GridSize))
	Yposition=Position[1,:,:,:,index].reshape((GridSize*GridSize*GridSize))
	Zposition=Position[2,:,:,:,index].reshape((GridSize*GridSize*GridSize))
	print (time.time())
	ax = fig.add_subplot(1,1,1, projection='3d')
	a=ax.scatter(Xposition,Yposition,Zposition,s=0.005,color='purple')
	ax.view_init(elev=20., azim=30+index)
	# ~ ax._axis3don = False
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])
	ax.set_xlim3d(np.amin(Xposition),np.amax(Xposition))
	ax.set_ylim3d(np.amin(Yposition),np.amax(Yposition))
	ax.set_zlim3d(np.amin(Zposition),np.amax(Zposition))	
	# make the panes transparent
	ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	# make the grid lines transparent
	ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
	ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
	ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

	st= 'movie_save_big'+str(index)+'.png'
	print (time.time())
	st_title='Box Size=' +'{0:.1f}'.format(dx*GridSize)+' Mpc/h' + '  RedShift=' +str(Redshift[index])[0:3]
	# ~ plt.title(st_title)
	#plt.title(Large Volume')
	plt.savefig(st,dpi=100,bbox_inches='tight', pad_inches=0,transparent=True)         ####saving figure takes the longest time##############
	plt.clf()    ######This is very important for speed of the loop for some reason###############
		
sys.exit()
import numpy.ma as ma 

if flag2==0:
	########################## Nearest Grid Point ############################################
	index=1
	rho_c = 9.5e-27    ## in SI units
	OmegaMatter = 0.3
	Volume = dx**3
	Mass = rho_c *OmegaMatter *dx**3
	
	
	inew = np.floor(np.mod(Position[0,:,:,:,index] + dx/2,XSize)/dx)
	jnew = np.floor(np.mod(Position[1,:,:,:,index]  + dx/2,XSize)/dx)
	knew = np.floor(np.mod(Position[2,:,:,:,index]  + dx/2,XSize)/dx)
	BinNumber = inew*GridSize**2  + GridSize *jnew + knew     

	
	
	### note of flatten() / reshape- unravels in row major style order
	#Row-major order,
	#Address 	Access 		
	#0 			A[0][0] 	 
	#0 			A[0][0] 	 
	#1 			A[0][1] 	
	#2 			A[0][2] 	
	#3 			A[1][0] 	 
	#4 			A[1][1] 	 
	#5 			A[1][2] 	
	
	start = time.time()
	#~ FieldDensity,edge = np.histogram(BinNumber,bins=np.arange(0,GridSize*GridSize+1))
	print (time.time()-start)
	start = time.time()
	FieldDensity = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),minlength=GridSize**3)
	print ('and now' ,time.time()-start)

	FieldDensity = FieldDensity * rho_c *OmegaMatter
	FieldDensity = FieldDensity.reshape([GridSize,GridSize,GridSize])
	plt.clf()
	DELTAX = (FieldDensity - rho_c *OmegaMatter)/(rho_c *OmegaMatter)
	#~ plt.imshow(np.log10(DELTAX[:,:,3]),cmap='jet')
	#~ plt.colorbar()
	#~ plt.show()
	print ('there')
	######  Smoothening  ##########
	R=0.004  
	DELTAXk = np.fft.fftn(DELTAX)
	DELTAXsmooth = np.fft.ifftn(DELTAXk*lb.GaussianWk(np.sqrt(ksquare),R)).real
	#~ plt.imshow(np.log10(DELTAXsmooth[:,:,3]),cmap='jet')
	#~ plt.colorbar()
	#~ plt.show()
	#~ 
	
	
	
	


########################## Cloud in a Cell ############################################	
time1=time.time()
if flag3==0:  
	print ('start')
	
	rho_c = 9.5e-27    ## in SI units
	OmegaMatter = 0.3
	Volume = dx**3
	Mass = rho_c *OmegaMatter *dx**3
		
	#### Cell 0	
	
	inew = np.floor(Position[0,:,:,:,index]/dx)
	jnew = np.floor(Position[1,:,:,:,index]/dx)
	knew = np.floor(Position[2,:,:,:,index]/dx)
	print ('im typing here',np.amax(inew),np.amax(Position))
	#~ BinNumber = inew*GridSize  + GridSize**2 *jnew + knew
	BinNumber = inew*GridSize**2  + GridSize *jnew + knew
	delx = Position[0,:,:,:,index] - inew*dx
	dely = Position[1,:,:,:,index] - jnew*dx
	delz = Position[2,:,:,:,index] - knew*dx
	Weight = (dx-delx)*(dx-dely)*(dx-delz)/dx**3
	FieldDensity_CIC = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
	print ('First Cell done')
	
	padcount = GridSize**3-FieldDensity_CIC.shape[0] 
	FieldDensity_CIC = np.pad(FieldDensity_CIC,(0,padcount),'constant')

	
	
	
	
	
	#### Cell 1	
	BinNumber = np.mod(inew + 1,GridSize)*GridSize**2  + GridSize *jnew + knew
	Weight	   =  delx * (dx- dely) * (dx - delz) / dx**3
	FieldDensity_CIC1 = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
	padcount = GridSize**3-FieldDensity_CIC1.shape[0] 
	FieldDensity_CIC1 = np.pad(FieldDensity_CIC1,(0,padcount),'constant')
	FieldDensity_CIC = FieldDensity_CIC + FieldDensity_CIC1
	
	
	#### Cell 2	
	
	BinNumber = np.mod(inew + 1,GridSize)*GridSize**2  + GridSize *np.mod(jnew + 1,GridSize) + knew
	Weight	   = delx * dely * (dx - delz)/dx**3
	FieldDensity_CIC1 = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
	padcount = GridSize**3-FieldDensity_CIC1.shape[0] 
	
	
	FieldDensity_CIC1 = np.pad(FieldDensity_CIC1,(0,padcount),'constant')
	FieldDensity_CIC = FieldDensity_CIC + FieldDensity_CIC1
	
	#~ #### Cell 3	

	BinNumber = inew*GridSize**2  + GridSize *np.mod(jnew + 1,GridSize) + knew
	Weight	  = (dx - delx)*dely * (dx-delz)/dx**3
	FieldDensity_CIC1 = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
	padcount = GridSize**3-FieldDensity_CIC1.shape[0]
	FieldDensity_CIC1 = np.pad(FieldDensity_CIC1,(0,padcount),'constant')
	FieldDensity_CIC = FieldDensity_CIC + FieldDensity_CIC1
	
	#~ #### Cell 4	
	BinNumber = inew*GridSize**2  + GridSize *jnew+ np.mod(knew + 1,GridSize)
	Weight	  =  (dx-delx) * (dx-dely) * delz /dx**3
	FieldDensity_CIC1 = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
	padcount = GridSize**3-FieldDensity_CIC1.shape[0]
	FieldDensity_CIC1 = np.pad(FieldDensity_CIC1,(0,padcount),'constant')
	FieldDensity_CIC = FieldDensity_CIC + FieldDensity_CIC1
	
	#~ #### Cell 5	
	BinNumber = np.mod(inew + 1,GridSize) *GridSize**2  + GridSize *jnew + np.mod(knew + 1,GridSize)
	Weight	  = delx * (dx- dely) * delz /dx**3
	FieldDensity_CIC1 = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
	padcount = GridSize**3-FieldDensity_CIC1.shape[0]
	FieldDensity_CIC1 = np.pad(FieldDensity_CIC1,(0,padcount),'constant')
	FieldDensity_CIC = FieldDensity_CIC + FieldDensity_CIC1
	
	#~ #### Cell 6	
	BinNumber = np.mod(inew +1,GridSize) *GridSize**2  + GridSize *np.mod(jnew +1,GridSize) + np.mod(knew + 1,GridSize)
	Weight	  = delx*dely*delz/dx**3
	FieldDensity_CIC1 = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
	padcount = GridSize**3-FieldDensity_CIC1.shape[0]
	FieldDensity_CIC1 = np.pad(FieldDensity_CIC1,(0,padcount),'constant')
	FieldDensity_CIC = FieldDensity_CIC + FieldDensity_CIC1
	
	#~ #### Cell 7	
	BinNumber = inew*GridSize**2  + GridSize *np.mod(jnew + 1,GridSize) +np.mod(knew + 1,GridSize)
	Weight	  = (dx - delx)*dely*delz/dx**3
	FieldDensity_CIC1 = np.bincount((BinNumber.astype(int)).reshape(GridSize**3),Weight.reshape(GridSize**3))
	padcount = GridSize**3-FieldDensity_CIC1.shape[0]
	FieldDensity_CIC1 = np.pad(FieldDensity_CIC1,(0,padcount),'constant')
	FieldDensity_CIC = FieldDensity_CIC + FieldDensity_CIC1
	
	FieldDensity_CIC = FieldDensity_CIC * rho_c *OmegaMatter
	FieldDensity_CIC = FieldDensity_CIC.reshape([GridSize,GridSize,GridSize])
	DELTAX_CIC = (FieldDensity_CIC - rho_c *OmegaMatter)/(rho_c *OmegaMatter)
	
	
	
	print ('FieldDensityAverage=' , np.mean(FieldDensity_CIC)/rho_c)
	print ('Time Taken=', time.time()-time1)
	
	
	
	
	
##################Cloud in a Cell Loop ##########################################

#~ for b in range(0,GridSize):
	#~ for c in range(0,GridSize):
		#~ 
		
		
	










if flag5==0:
	plt.subplot(4,2,1)
	plt.imshow(np.real(deltax)[:,:,3],cmap='jet',aspect='auto')
	plt.colorbar()
	plt.subplot(4,2,2)
	plt.gca().invert_yaxis()
	plt.scatter(Yposition,Xposition,s=0.09)
	
	plt.subplot(4,2,3)
	plt.title('NGP')
	#~ plt.imshow((DELTAX[:,:,3]),cmap='jet',vmin=-1,vmax=np.amax(DELTAX[:,:,3]))
	plt.imshow((DELTAX[:,:,3]),cmap='jet',aspect='auto')
	plt.colorbar()
	
	log_DELTAX = np.log10(DELTAX)
	masked_log_DELTAX = ma.masked_invalid(log_DELTAX)
	
	plt.subplot(4,2,4)
	print (np.log10(DELTAX[:,:,3]))
	print ('hello')
	#plt.imshow(np.log10(DELTAX[:,:,3]),cmap='jet',vmin = np.amin(np.log10(DELTAXsmooth[:,:,3])),vmax=np.amax(np.log10(DELTAX[:,:,3])))
	plt.imshow(masked_log_DELTAX[:,:,3],cmap='jet',aspect='auto')
	plt.colorbar()
	plt.title('NGP log')
	
	
	plt.subplot(4,2,5)
	#~ plt.imshow((DELTAXsmooth[:,:,3]),cmap='jet',vmin=-1,vmax=np.amax(DELTAX[:,:,3]))
	plt.imshow((DELTAXsmooth[:,:,3]),cmap='jet',aspect='auto')
	plt.colorbar()
	plt.title('NGP smoothened')
	
	
	plt.subplot(4,2,6)
	#~ plt.imshow(np.log10(DELTAXsmooth[:,:,3]),cmap='jet' ,vmin = np.amin(np.log10(DELTAXsmooth[:,:,3])),vmax=np.amax(np.log10(DELTAXsmooth[:,:,3])))
	plt.imshow(np.log10(DELTAXsmooth[:,:,3]),cmap='jet',aspect='auto' )
	plt.title('NGP smoothened log')
	plt.colorbar()
	
	plt.subplot(4,2,7)
	plt.title('Cloud In Cell')
	#~ plt.imshow((DELTAX_CIC[:,:,3]),cmap='jet',vmin=-1,vmax=np.amax(DELTAX[:,:,3]))
	plt.imshow((DELTAX_CIC[:,:,3]),cmap='jet',aspect='auto')
	plt.colorbar()
	
	
	plt.subplot(4,2,8)
	plt.title('Cloud In Cell lOG')
	plt.imshow(np.log10(DELTAX_CIC[:,:,3]),cmap='jet',aspect='auto')
	plt.colorbar()
	plt.savefig('interpolation.pdf')
	plt.show()
	
	









	
	

######## Initialising Tensor Potential ####################################################

TidalTensor = np.zeros([GridSize**3,3,3])

TidalTensor[:,0,0] = (GridSize**3 * np.fft.ifftn( - psik_scalar * k_x * k_x ).real).reshape(GridSize**3)
TidalTensor[:,0,1] = (GridSize**3 * np.fft.ifftn( - psik_scalar * k_x * k_y ).real).reshape(GridSize**3)
TidalTensor[:,0,2] = (GridSize**3 * np.fft.ifftn( - psik_scalar * k_x * k_z ).real).reshape(GridSize**3)
TidalTensor[:,1,0] = (GridSize**3 * np.fft.ifftn( - psik_scalar * k_y * k_x ).real).reshape(GridSize**3)
TidalTensor[:,1,1] = (GridSize**3 * np.fft.ifftn( - psik_scalar * k_y * k_y ).real).reshape(GridSize**3)
TidalTensor[:,1,2] = (GridSize**3 * np.fft.ifftn( - psik_scalar * k_z * k_z ).real).reshape(GridSize**3)
TidalTensor[:,2,0] = (GridSize**3 * np.fft.ifftn( - psik_scalar * k_z * k_x ).real).reshape(GridSize**3)
TidalTensor[:,2,1] = (GridSize**3 * np.fft.ifftn( - psik_scalar * k_z * k_y ).real).reshape(GridSize**3)
TidalTensor[:,2,2] = (GridSize**3 * np.fft.ifftn( - psik_scalar * k_z * k_z ).real).reshape(GridSize**3)

eigenValue = np.linalg.eigvalsh(TidalTensor)


#############Web Classification#################################
########Void       =0 
########Filament   =1
########Sheet      =2
########Node       =3

WebClassicfication = np.zeros([GridSize**3])
#~ index_1 = (eigenValue[:,2]>0).astype(int)        
#~ 
#~ print index_1
               #~ 
#~ WebClassicfication[(eigenValue[index_1,1]<0).astype(int)   ] = 1
#~ print WebClassicfication
#~ sys.exit()
#~ index_2 = (eigenValue[index_1,1]>0)[0]
#~ WebClassicfication[eigenValue[index_2,1]<0] = 2
#~ index_3 = (eigenValue[index_2,1]>0)[0]
#~ WebClassicfication[eigenValue[index_3,1]<0] = 3

WebClassicfication[np.where((eigenValue[:,2]>0)&(eigenValue[:,1]<0))[0]] = 1
WebClassicfication[np.where((eigenValue[:,1]>0)&(eigenValue[:,0]<0))[0]] = 2
WebClassicfication[np.where((eigenValue[:,0]>0))[0]] = 3
WebClassicfication = WebClassicfication.reshape([GridSize,GridSize,GridSize])
WebClassicfication = np.flipud(WebClassicfication.transpose())
plt.pcolor(WebClassicfication[:,:,3],cmap='binary')
plt.colorbar()
plt.show()
sys.exit()



#################### Taking Gradient of DElta field ###################

DELTAK   = np.fft.fft2(DELTAX)
DELTA_Y= np.fft.ifft2(DELTAK)
DELTA_KX = DELTAK * k_x * 1j
DELTA_KY = DELTAK * k_y * 1j
DELTA_X  = np.fft.ifft2(DELTA_KX)
DELTA_Y  = np.fft.ifft2(DELTA_KY)
#~ print DELTA_Y

Position[0,:,:] = np.mod(Position[0,:,:] - DELTA_X.real,XSize)
Position[1,:,:] = np.mod(Position[1,:,:] - DELTA_Y.real,XSize)
Xposition=Position[0,:,:].reshape((GridSize*GridSize))
Yposition=Position[1,:,:].reshape((GridSize*GridSize))

plt.scatter(Xposition,Yposition,0.1)
string=str(k)+'.png'
plt.savefig(string)
plt.show()
