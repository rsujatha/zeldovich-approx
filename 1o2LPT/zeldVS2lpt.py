import numpy as np
import lpt as l
import sys
import matplotlib.pyplot as plt
import time
start =time.time()
XSize=128
i=0
fig=plt.figure(figsize=(15,7), dpi= 100, facecolor='w', edgecolor='k')
a = l.LPT(GridSize=128, XSize = XSize,Seed = 300000)
psi_vector_x0, psi_vector_y0, psi_vector_z0 = a.psi_vector0()
psi2_vector_x0 , psi2_vector_y0 , psi2_vector_z0 = a.psi2_vector0()
RS = np.logspace(0,2,100)[::-1]
for RedShift in RS:
	
	psi_vector_x, psi_vector_y, psi_vector_z = a.psi_vector(RedShift,psi_vector_x0=psi_vector_x0,psi_vector_y0=psi_vector_y0,psi_vector_z0=psi_vector_z0)
	psi2_vector_x, psi2_vector_y, psi2_vector_z = a.psi2_vector(RedShift,psi2_vector_x0=psi2_vector_x0,psi2_vector_y0=psi2_vector_y0,psi2_vector_z0=psi2_vector_z0)
	
	position2lpt=a.Position_2lpt(RedShift,psi_vector_x=psi_vector_x, psi_vector_y=psi_vector_y, psi_vector_z=psi_vector_z,psi2_vector_x=psi2_vector_x, psi2_vector_y=psi2_vector_y, psi2_vector_z=psi2_vector_z)
	ind = np.where(position2lpt[2,:,:,:].reshape([128**3])<XSize/10.)
	X=position2lpt[0,:,:,:].reshape([128**3])
	X=X[ind]
	Y=position2lpt[1,:,:,:].reshape([128**3])
	Y=Y[ind]
	
	plt.subplot(1,2,1)
	plt.plot(X,Y,'.',markersize=0.1,alpha=1)
	plt.text(X.min(), Y.max(), ' RedShift = '+str(RedShift)[0:3] , color='black',fontsize=10)

	plt.title('2lpt')
	plt.xlabel('Mpc/h')
	plt.ylabel('Mpc/h')
	
	positionzeld=a.Position_zeld(RedShift,psi_vector_x=psi_vector_x, psi_vector_y=psi_vector_y, psi_vector_z=psi_vector_z)
	ind = np.where(positionzeld[2,:,:,:].reshape([128**3])<XSize/10.)
	X=positionzeld[0,:,:,:].reshape([128**3])
	X=X[ind]
	Y=positionzeld[1,:,:,:].reshape([128**3])
	Y=Y[ind]
	
	plt.subplot(1,2,2)
	plt.plot(X,Y,'.',markersize=0.1,alpha=1)
	plt.title ('zeldovich')
	plt.xlabel('Mpc/h')
	plt.ylabel('Mpc/h')
	plt.text(X.min(), Y.max(), ' RedShift = '+str(RedShift)[0:3] , color='black',fontsize=10)

	plt.savefig('zeldVs2lpt_proper128'+str(i)+'.png',dpi=100,bbox_inches='tight', pad_inches=0)
	i+=1
	plt.clf()
plt.show()
print 'total time taken is ',time.time()-start
