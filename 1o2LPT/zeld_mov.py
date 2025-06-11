import zeldovich as z
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import gc


####### Variables for changing movie frame ##################
RedShift = 2**(np.linspace(3,7,200))[::-1]
zoom = np.linspace(0,0.64,200)[::-1]
colorD = np.linspace(3.,20.,200)[::-1]
colorC = np.linspace(0,0.35,200)[::-1]

alpha = np.zeros([10,200])
alpha[0] = np.linspace(0.5,1,200)
alpha[1] = np.linspace(0.45,0.95,200)
alpha[2] = np.linspace(0.4,0.9,200)
alpha[3] = np.linspace(0.35,0.85,200)
alpha[4] = np.linspace(0.3,0.8,200)
alpha[5] = np.linspace(0.25,0.75,200)
alpha[6] = np.linspace(0.2,0.7,200)
alpha[7] = np.linspace(0.15,0.65,200)
alpha[8] = np.linspace(0.1,0.6,200)
alpha[9] = np.linspace(0.5,0.55,200)




#############################################################
XSize = 2.56
a = z.zeldovich(GridSize=256, XSize=XSize ,Seed = 300000)
GridSize = a.GridSize
Lbox = a.XSize
dx=a.dx
kx,ky,kz = a.k_array()
psik_scalar = a.psik_scalar()
psi_vector_x0,psi_vector_y0,psi_vector_z0 = a.psi_vector0()
WebC = a.web_classification(psik_scalar,kx,ky,kz)
del kx,ky,kz,psik_scalar
gc.collect()
print ('loop begins')
plt.rcParams['axes.facecolor'] = 'black'

for i in range(199,200):
	print ('generating fig no:',i)
	psi_vector_x,psi_vector_y,psi_vector_z = a.psi_vector_multiple(RedShift[i],psi_vector_x0,psi_vector_y0,psi_vector_z0 )
	Position = a.Position_multiple(psi_vector_x,psi_vector_y,psi_vector_z,RedShift[i])
	PositionX = Position[0,:,:,:].flatten()
	PositionY = Position[1,:,:,:].flatten()
	PositionZ = Position[2,:,:,:].flatten()
	colors = plt.cm.hot(WebC.reshape([GridSize**3])/colorD[i]+colorC[i])
	knew = np.mod(np.floor(PositionZ/dx),GridSize)
	for j in range(0,10)[::-1]:	
		ind = np.where(knew==j)[0]
		X = PositionX[ind]
		Y = PositionY[ind]
		color = colors[ind]
		plt.scatter(Y,X,s=0.005,color=color,alpha=alpha[j][i])
		plt.xticks([])
		plt.yticks([])
	plt.axis([Y.min()+zoom[i], Y.max()-zoom[i], X.min()+zoom[i], X.max()-zoom[i]])
	plt.gca().set_aspect('equal')
	plt.text(X.min()+zoom[i], Y.min()+zoom[i]+0.01, 'Scale = '+str(XSize)+'Mpc/h , RedShift = '+str(RedShift[i])[0:3] , color='white',fontsize=10,bbox={'facecolor':'black', 'alpha':0.6, 'pad':0})
	plt.savefig('zeld_mov_'+str(i)+'.png',dpi=100,bbox_inches='tight', pad_inches=0)
	plt.clf()
