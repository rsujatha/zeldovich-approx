import zeldovich as z
import time
import matplotlib.pyplot as plt
import numpy as np
import sys

a = z.zeldovich(GridSize=256, XSize = 256,Seed = 300000)
GridSize = a.GridSize
kx,ky,kz = a.k_array()
psik_scalar = a.psik_scalar()
start=time.time()
WebC = a.web_classification(psik_scalar,kx,ky,kz)
print time.time()-start
RedShift = np.logspace(0,2,50)[::-1]
#~ RedShift = RedShift[]
i=0
for RS in RedShift:
	Position = a.Position(RS)
	print time.time()-start
	X = Position[0,:,:,3].reshape([GridSize**2])
	Y = Position[1,:,:,3].reshape([GridSize**2])
	colors = plt.cm.jet(WebC[:,:,3].reshape([GridSize**2])/3.)
	plt.scatter(Y,X,s=0.2,color=colors)
	plt.title('Redshift = ' + str(RS))
	
	plt.savefig('web_classification_position_scatter'+str(i)+'.png')
	i=i+1
	plt.clf()
