import numpy as np
import lpt as l
import sys
import matplotlib.pyplot as plt

a = l.LPT()
psi_vector_x0, psi_vector_y0, psi_vector_z0 = a.psi_vector0()
psi2_vector_x0 , psi2_vector_y0 , psi2_vector_z0 = a.psi2_vector0()
i=0
RS = np.logspace(0,2,100)[::-1]
for RedShift in RS:
	psi_vector_x, psi_vector_y, psi_vector_z = a.psi_vector(RedShift,psi_vector_x0=psi_vector_x0,psi_vector_y0=psi_vector_y0,psi_vector_z0=psi_vector_z0)
	psi2_vector_x, psi2_vector_y, psi2_vector_z = a.psi2_vector(RedShift,psi2_vector_x0=psi2_vector_x0,psi2_vector_y0=psi2_vector_y0,psi2_vector_z0=psi2_vector_z0)
	positionzeld=a.Position_zeld(RedShift,psi_vector_x=psi_vector_x, psi_vector_y=psi_vector_y, psi_vector_z=psi_vector_z)
	position2lpt=a.Position_2lpt(RedShift,psi_vector_x=psi_vector_x, psi_vector_y=psi_vector_y, psi_vector_z=psi_vector_z,psi2_vector_x=psi2_vector_x, psi2_vector_y=psi2_vector_y, psi2_vector_z=psi2_vector_z)

	X=position2lpt[0,50,50,50]
	Y=position2lpt[1,50,50,50]
	plt.plot(X,Y,'.',color='black',markersize=2)
	X=positionzeld[0,50,50,50]
	Y=positionzeld[1,50,50,50]
	plt.plot(X,Y,'.',color='blue',markersize=2)
	plt.xlim([0.45,0.65])
	plt.ylim([0.45,0.6])
	#~ plt.text(2, 2, ' RedShift = '+str(RedShift)[0:3] , color='black',fontsize=10)

	plt.title('zeld(blue)vs2lpt(black)')
	plt.xlabel('Mpc/h')
	plt.ylabel('Mpc/h')
	plt.savefig('2lpt_particle0'+str(i)+'.png',dpi=100,bbox_inches='tight', pad_inches=0)
	i+=1
