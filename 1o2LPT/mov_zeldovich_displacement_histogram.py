import numpy as np
import matplotlib.pyplot as plt
import lpt as lpt
import time





start=time.time()
##Call an instance of lpt
GridSize=256
XSize=256
init = lpt.LPT(GridSize=GridSize, XSize = XSize ,Seed = 300000)
##Initial displacement at redshift 100 
zeldx,zeldy,zeldz = init.psi_vector0()
lpt2x,lpt2y,lpt2z = init.psi2_vector0()
##Final Displacement and histogram of displacement at different redshifts in a loop
print 'starting loop'
i=0
bins=np.linspace(0,1,100)
titlestring='GridSize='+str(GridSize)+',XSize'+str(XSize)
for RedShift in np.linspace(80,100,100)[::-1]:
	X,Y,Z = init.psi_vector(RedShift,psi_vector_x0 = zeldx,psi_vector_y0 = zeldy,psi_vector_z0 = zeldz)
	disp = np.sqrt(X**2+Y**2+Z**2)
	probd,edge = np.histogram(disp,bins=bins,density = 1)
	string = 'z='+str(RedShift)[0:4]
	
	plt.plot((edge[0:-1]+edge[1:])/2,probd,label=string)
	plt.xlabel('displacement')
	plt.ylabel('probd')
	plt.ylim([0,9])
	plt.xlim([0,np.amax(bins)])
	plt.title(titlestring)
	plt.legend()
	plt.savefig('early_zeld_displacement_histogram_'+str(i)+'.svg')
	plt.clf()
	X,Y,Z = init.psi2_vector(RedShift,psi2_vector_x0 = lpt2x,psi2_vector_y0 = lpt2y,psi2_vector_z0 = lpt2z)
	disp = np.sqrt(X**2+Y**2+Z**2)
	probd,edge = np.histogram(disp,bins=bins,density = 1)
	plt.plot((edge[0:-1]+edge[1:])/2,probd,label=string)
	plt.xlabel('displacement')
	plt.ylabel('probd')
	plt.ylim([0,9])
	plt.xlim([0,np.amax(bins)])
	plt.title(titlestring)
	plt.legend()
	plt.savefig('early_2lpt_displacement_histogram_'+str(i)+'.svg')
	plt.clf() ## For speed
	i+=1
print 'code ended is',time.time()-start,'sec'
