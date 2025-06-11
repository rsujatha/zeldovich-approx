#######phase space for a no gravity and collisionless particles###############
import numpy as np 
import matplotlib.pyplot as plt

x=np.linspace(0,9.5,25)
v=np.sin(x)
i=0
for t in np.linspace(0,10,100):
	xnew=x+v*t
	plt.plot(xnew,v)
	plt.scatter(xnew,v)
	plt.xlabel('position')
	plt.ylabel('velocity')
	#~ plt.title('No Gravity No Collision particles Phase Space')
	plt.xlim([-5,20])
	plt.xticks([])
	plt.yticks([])
	string = 'phasespace_nogravity_nocollision'+str(i)+'.png'
	plt.savefig(string)
	#~ plt.show()
	plt.clf()
	i = i+1
