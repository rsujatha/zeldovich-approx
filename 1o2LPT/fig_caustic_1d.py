#######phase space for a no gravity and collisionless particles###############
import numpy as np 
import matplotlib.pyplot as plt

x=np.linspace(0,9.5,30)
v=np.sin(x)
t=0
xnew=x+v*t
f, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True,figsize=(15,5))
ax1.plot(xnew,v)
ax1.scatter(xnew,v,label='time t1')
ax1.set_xlim([-1,11])
ax1.legend(loc='best')
ax1.set_ylabel('velocity')
ax1.set_xlabel('position')

t=1
xnew=x+v*t
ax2.plot(xnew,v)
ax2.scatter(xnew,v,label='t2>t1')
ax2.set_xlim([-1,11])
ax2.set_ylabel('velocity')
ax2.set_xlabel('position')

#~ string = 'phasespace_nogravity_nocollision'+str(t)+'.png'
#~ plt.savefig(string)
ax2.legend(loc='best')

t=2.5
xnew=x+v*t
ax3.plot(xnew,v)
ax3.scatter(xnew,v,label='t3>t2')
ax3.set_xlim([-1,11])
ax3.legend(loc='best')
f.subplots_adjust(hspace=0)
#~ plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.xlabel('position')
plt.ylabel('velocity')
plt.savefig('fig_caustic_1d.pdf',bbox_inches='tight', pad_inches=0)


plt.tight_layout()
plt.show()
