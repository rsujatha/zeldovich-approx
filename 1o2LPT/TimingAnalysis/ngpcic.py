import numpy as np
import matplotlib.pyplot as plt
t1,t2,t3=np.loadtxt("TIME.txt",unpack=True)

x=np.array(range(16,256,8))
x=x**3
fit1 = np.polyfit(np.log10(x),np.log10( t1), deg=1)
fit2 = np.polyfit(np.log10(x),np.log10( t2), deg=1)
fit3 = np.polyfit(np.log10(x),np.log10( t3), deg=1)
leg1 = 'ngp-slope='+str(fit1[0])[0:4]	
leg2 = 'ngpsmooth-slope='+str(fit2[0])[0:4]
leg3 = 'cic-slope='+str(fit3[0])[0:4] 
plt.plot(x,t1,'.',label=leg1)
plt.plot(x,t1)
plt.plot(x,t2,'.',label=leg2)
plt.plot(x,t2)
plt.plot(x,t3,'.',label=leg3)
plt.plot(x,t3)
plt.yscale('log')
plt.xscale('log')
plt.title('timing analysis')
plt.legend(loc='best')
plt.xlabel('GridSize**3')
plt.ylabel('time in sec')
plt.savefig('timeinterpollog.pdf')
plt.clf()
plt.plot(x,t1,label='ngp')
plt.plot(x,t2,label='ngpsmooth')
plt.plot(x,t3,label='cic')
plt.xlabel('GridSize**3')
plt.ylabel('time in sec')
plt.legend(loc='best')
plt.title('timing analysis')
plt.savefig('timeinterpol.pdf')
