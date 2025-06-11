import numpy as np
import matplotlib.pyplot as plt
import library as lb
import time
a=np.linspace(0.001,1,100)

D_Eds=a


D_Lcdm0=lb.GrowthFunctionNumerical(a,0.3,0.7)
D_Lcdm1=lb.GrowthFunctionNumerical(a,0.2,0.7)
D_Lcdmm1=lb.GrowthFunctionNumerical(a,0.36,0.7)
D_Lcdmm2=lb.GrowthFunctionNumerical(a,1.4,0)



D_Lcdm=lb.GrowthFunctionAnalytic(a)


plt.plot(a,D_Eds,label='EdS')
#plt.plot(a,D_Lcdm0,label='$\Lambda$CDM n k0 ($\Omega_m$=0.3,$\Omega_{\Lambda}$=0.7)')
#plt.plot(a,D_Lcdm1,label='$\Lambda$CDM n k -ve ($\Omega_m=0.2$,$\Omega_{\Lambda}=0.7$)')
#plt.plot(a,D_Lcdmm1,label='$\Lambda$CDM n k+ve ($\Omega_m=0.36$,$\Omega_{\Lambda}=0.7$)')
#~ plt.plot(a,D_Lcdm,label='$\Lambda$CDM ($\Omega_m=0.3$,$\Omega_{\Lambda}=0.7$)')
plt.plot(a,D_Lcdm,label='$\Lambda$CDM ')
#plt.plot(a,D_Lcdmm2,'.',label='$\Lambda$CDM a ($\Omega_m=1.4$,$\Omega_{\Lambda}=0$)')
plt.legend(loc='best')
plt.title('Growth Function')
plt.xlabel('a')
plt.ylabel('$D^{+}$(a)')
plt.savefig('GrowthFnFlat.svg',dpi=100,bbox_inches='tight', pad_inches=0)
plt.show()


