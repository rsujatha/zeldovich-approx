from scipy import integrate
import numpy as np
import sys
import zeldovich as z
import numpy as np
import time
import matplotlib.pyplot as plt
start=time.time()
a = z.initial_density_field(GridSize=256, XSize = 2.56,Seed = 300000)
deltax=a.initial_deltax()
var = np.var(deltax)
std = np.sqrt(var)
print std
def DoroshkevishProb(l1,l2,l3):
	I1=l1+l2+l3
	I2=l1*l2+l2*l3+l3*l1
	return 15**3/(8*np.pi*np.sqrt(5)*std**6)*np.exp(-3*I1**2/std**2 +15*I2/(2*std**2))*(l1-l2)*(l2-l3)*(l1-l3)
print DoroshkevishProb(.002,.003,0.04)
llim=0.0
ulim=.086
print integrate.tplquad(DoroshkevishProb, llim, ulim,lambda l2: llim,lambda l2: l1,lambda l2,l3: llim,lambda l2,l3: l2,epsabs=1.49e-04, epsrel=1.49e-04)
print time.time()-start
