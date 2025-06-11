from matplotlib import pyplot as plt
import EH_transfer as eh
import cosmolopy.perturbation as cseu
import numpy as np

omega0 = 0.3
omegab=0.045
hubble = 0.7
f = omegab/omega0
Tcmb = -1     ##Tcmb -- the CMB temperature in Kelvin. T<=0 uses the COBE value 2.728.
k = np.logspace(-5,2,1000)
numk = k.shape[0]


tf = np.zeros(numk,dtype=float)
ck =  eh.floatArray(numk)
ctf = eh.floatArray(numk)

for i in range(numk):
	ck[i] = k[i]

eh.TFfit_hmpc(omega0, f,  hubble, Tcmb,
		       numk, ck, ctf)

for i in range(numk):
	tf[i] = ctf[i]
	
print 'tf is ',tf
h = 0.7
omega_matter = omega0
omega_baryon = omegab
omega_cdm = omega_matter-omega_baryon
omega_lambda = 0.7
ns = 0.96
sigma_8 = 0.8

cosmology = { 'omega_n_0': 0.0,  'h': h, 'N_nu': 0, 'omega_lambda_0': omega_lambda,  'omega_b_0': omega_baryon,  'omega_M_0': omega_matter}
T_eu = cseu.transfer_function_EH(k*h, **cosmology)

print 'Teu is ',T_eu[0]
print 'diff',(T_eu[0]-tf).min(),(T_eu[0]-tf).max(),np.mean(T_eu[0]-tf)
plt.plot(k,tf)
plt.plot(k,T_eu[0])
plt.show()
plt.plot(k,(T_eu[0]/tf)-1)
plt.show()
