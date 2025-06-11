import mathtools as mt
import numpy as np
import matplotlib.pyplot as plt
import time

############ Initialising input vector################################

j=0
number = range(100,2000,100)
numberLog = np.floor(10**(np.linspace(1,3,50)))
print numberLog
timeD  = np.zeros(np.size(number))
timeF  = np.zeros(np.size(number))

for GridSize in number:
	A=np.random.normal(0,1,[GridSize,GridSize]) + np.random.normal(0,1,[GridSize,GridSize])*1j
	t1=time.time()
	B=mt.IDFT2(A)
	t2=time.time()
	C=np.fft.ifft2(A)
	t3=time.time()
	timeD[j] = t2-t1
	timeF[j] = t3-t2
	j=j+1

plt.plot(number,timeD,label='IDFT2')
plt.plot(number,timeF,label='FFT2')
plt.legend(loc='best')
plt.xlabel('Number of Data Points')
plt.ylabel('Time to perform Inverse Fourier Transform (s)')
plt.title('timing analysis ')
plt.savefig('timingAnalysis_IFFT2vsIDFT2.pdf')
plt.show()


j=0
timeD  = np.zeros(np.size(numberLog))
timeF  = np.zeros(np.size(numberLog))
for GridSize in numberLog:
	A=np.random.normal(0,1,[GridSize,GridSize]) + np.random.normal(0,1,[GridSize,GridSize])*1j
	t1=time.time()
	B=mt.IDFT2(A)
	t2=time.time()
	C=np.fft.ifft2(A)
	t3=time.time()
	timeD[j] = t2-t1
	timeF[j] = t3-t2
	j=j+1

plt.plot(numberLog,timeD,label='IDFT2')
plt.plot(numberLog,timeF,label='FFT2')
plt.legend(loc='best')
plt.xlabel('Number of Data Points')
plt.ylabel('Time to perform Inverse Fourier Transform (s)')
plt.title('timing analysis ')
plt.savefig('timingAnalysis_IFFT2vsIDFT2(log).pdf')
plt.xscale('log')
plt.yscale('log')
plt.show()
