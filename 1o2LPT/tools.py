import numpy as np
import time 
import matplotlib.pyplot as plt
def timeit(start):
	return "time taken is "+ str(time.time()-start) +"seconds"

def mean(array,string):
	if (string == "AM"):
		d = (array[1:]+array[:-1])/2.
	elif (string == "GM") :
		d = np.sqrt(array[1:]*array[:-1])
	return d
def histogramit(a,bins=10,xlabel='xlabel',ylabel='ylabel'):
		hist,edge = np.histogram(a,bins=bins)
		d = plt.plot(mean(edge,"AM"),hist)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()
		return 
