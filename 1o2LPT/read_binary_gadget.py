## This class reads binary file of gadget output
from __future__ import division
import numpy as np


class read_binary_gadget(object):
	def __init__(self,string = None):
		assert type(string) is type(""), "This class requires the gadget filename as input"
		self.g = open(string,'rb')
		start_header = np.fromfile(self.g,dtype=np.uint32,count=1)
		print "reading the first block (header) which contains ",start_header," bytes"
		self.NumPart_ThisFile = np.fromfile(self.g,dtype=np.int32,count=6)    ## The number of particles of each type present in the file
		self.MassTable     = np.fromfile(self.g,dtype=np.float64,count=6)     ## Gives the mass of different particles
		self.Time          = np.fromfile(self.g,dtype=np.float64,count=1)[0]   ##Time of output, or expansion factor for cosmological simulations
		self.Redshift      = np.fromfile(self.g,dtype=np.float64,count=1)[0]   ## REdshift of the snapshot
		self.Flag_Sfr       = np.fromfile(self.g,dtype=np.int32,count=1)[0]     ##Flag for star 
		self.Flag_Feedback  = np.fromfile(self.g,dtype=np.int32,count  = 1)[0]  ##Flag for feedback
		self.NumPart_Total           = np.fromfile(self.g,dtype=np.int32,count = 6)  ## Total number of each particle present in the simulation
		self.Flag_Cooling   = np.fromfile(self.g,dtype=np.int32,count =1)[0]     ## Flag used for cooling
		self.NumFilesPerSnapshot = np.fromfile(self.g,dtype=np.int32,count = 1)[0] ## Number of files in each snapshot
		self.BoxSize = np.fromfile(self.g,dtype = np.float64,count = 1)[0]  ## Gives the box size if periodic boundary conditions are used
		self.Omega0 = np.fromfile(self.g,dtype = np.float64,count=1)[0]     ## Matter density at z = 0 in the units of critical density
		self.OmegaLambda = np.fromfile(self.g,dtype = np.float64,count=1)[0]## Vacuum Energy Density at z=0 in the units of critical density
		self.HubbleParam = np.fromfile(self.g,dtype = np.float64,count =1 )[0] ## gives the hubble constant in units of 100 kms^-1Mpc^-1  
		self.Flag_StellarAge = np.fromfile(self.g,dtype = np.int32 ,count =1)[0]  ##Creation time of stars
		self.Flag_Metals  = np.fromfile(self.g,dtype = np.int32 ,count =1)[0] ##Flag for metallicity values
		self.NumPart_Total_HW = np.fromfile(self.g,dtype = np.int32,count = 6) ## For simulations more that 2^32 particles this field holds the most significant word of the 64 bit total particle number, otherwise 0
		self.Flag_Entropy_ICs = np.fromfile(self.g,dtype = np.int32,count = 1)[0] ## Flag that initial conditions contain entropy instead of thermal energy in the u block
		self.g.seek(256 +4 ,0)
		end = np.fromfile(self.g,dtype = np.int32,count =1)[0]
		print 'Header block is read and it contains ',end,'bytes'
		
	def read_posd(self):
		start_position = np.fromfile(self.g,dtype = np.int32,count =1)[0]
		print "reading the second block (position) which contains ",start_position," bytes"
		posd = np.fromfile(self.g,dtype = np.float32,count =self.NumPart_ThisFile[1] *3)  ### The positions are arranged in the binary file as follow: x1,y1,z1,x2,y2,z2,x3,y3,z3 and so on till xn,yn,zn
		posd= posd.reshape([self.NumPart_ThisFile[1],3])   ## reshape keeps the fastest changing axis in the end, since x,y,z dimensions are the ones changing the fastest they are given the last axis.
		end = np.fromfile(self.g,dtype = np.int32,count =1)[0]
		print 'Position block is read and it contains ',end,'bytes'
		return posd
		
	def read_veld(self):
		self.g.seek(256 + 8 + 8 + self.NumPart_ThisFile[1]*3*4,0)
		start_position = np.fromfile(self.g,dtype = np.int32,count =1)[0]
		print "reading the third block (velocity) which contains ",start_position," bytes"
		veld = np.fromfile(self.g,dtype = np.float32,count =self.NumPart_ThisFile[1] *3)  ### The velocities are arranged in the binary file as follow: x1,y1,z1,x2,y2,z2,x3,y3,z3 and so on till xn,yn,zn
		veld= veld.reshape([self.NumPart_ThisFile[1],3])   ## reshape keeps the fastest changing axis in the end, since x,y,z dimensions are the ones changing the fastest they are given the last axis.
		end = np.fromfile(self.g,dtype = np.int32,count =1)[0]
		print 'velocity block is read and it contains ',end,'bytes'
		return veld
		
	def read_pid(self):
		self.g.seek(256 + 8 + 8 + 8 + 2 * self.NumPart_ThisFile[1]*3*4,0)
		start_position = np.fromfile(self.g,dtype = np.int32,count =1)[0]
		print "reading the fourth block (pID) which contains ",start_position," bytes"
		pid = np.fromfile(self.g,dtype = np.uint32,count =self.NumPart_ThisFile[1])
		end = np.fromfile(self.g,dtype = np.int32,count =1)[0]
		print 'pID block is read and it contains ',end,'bytes'
		return pid
