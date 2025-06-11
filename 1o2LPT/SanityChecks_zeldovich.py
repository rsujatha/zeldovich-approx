
# coding: utf-8

# In[1]:

import zeldovich as z
import numpy as np
import matplotlib.pyplot as plt
import gc
import sys

# # Sanity check for initial Density Field Class

# In[8]:

a=np.array([[1,2],[3,4]])
plt.pcolor(a)
plt.show()
plt.imshow(a)
plt.show()


a = z.initial_density_field()
deltax=a.initial_deltax()
GridSize=a.GridSize
variance = a.var() ####variance of the real and imaginary part of deltak random number  
variance = np.append(variance[:,:,::-2],variance)
print 'Variance of  delta x' ,np.var(deltax)
print 'Sum of Discrete Power Spectrum is ',np.sum(variance*2)
x_array = a.x_range()


# In[9]:

del a 
del variance
gc.collect()


# # Sanity check for Interpolation method class

# In[10]:

a = z.zeldovich(GridSize=128)
Position = a.Position(RedShift=10)

PositionX = Position[0,:,:,:].flatten()
PositionY = Position[1,:,:,:].flatten()
PositionZ = Position[2,:,:,:].flatten()
b= z.pmInterpolation()
Lbox = a.XSize
R = 0.004
ngp = b.ngp(PositionX,PositionY,PositionZ,Lbox,GridSize)
ngps = b.ngp_smooth(PositionX,PositionY,PositionZ,Lbox,GridSize,R)
cic = b.cic(PositionX,PositionY,PositionZ,Lbox,GridSize)
cicl = b.cicl(PositionX,PositionY,PositionZ,Lbox,GridSize)


# #### 1)   Sum of all  DELTAX = Summation(TotalCounts-1)=TotalCounts-TotalCounts=0

# In[11]:

print 'Sum of delta x (NGP)        ',np.sum(ngp-1)
print 'Sum of delta x (NGP_smooth) ',np.sum(ngps-1)
print 'Sum of delta x (CIC)        ',np.sum(cic-1)
print 'Sum of delta x (CIC_LOOP)   ',np.sum(cicl-1)


# #### 2)   Residues between different Interpolation

# In[12]:

fig=plt.figure(figsize=(15,15), dpi= 100, facecolor='w', edgecolor='k')
plt.subplot(2,2,1)
ngp_ngps ,edges = np.histogram((ngp-ngps),bins=100)
plt.plot((edges[0:-1]+edges[1:])/2,ngp_ngps)
plt.title('Residue between NGP-NGP smoothed R=0.004')
plt.xlabel('Particle Counts off')
plt.ylabel('Number of Grid Cells')
plt.subplot(2,2,2)
ngp_cic ,edges =np.histogram((ngp-cic),bins=100)
plt.plot((edges[0:-1]+edges[1:])/2,ngp_cic)
plt.title('Residue between NGP-CIC')
plt.xlabel('Particle Counts off')
plt.ylabel('Number of Grid Cells')
plt.subplot(2,2,3)
cic_ngps ,edges =np.histogram((cic-ngps),bins=100)
plt.plot((edges[0:-1]+edges[1:])/2,cic_ngps)
plt.title('Residue between CIC-NGP smoothed R=0.004')
plt.xlabel('Particle Counts off')
plt.ylabel('Number of Grid Cells')
plt.subplot(2,2,4)
cic_cicl ,edges =np.histogram((cic-cicl),bins=100)
plt.plot((edges[0:-1]+edges[1:])/2,cic_cicl)
plt.title('Residue between CIC-CIC loop')
plt.xlabel('Particle Counts off')
plt.ylabel('Number of Grid Cells')
plt.show()


# In[13]:

X = Position[0,:,:,:].reshape([GridSize**3])
Y = Position[1,:,:,:].reshape([GridSize**3])
fig=plt.figure(figsize=(12,12), dpi= 100, facecolor='w', edgecolor='k')
plt.subplot(2,2,1)
plt.pcolor(x_array,x_array,deltax[:,:,3])
plt.gca().invert_yaxis()
plt.scatter(Y,X,s=0.00005)
#plt.title('Residue between NGP-NGP smoothed')
#plt.xlabel('Particle Counts off')
#plt.ylabel('Number of Grid Cells')
plt.subplot(2,2,2)
plt.imshow(ngp[:,:,3])
plt.colorbar()
plt.title('NGP')
#plt.xlabel('Particle Counts off')
#plt.ylabel('Number of Grid Cells')
plt.subplot(2,2,3)
plt.imshow(ngps[:,:,3])
plt.colorbar()
plt.title('NGP smoothed R=0.004')
#plt.xlabel('Particle Counts off')
#plt.ylabel('Number of Grid Cells')
plt.subplot(2,2,4)
plt.imshow(cic[:,:,3])
plt.title(' CIC')
plt.colorbar()
#plt.xlabel('Particle Counts off')
#plt.ylabel('Number of Grid Cells')
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:



