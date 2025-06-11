import zeldovich as z
import numpy as np
import matplotlib.pyplot as plt
import gc
import time
Lbox=256 
inst_Np256 = z.zeldovich(GridSize=256, XSize = 256,Seed = 300000)
Position = inst_Np256.Position(RedShift=2)
PositionX = Position[0,:,:,:].flatten()
PositionY = Position[1,:,:,:].flatten()
PositionZ = Position[2,:,:,:].flatten()

a = z.pmInterpolation()
##Computing Residues for Np=256
Res_Ng256_Np256 = a.ngp(PositionX,PositionY,PositionZ,Lbox,GridSize=256)-a.cic(PositionX,PositionY,PositionZ,Lbox,GridSize=256) 
Res_Ng256_Np256 = a.ngp(PositionX,PositionY,PositionZ,Lbox,GridSize=128)-a.cic(PositionX,PositionY,PositionZ,Lbox,GridSize=128)
Res_Ng256_Np256 = a.ngp(PositionX,PositionY,PositionZ,Lbox,GridSize=64)-a.cic(PositionX,PositionY,PositionZ,Lbox,GridSize=64)
