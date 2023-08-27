import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

d = np.loadtxt("stdout")
NX = NY = 128

fig,ax = plt.subplots(layout="constrained")
plt.plot(d[:,0],d[:,1],"o-")
plt.xlabel(r"$t$")
plt.ylabel(r"$\langle\phi^2\rangle$")
plt.savefig("phi2.png")
plt.close()

fig,ax = plt.subplots(layout="constrained")
plt.loglog(d[:,0],1./(1.-d[:,1]),"o-")
plt.xlim(10.0,None)
plt.xlabel(r"$t$")
plt.ylabel(r"$s=(1-\langle\phi^2\rangle)^{-1}$")
plt.savefig("s.png")
plt.close()



frames = 200
fig = plt.figure(figsize=(12,4),layout="constrained")
def func(frame):
  fig.clear()
  axes = list()
  ax0 = fig.add_subplot(1,2,1)
  ax1 = fig.add_subplot(1,2,2)
  d = np.loadtxt(f"dat/fluid_{frame}.dat",skiprows=1)[:,:].reshape(NX,NY,5)
  im0 = ax0.pcolor (d[:,:,0],d[:,:,1],d[:,:,4],cmap="bwr",norm=Normalize(vmin=-1.1, vmax=1.1))
  im1 = ax1.pcolor(d[:,:,0],d[:,:,1],np.sqrt(d[:,:,2]**2+d[:,:,3]**2),cmap="plasma",norm=Normalize(vmin=0.0, vmax=1.0))
  cb0 = fig.colorbar(im0,ax=ax0)
  cb1 = fig.colorbar(im1,ax=ax1)
  # cb0.set_clim(-1.1,1.1)
  # cb1.set_clim(0.0,1.0)
ani = FuncAnimation(fig, func, frames=range(frames), interval=40)
ani.save('phi_uabs.mp4')
plt.close()
