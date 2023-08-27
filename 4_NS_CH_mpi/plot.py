import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

d = np.loadtxt("stdout")
NX = NY = 256

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



procs = 8
frames = 100
fig = plt.figure(figsize=(8,3),layout="constrained")
def func(frame):
  fig.clear()
  axes = list()
  ax0 = fig.add_subplot(1,2,1)
  ax1 = fig.add_subplot(1,2,2)
  d = np.vstack([np.loadtxt(f"dat/fluid_{frame}_{k}.dat",skiprows=1) for k in range(procs)]).reshape(NX,NY,5)
  im0 = ax0.pcolor (d[:,:,0],d[:,:,1],d[:,:,4],cmap="bwr",norm=Normalize(vmin=-1.1, vmax=1.1))
  im1 = ax1.pcolor(d[:,:,0],d[:,:,1],np.sqrt(d[:,:,2]**2+d[:,:,3]**2),cmap="plasma",norm=Normalize(vmin=0.0, vmax=1.0))
  cb0 = fig.colorbar(im0,ax=ax0)
  cb1 = fig.colorbar(im1,ax=ax1)
  # cb0.set_clim(-1.1,1.1)
  # cb1.set_clim(0.0,1.0)
ani = FuncAnimation(fig, func, frames=range(0,frames,2), interval=100)
ani.save('phi_uabs.mp4')
ani.save('phi_uabs.gif')
plt.close()
