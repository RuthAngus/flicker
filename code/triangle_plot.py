import numpy as np
import matplotlib.pyplot as plt
import h5py
import corner

plotpar = {'axes.labelsize': 15,
           'text.fontsize': 18,
           'legend.fontsize': 18,
           'xtick.labelsize': 15,
           'ytick.labelsize': 15,
           'text.usetex': True}
plt.rcParams.update(plotpar)

with h5py.File("rho_samples.h5") as f:
    samples = f["samples"][...]
nwalkers, nsteps, ndim = np.shape(samples)
flat = samples.reshape((nwalkers*nsteps, ndim))

labels = ["$\\alpha$", "$\\beta$", "$\sigma$"]
fig = corner.corner(flat, labels=labels)
plt.savefig("rho_triangle.pdf")
