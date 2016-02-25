import numpy as np
import matplotlib.pyplot as plt
import h5py
import corner

plotpar = {'axes.labelsize': 19,
           'text.fontsize': 19,
           'legend.fontsize': 19,
           'xtick.labelsize': 17,
           'ytick.labelsize': 17,
           'text.usetex': True}
plt.rcParams.update(plotpar)

with h5py.File("rho_samples.h5") as f:
    samples = f["samples"][...]
nwalkers, nsteps, ndim = np.shape(samples)
flat = samples.reshape((nwalkers*nsteps, ndim))

labels = ["$\\alpha_\\rho$", "$\\beta_\\rho$", "$\sigma_\\rho$"]
fig = corner.corner(flat, labels=labels)
plt.savefig("../version1.0/rho_triangle.pdf")
