import numpy as np
import matplotlib.pyplot as plt
import emcee
import triangle
from model import load_data
import sys
import h5py

fname = "simple"
whichx = str(sys.argv[1])
x, y, xerr, yerr = load_data(whichx, nd=0, bigdata=True)
m_init, b_init, lnsy_init = -2, 6, np.log(.0065)
if whichx == "rho":
    m_init, b_init, lnsy_init = -1.85, 7., np.log(.0065)

def lnlike((m, b, lns)):
    if not (-10 < lns < 5.):
        return -np.inf
    sig2 = (m*xerr)**2 + (yerr**2 + np.exp(2*lns))
    return -.5 * np.sum((m*x-y+b)**2/sig2 + np.log(sig2))

ndim, nwalkers = 3, 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike)

p0 = np.array([m_init, b_init, lnsy_init]) \
        + 1e-5 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 2000)
sampler.reset()
sampler.run_mcmc(p0, 5000)

triangle.corner(sampler.flatchain, truths=[-1.7, 2.013, np.log(.022)])
plt.savefig("blah_logg")

samp = sampler.chain[:, :, :].reshape(-1, ndim)
f = h5py.File("%s_samples_%s.h5" % (whichx, fname), "w")
data = f.create_dataset("samples", np.shape(samp))
data[:, 0] = samp[:, 0]
data[:, 1] = samp[:, 1]
data[:, 2] = np.exp(samp[:, 2])
f.close()
