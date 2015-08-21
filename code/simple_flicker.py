import numpy as np
import matplotlib.pyplot as plt
import emcee
import triangle
from model import load_data
import sys

# N = 25
# m_true, b_true = -1.5, .25
# lnsy_true = -1
# xerr = np.random.uniform(.1, .2, N)
# x0 = np.random.uniform(0, 5, N)
# x = x0 + xerr * np.random.randn(N)
# yerr = np.random.uniform(.1, .5, N)
# y0 = m_true * x0 + b_true
# y = y0 + np.sqrt(yerr**2 + np.exp(2*lnsy_true)) * np.random.randn(N)

whichx = str(sys.argv[1])
x, y, xerr, yerr = load_data(whichx, nd=0, bigdata=True)
m_init, b_init, lnsy_init = -2, 6, np.log(.0065)
if whichx == "rho":
    m_init, b_init, lnsy_init = -1.85, 7., np.log(.0065)

plt.clf()
plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt="k.", capsize=0)
plt.savefig("simple1")

def lnlike((m, b, lns)):
    if not (-10 < lns < 5.):
        return -np.inf
    sig2 = (m*xerr)**2 + (yerr**2 + np.exp(2*lns))
    return -.5 * np.sum((m*x-y+b)**2/sig2 + np.log(sig2))

ndim, nwalkers = 3, 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike)

p0 = np.array([m_init, b_init, lnsy_init]) \
        + 1e-5 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 1000)
sampler.reset()
sampler.run_mcmc(p0, 1000)

plt.clf()
plt.plot(sampler.chain[:, :, 0].T, "k", alpha=.2)
plt.savefig("simple2")

plt.clf()
triangle.corner(sampler.flatchain, truths=[m_init, b_init, lnsy_init])
plt.savefig("simple3")
