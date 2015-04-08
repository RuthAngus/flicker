import numpy as np
import matplotlib.pyplot as plt
import h5py
import emcee
import scipy.optimize as op
import triangle

# log-Likelihood
def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

# Priors
def lnprior(theta):
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf

# Posterior probability dist
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

# Set initial parameter values
m_init = -0.526023
b_init = 2.89219
f_init = 0.0043328

# load data
f8, f8_err, rho, rho_err = np.genfromtxt("flickers.dat").T
x, xerr = rho, rho_err
y, yerr = f8, f8_err

# Calculate least-square values
A = np.vstack((np.ones_like(x), x)).T
C = np.diag(yerr * yerr)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))

# Calculate maximum-likelihood values
nll = lambda *args: -lnlike(*args)
result = op.fmin(nll, [m_init, b_init, np.log(f_init)], args=(x, y, yerr))
m_ml, b_ml, lnf_ml = result

# run emcee
ndim, nwalkers = 3, 100
pos = [result + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
sampler.run_mcmc(pos, 500)

# Flatten chain and make triangle plot
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
samples[:, 2] = np.exp(samples[:, 2])
fig = triangle.corner(samples, labels=["$m$", "$b$", "$f$"],
                      truths=[m_init, b_init, np.log(f_init)])
fig.savefig("emcee_triangle.png")

# plot result
plt.clf()
xl = np.arange(0, 10, 0.01)
for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
    plt.plot(xl, m*xl+b, color="k", alpha=0.1)
plt.plot(xl, m_init*xl+b_init, color="r", lw=2, alpha=0.8)
plt.errorbar(x, y, yerr=yerr, fmt=".k")
plt.savefig("emcee_result")

# print results
m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
print m_mcmc, b_mcmc, f_mcmc

# save samples
f = h5py.File("emcee_samples.h5", "w")
data = f.create_dataset("samples", (np.shape(samples)))
data[:, 0] = samples[:, 1]
data[:, 1] = samples[:, 0]
data[:, 2] = samples[:, 2]**2  # convert to variance
f.close()
