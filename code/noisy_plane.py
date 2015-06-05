import numpy as np
import matplotlib.pyplot as plt
from model import model, model1

def generate_samples(obs, u, N):
    '''
    obs is a 2darray of observations. shape = (ndims, nobs)
    u in an ndarray of the uncertainties associated with the observations
    returns a 3d array of samples. shape = (nobs, ndims, nsamples)
    '''
    ndims, nobs = np.shape(obs)
    samples = np.zeros((ndims, nobs, N))
    for i in range(ndims):
        samples[i, :, :] = np.vstack([x0+xe*np.random.randn(N) for x0, xe in \
                zip(obs[i, :], u[i, :])])
    return samples

def lnlike(pars, samples, obs, u):
    '''
    Generic likelihood function for importance sampling with any number of
    dimensions.
    In samples, obs and u, the dependent variable should be 1st
    '''
    ndims, nobs, nsamp = samples.shape
    zpred = model(pars, samples)
    zobs = obs[1, :]
    zerr = u[1, :]
    ll = np.zeros((nobs, nsamp*nobs))
    for i in range(nobs):
        ll[i, :] = -.5*((zobs[i] - zpred)/zerr[i])**2
    loglike = np.sum(np.logaddexp.reduce(ll, axis=1))
    return loglike
#     return np.logaddexp.reduce(loglike, axis=0)

# n-D, hierarchical log-likelihood
def lnlikeH(pars, samples, obs, u):
    '''
    Generic likelihood function for importance sampling with any number of
    dimensions.
    Now with added jitter parameter (hierarchical)
    obs should be a 2d array of observations. shape = (ndims, nobs)
    u should be a 2d array of uncertainties. shape = (ndims, nobs)
    samples is a 3d array of samples. shape = (ndims, nobs, nsamp)
    '''
    ndims, nobs, nsamp = samples.shape
    zpred = model(pars, samples)
    zobs = obs[1, :]
    zerr = u[1, :]
    ll = np.zeros((nobs, nsamp*nobs))
    for i in range(nobs):
        inv_sigma2 = 1.0/(zerr[i]**2 + pars[2]**2)
        ll[i, :] = -.5*((zobs[i] - zpred)**2*inv_sigma2) + np.log(inv_sigma2)
    loglike = np.sum(np.logaddexp.reduce(ll, axis=1))
    return loglike

# n-D, hierarchical log-likelihood with sigma as a function of y
def lnlikeHF(pars, samples, obs, u):
    '''
    Generic likelihood function for importance sampling with any number of
    dimensions.
    Now with added jitter parameter (hierarchical)
    obs should be a 2d array of observations. shape = (ndims, nobs)
    u should be a 2d array of uncertainties. shape = (ndims, nobs)
    samples is a 3d array of samples. shape = (ndims, nobs, nsamp)
    '''
    ndims, nobs, nsamp = samples.shape
    ypred = model(pars, samples)
    yobs = obs[1, :]
    xobs = obs[0, :]
    yerr = u[1, :]
    ll = np.zeros((nobs, nsamp*nobs))
    for i in range(nobs):
        inv_sigma2 = 1.0/(yerr[i]**2 + pars[2]**2 + pars[3]*model1(pars, xobs[i]))
#         inv_sigma2 = 1.0/(yerr[i]**2 + (m*model1(pars, xobs[i]))**2)
#         inv_sigma2 = 1.0/(yerr[i]**2 + m/model1(pars, xobs[i]))
        ll[i, :] = -.5*((yobs[i] - ypred)**2*inv_sigma2) + np.log(inv_sigma2)
    loglike = np.sum(np.logaddexp.reduce(ll, axis=1))
    if np.isfinite(loglike):
        return loglike
    return -np.inf

def generate_samples_log(obs, up, um, N):
    '''
    obs is a 2darray of observations. shape = (ndims, nobs)
    up is a 2darray of the upper uncertainties associated with the
    observations, um is a 2d array of the lower uncertainties.
    returns a 3d array of samples. shape = (nobs, ndims, nsamples)
    '''
    ndims, nobs = np.shape(obs)
    samples = np.zeros((ndims, nobs, N))
    for j in range(nobs):
        for i in range(ndims):
            sp = obs[i, j] + up[i, j]*np.random.randn(2*N*int(up[i, j]/um[i, j]))
            sm = obs[i, j] + um[i, j]*np.random.randn(2*N)
            sp = sp[sp > obs[i, j]]
            sm = sm[sm < obs[i, j]]
            s = np.concatenate((sp, sm))
            samples[i, j, :] = np.random.choice(s, N)

#             if i == 1:
#                 plt.clf()
# #                 plt.hist(samples[i, j, :], 50)
#                 print samples[0, j, :]
#                 plt.plot(samples[0, j, :], samples[1, j, :], "r.", alpha=.01)
#                 plt.errorbar([obs[0, j]], [obs[1, j]],
#                              xerr=([um[0, j]], [up[0, j]]),
#                              yerr=([um[1, j]], [up[1, j]]), fmt="k.")
#                 plt.show()
    return samples

if __name__ == "__main__":
    pars = [.5, 10]
    # make fake data
    nobs = 20
    x = np.random.uniform(0, 10, nobs)
    xerr = np.ones_like(x) * .2
    x += np.random.randn(nobs) * .5
    y = pars[0] * x + pars[1]
    yerr = np.ones_like(y) * .5
    y += np.random.randn(nobs) * .5

    obs = np.vstack((y, x))  # dependent variable 1st
    u = np.vstack((yerr, xerr))
    nsamp = 50
    samples = generate_samples(obs, u, nsamp)
    print lnlike(pars, samples, obs, u)

#     plt.clf()
#     plt.plot(samples[0, 1, :], samples[1, 1, :], "r.", markersize=2)
#     plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="k.", capsize=0)
