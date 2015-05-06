import numpy as np
import matplotlib.pyplot as plt
from noisy_plane import generate_samples, lnlike, lnlikeH
from model import model1, model
import emcee
import triangle

def lnprior(pars):
    return 0.

def lnprob(pars, samples, obs, u, is2d=False):
    if is2d:
        return lnlikeH(pars, samples, obs, u) + lnprior(pars)
    return lnlike(pars, samples, obs, u) + lnprior(pars)

if __name__ == "__main__":

    # load data. x = f, y = rho, z = teff
    x, xerr, y, yerr = np.genfromtxt("data/flickers.dat").T

    # generate nsamp importance samples unless 2d is False
    is2d = False
    obs = np.vstack((x, y))
    u = np.vstack((xerr, yerr))
    nsamp = 1
    if is2d:
        nsamp = 3
    s = generate_samples(obs, u, nsamp)

    # rough values from Kipping paper
    pars_init = [-.5, 3.6, .065]

    ndim, nwalkers = len(pars_init), 32
    pos = [pars_init + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(s, obs, u))

    print "burning in..."
    pos, _, _, = sampler.run_mcmc(pos, 200)
    sampler.reset()
    print "production run..."
    sampler.run_mcmc(pos, 1000)
    samp = sampler.chain[:, 50:, :].reshape((-1, ndim))
    m, c, lnf = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
               zip(*np.percentile(samp, [16, 50, 84], axis=0)))
    pars = [m[0], c[0], lnf[0]]

    plt.clf()
    plt.plot(s[0, :, :], s[1, :, :], "r.", markersize=2)
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="k.", capsize=0)
    print pars_init, pars
    plt.plot(x, model1(pars_init, x), color="b")
    plt.plot(x, model1(pars, x), color="g")
    plt.savefig("rhomcmc")

    fig = triangle.corner(samp, truths=pars_init)
    fig.savefig("rhotriangle.png")
