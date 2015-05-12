import numpy as np
import matplotlib.pyplot as plt
from noisy_plane import generate_samples, lnlike, lnlikeH
from model import model1, model
import emcee
import triangle
import sys

def lnprior(pars):
    return 0.

def lnprob(pars, samples, obs, u):
    return lnlikeH(pars, samples, obs, u) + lnprior(pars)

def MCMC(whichx, nsamp):

    # set initial params
    rho_pars = [-.5, 3.6, .065]
    logg_pars = [-1.850, 5.413, .065]
    pars_init = logg_pars
    if whichx == "rho":
        pars_init = rho_pars

    # load data
    fr, frerr, r, rerr = np.genfromtxt("../data/flickers.dat").T
    fl, flerr, l, lerr, t, terr = np.genfromtxt("../data/log.dat").T
    x, xerr, y, yerr = fl[:10], flerr[:10], l[:10], lerr[:10]
    if whichx == "rho":
        x, xerr, y, yerr = fr[:10], frerr[:10], r[:10], rerr[:10]

    # format data and generate samples
    obs = np.vstack((x, y))
    u = np.vstack((xerr, yerr))
    s = generate_samples(obs, u, nsamp)

    # set up and run emcee
    ndim, nwalkers = len(pars_init), 32
    pos = [pars_init + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(s, obs, u))
    print "burning in..."
    pos, _, _, = sampler.run_mcmc(pos, 500)
    sampler.reset()
    print "production run..."
    sampler.run_mcmc(pos, 1000)
    samp = sampler.chain[:, 50:, :].reshape((-1, ndim))
    m, c, sig = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
               zip(*np.percentile(samp, [16, 50, 84], axis=0)))
    pars = [m[0], c[0], sig[0]]

    # make plots
    plt.clf()
    plt.plot(s[0, :, :], s[1, :, :], "r.", markersize=2)
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="k.", capsize=0)
    print pars_init, pars
    plt.plot(x, model1(pars_init, x))
    plt.plot(x, model1(pars, x))
    plt.savefig("mcmc_%s" % whichx)
    fig = triangle.corner(samp, truths=pars_init)
    fig.savefig("triangle_%s" % whichx)

if __name__ == "__main__":
    whichx = str(sys.argv[1])
    MCMC(whichx, 10)
