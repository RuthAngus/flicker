import numpy as np
import matplotlib.pyplot as plt
from noisy_plane import generate_samples, generate_samples_log, \
        lnlike, lnlikeH, lnlikeHF
from model import model1, model, load_data
import emcee
import triangle
import sys
import h5py

def lnprior(pars, mm=False):
    if -100 < pars[0] < 100 and -100 < pars[1] < 100 and -10 < pars[2] < 10 \
            and -10 < pars[3] < 10:
        return 0.
    return -np.inf

def lnprob(pars, samples, obs, u):
    return lnlikeHF(pars, samples, obs, u) + lnprior(pars)

def MCMC(whichx, nsamp, fname, nd, bigdata, burnin=500, run=500):
    """
    nsamp (int) = number of samples.
    whichx (str) = logg or rho.
    fname (str) = the name for saving all output
    nd (int) = number of data points (for truncation).
    If this is zero, all the data are used.
    bigdata (boolean) which data file to use.
    """

    # set initial parameters
    rho_pars = [-2., 6., np.log(.0065), np.log(.05)]
    logg_pars = [-1.850, 7., np.log(.0065), np.log(.05)]
    pars_init = logg_pars
    if whichx == "rho":
        pars_init = rho_pars

    # load the data
    x, y, xerr, yerr = load_data(whichx, nd=nd, bigdata=True)

    # format data and generate samples
    obs = np.vstack((x, y))
    u = np.vstack((xerr, yerr))
    up = np.vstack((xerr, yerr))
    um = np.vstack((xerr*.5, yerr*.5))
    s = generate_samples(obs, u, nsamp)

    # set up and run emcee
    ndim, nwalkers = len(pars_init), 32
    pos = [pars_init + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(s, obs, u))
    print "burning in..."
    pos, _, _, = sampler.run_mcmc(pos, burnin)
    sampler.reset()
    print "production run..."
    sampler.run_mcmc(pos, run)
    samp = sampler.chain[:, 50:, :].reshape((-1, ndim))
    m, c, ln_sig, lnf = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
               zip(*np.percentile(samp, [16, 50, 84], axis=0)))
    pars = np.exp([m[0], c[0], ln_sig[0], lnf[0]])

    # save samples
    f = h5py.File("%s_samples_%s.h5" % (whichx, fname), "w")
    data = f.create_dataset("samples", np.shape(samp))
    data[:, 0] = np.exp(samp[:, 0])
    data[:, 1] = np.exp(samp[:, 1])
    data[:, 2] = np.exp(samp[:, 2])
    data[:, 3] = np.exp(samp[:, 3])
    f.close()

def make_plots(whichx, fname):

    x, y, xerr, yerr = load_data(whichx)

    with h5py.File("%s_samples_%s.h5" % (whichx, fname)) as f:
        samp = f["samples"][...]
    m, c, ln_sig, lnf = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
               zip(*np.percentile(samp, [16, 50, 84], axis=0)))
    pars = [m[0], c[0], ln_sig[0], lnf[0]]
    print pars

    labels = ["$m$", "$c$", "$\ln(\sigma)$", "$\ln(f)$"]
    plt.clf()
    fig = triangle.corner(samp, labels=labels)
    fig.savefig("triangle_%s_%s" % (whichx, fname))

if __name__ == "__main__":
    whichx = str(sys.argv[1])
    fname = "f_extra"
    nd = 0 # set to zero to use all the data
    MCMC(whichx, 500, fname, nd, bigdata=True)
    make_plots(whichx, fname)
