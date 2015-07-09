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
    if -10 < pars[0] < 10 and -10 < pars[1] < 10 and 0 < pars[2] < 1:
        return 0.
    return -np.inf

def lnpriorHF(pars):
    if pars[2] < 0:
        return -np.inf
    return 0

def lnprob(pars, samples, obs, u, mm=False):
    return lnlikeH(pars, samples, obs, u) + lnpriorHF(pars)

def MCMC(whichx, nsamp, fname, nd, bigdata, burnin=500, run=500):

    rho_pars = [-2., 6., .0065]
    logg_pars = [-1.850, 7., .0065]
    pars_init = logg_pars
    if whichx == "rho":
        pars_init = rho_pars

    x, y, xerr, yerr = load_data(whichx, nd=nd, bigdata=True)

    # format data and generate samples
    obs = np.vstack((x, y))
    u = np.vstack((xerr, yerr))
    up = np.vstack((xerr, yerr))
    um = np.vstack((xerr*.5, yerr*.5))
#     s = generate_samples_log(obs, up, um, nsamp) # FIXME
    s = generate_samples(obs, u, nsamp) # FIXME
#     if nsamp == 1:
#         s[0, :, :] = x
#         s[1, :, :] = y
#     print np.shape(s)
#     assert 0

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
    m, c, sig = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
               zip(*np.percentile(samp, [16, 50, 84], axis=0)))
    pars = [m[0], c[0], sig[0]]

    # save samples
    f = h5py.File("%s_samples_%s.h5" % (whichx, fname), "w")
    data = f.create_dataset("samples", np.shape(samp))
    data[:, 0] = samp[:, 0]
    data[:, 1] = samp[:, 1]
    data[:, 2] = samp[:, 2]
    f.close()

def make_plots(whichx, fname):

    x, y, xerr, yerr = load_data(whichx)

    with h5py.File("%s_samples.h5" % whichx) as f:
        samp = f["samples"][...]
    m, c, sig = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
               zip(*np.percentile(samp, [16, 50, 84], axis=0)))
    pars = [m[0], c[0], sig[0]]
    print pars

    plt.clf()
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="k.", capsize=0, ecolor=".7")
    plt.plot(x, model1(pars, x), "k")
    ndraws = 100
    p0s = np.random.choice(samp[:, 0], ndraws)
    p1s = np.random.choice(samp[:, 1], ndraws)
    p2s = np.random.choice(samp[:, 2], ndraws)
    for i in range(ndraws):
        y = p0s[i] * x + p1s[i]
        plt.plot(x, (y + p2s[i]), "k", alpha=.1)
    plt.savefig("mcmc_%s_%s" % (whichx, fname))
    labels = ["$m$", "$c$", "$\sigma$"]
    plt.clf()
    fig = triangle.corner(samp, labels=labels)
    fig.savefig("triangle_%s_%s" % (whichx, fname))

if __name__ == "__main__":
    whichx = str(sys.argv[1])
    fname = "f"
    MCMC(whichx, 10, fname, 0, bigdata=True)
    make_plots(whichx, fname)

#     # load data
#     data = np.genfromtxt("../data/BIGDATA.filtered.dat").T
#     kid = data[0]
#     all_mass = data[1]
#     r, rerrp, rerrm = data[7:10]
#     rerrp, rerrm = rerrp/r/np.log(10), rerrm/r/np.log(10)
#     r = np.log10(1000*r)
#     f, ferr = data[20:22]
#     data2 = np.genfromtxt("../data/BIGDATA.nohuber.filtered.dat").T
# #     data2 = np.genfromtxt("../data/BIGDATA.filtered.dat").T
#     kid_no_huber = data2[0]
#     logg, loggerrp, loggerrm = data2[10:13]
#
# #     plt.clf()
# #     plt.plot(f, r, "k.")
# #     plt.plot(f, logg, "r.")
# #     plt.savefig("bastien_comparison")
#
#     if whichx == "rho":
#         f, ferr = data[20:22]
#     ferrp, ferrm = ferr/f/np.log(10), ferr/f/np.log(10)
#     f = np.log10(1000*f)
#
#     # format data
#     nd = len(f)
#     m = np.isfinite(logg[:nd])
#     x, xerrp, xerrm = f[:nd][m]-3, ferrp[:nd][m], ferrm[:nd][m]
#     y, yerrp, yerrm = logg[:nd][m], loggerrp[:nd][m], loggerrm[:nd][m]
#     xerr = .5*(xerrp + xerrm)
#     yerr = .5*(yerrp + yerrm)
#     if whichx == "rho":
#         m = np.isfinite(r[:nd])
#         kid = kid[:nd][m]
#         all_mass = all_mass[:nd][m]
#         x, xerrp, xerrm = f[:nd][m]-3, ferrp[:nd][m], ferrm[:nd][m]
#         y, yerrp, yerrm = r[:nd][m], rerrp[:nd][m], rerrm[:nd][m]
#         xerr = .5*(xerrp + xerrm)
#         yerr = .5*(yerrp + yerrm)
#
#     # just huber stars
#     huber, mass, teff = np.genfromtxt("../data/huber/TableY.dat",
#                                       usecols=(0,1,16)).T
#     hub, hf, hferr, hr, hrerr, hteff, hmass = [], [], [], [], [], [], []
#     for i, k in enumerate(huber):
#         m = kid == k
#         if len(kid[m]):
#             hteff.append(teff[i])
#             hmass.append(mass[i])
#             hub.append(kid[m][0])
#             print len(x), len(kid)
#             hf.append(x[m][0])
#             hferr.append(xerr[m][0])
#             hr.append(y[m][0])
#             hrerr.append(yerr[m][0])
#
#     pkid, rpl, porb, prot = np.genfromtxt("../data/KOIrotation_periods.txt",
#                                           delimiter=",", skip_header=20,
#                                           usecols=(1,4,5,6)).T
#
#     hub = np.array(hub)
#     hf = np.array(hf)
#     hr = np.array(hr)
#     phub, phf, phr, phrpl, phporb, phprot = [], [], [], [], [], []
#     for i, k in enumerate(hub):
#         m = pkid == k
#         if len(pkid[m]):
#             phub.append(pkid[m][0])
#             phf.append(hf[i])
#             phr.append(hr[i])
#             phrpl.append(rpl[m][0])
#             phporb.append(porb[m][0])
#             phprot.append(prot[m][0])
#
# #     plt.clf()
# #     plt.hist(hmass, normed=True)
# #     plt.hist(all_mass, normed=True, alpha=.5, color="r")
# #     plt.savefig("huber_hist")
#     plt.clf()
#     plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="k.", zorder=0)
# #     plt.scatter(x, y, c=all_mass, vmin=.8, vmax=1.6, zorder=0, edgecolor=None)
# #     plt.scatter(hf, hr, c=hmass, vmin=.8, vmax=1.6, s=80, marker="^", zorder=1)
#     plt.scatter(hf, hr, c=hteff, s=80, marker="^", zorder=1)
# #     plt.errorbar(hf, hr, xerr=hferr, yerr=hrerr, fmt="k.", zorder=1)
# #     print phprot
# #     print phporb
# #     plt.scatter(hf, hr, c=np.log10(phporb), s=80, marker="^", zorder=1)
#     plt.xlabel("flicker")
#     plt.ylabel("density")
# #     plt.colorbar()
#     plt.savefig("huber_test")
