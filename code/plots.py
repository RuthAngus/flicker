import numpy as np
import matplotlib.pyplot as plt
import triangle
from colours import plot_colours
cols = plot_colours()
import scipy.interpolate as spi
import h5py
import sys

def interp(xold, yold, xnew, s):
    tck = spi.splrep(xold, yold, s=s)
    ynew = spi.splev(xnew, tck, der=0)
    return ynew

def make_inverse_flicker_plot(x, xerr, y, yerr, samples, plot_samp=False):
    assert np.shape(samples)[0] < np.shape(samples)[1], \
            "samples is wrong shape"
    alpha, beta, tau = np.median(samples, axis=1)
    sigma = np.sqrt(tau)
    print alpha, beta, tau, sigma
    ys = alpha + beta * x
    sig_tot = sigma + yerr
    ys = np.linspace(min(y)-5, max(y)+5, 1000)
    xs = np.linspace(min(x), max(x), 1000)
    plt.clf()

    if fname == "rho":
        plt.ylabel("$\log_{10}(\\rho_{\star}[\mathrm{g~cm}^{-3}])$")
        col = cols.blue
        plt.text(1.7, .5, "$\log_{10} (\\rho_{\star}) \sim \mathcal{N} (\\alpha + \\beta F_8, \sigma)$")
        plt.text(1.7, .2, "$\\alpha = %.3f$" % (alpha-2))
        plt.text(1.7, .05, "$\\beta = %.3f$" % beta)
        plt.text(1.7, -.1, "$\sigma = %.3f$" % tau**.5)
        plt.fill_between(ys, ((ys-alpha)/beta)-sigma-3,
                         ((ys-alpha)/beta)+sigma-3,
                         color=col, alpha=.3, edgecolor="w")
        plt.fill_between(ys, ((ys-alpha)/beta)-2*sigma-3,
                         ((ys-alpha)/beta)+2*sigma-3,
                         color=col, alpha=.2, edgecolor="w")
        dx = x/xerr
        plt.errorbar(y, x-3, xerr=yerr, yerr=xerr, fmt="k.", capsize=0,
                     alpha=.5, ecolor=".5", mec=".2")
        plt.plot(ys, (ys-alpha)/beta-3, ".2", linewidth=1)
        plt.ylim(-2, 1)
        plt.subplots_adjust(bottom=.1)

    elif fname == "logg":
        plt.ylim(3, 5)
        col = cols.pink
        plt.ylabel("$\log_{10}(g [\mathrm{cm~s}^{-2}])$")
        plt.text(1.7, 4.7, "$\log(g) \sim \mathcal{N} (\\alpha + \\beta F_8, \sigma)$")
        plt.text(1.7, 4.5, "$\\alpha = %.3f$" % alpha)
        plt.text(1.7, 4.4, "$\\beta = %.3f$" % beta)
        plt.text(1.7, 4.3, "$\sigma = %.3f$" % tau**.5)

        plt.fill_between(ys, ((ys-alpha)/beta)-sigma, ((ys-alpha)/beta)+sigma,
                         color=col, alpha=.3, edgecolor="w")
        plt.fill_between(ys, ((ys-alpha)/beta)-2*sigma, ((ys-alpha)/beta)+2*sigma,
                         color=col, alpha=.2, edgecolor="w")
        plt.errorbar(y, x, xerr=yerr, yerr=xerr, fmt="k.", capsize=0, alpha=.5,
                     ecolor=".5", mec=".2")
        plt.plot(ys, (ys-alpha)/beta, ".2", linewidth=1)
        plt.subplots_adjust(bottom=.1)

    plt.xlim(1, 2.4)
    plt.xlabel("$\log_{10}\mathrm{(F}_8~\mathrm{[ppm]})$")
    plt.savefig("/Users/angusr/Dropbox/Flickerhackday/figs/%s_vs_flicker.pdf" % fname)
    plt.savefig("flicker_inv_%s" % fname)

def make_flicker_plot(x, xerr, y, yerr, samples, plot_samp=False):

    assert np.shape(samples)[0] < np.shape(samples)[1], \
            "samples is wrong shape"

    alpha, beta, tau = np.median(samples, axis=1)
    sigma = np.sqrt(tau)
    print alpha, beta, tau, sigma

    ys = alpha + beta * x
    sig_tot = sigma + yerr
    xs = np.linspace(min(x), max(x), 1000)

    plt.clf()
    # plot posterior samples
    if plot_samp:
        # sample posteriors
        nsamp = 1000
        a = np.random.choice(samples[0, :], nsamp)
        b = np.random.choice(samples[1, :], nsamp)
        t = np.random.choice(samples[2, :], nsamp)
        s = np.sqrt(t)
        for i in range(nsamp):
            plt.plot(xs, a[i]+b[i]*xs+np.random.randn(1)*s[i], color=".5",
                     alpha=.03)
    else:
        # intrinsic scatter
        plt.fill_between(xs, alpha+beta*xs-sigma, alpha+beta*xs+sigma,
                         color=cols.blue, alpha=.4, edgecolor="w")
        plt.fill_between(xs, alpha+beta*xs-2*sigma, alpha+beta*xs+2*sigma,
                         color=cols.blue, alpha=.2, edgecolor="w")

    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="k.", capsize=0, alpha=.5,
                 ecolor=".5", mec=".2")
    plt.plot(xs, alpha+beta*xs, ".2", linewidth=1)

    plt.xlim(min(xs), max(xs))
    plt.ylim(.5, 2.5)
    plt.xlabel("$\log_{10}(\\rho_{star}[\mathrm{kgm}^{-3}])$")
    plt.ylabel("$\log_{10}\mathrm{(8-hour~flicker~[ppm])}$")
    if plot_samp:
        plt.savefig("flicker2")
    else:
        plt.savefig("flicker")

    resids = y - (alpha+beta*x)
    normed_resids = resids / np.sqrt(yerr**2 + sigma**2)
    np.savetxt("normed_resids_%s.txt", np.transpose(normed_resids))
    plt.clf()
    plt.hist(normed_resids, 20, histtype="stepfilled", color="w")
    plt.xlabel("Normalised residuals")
    plt.savefig("residual_hist_%s" % fname)

if __name__ == "__main__":

    plotpar = {'axes.labelsize': 18,
               'text.fontsize': 18,
               'legend.fontsize': 18,
               'xtick.labelsize': 18,
               'ytick.labelsize': 18,
               'text.usetex': True}
    plt.rcParams.update(plotpar)

    fname = str(sys.argv[1]) # should be either "rho" or "logg"

    if fname == "rho":
        y, yerr, x, xerr = np.genfromtxt("../data/flickers.dat").T
    elif fname == "logg":
        y, yerr, x, xerr, _, _ = np.genfromtxt("../data/log.dat").T
    inds = np.argsort(x)
    x = x[inds]
    y = y[inds]
    yerr = yerr[inds]

    # load chains
    with h5py.File("%s_samples.h5" % fname, "r") as f:
        samples = f["samples"][...]
    samples = samples.T

    make_flicker_plot(x, xerr, y, yerr, samples, fname)
    make_inverse_flicker_plot(x, xerr, y, yerr, samples, fname)
