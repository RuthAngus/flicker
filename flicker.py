import numpy as np
import matplotlib.pyplot as plt
import triangle
from params import plot_params
reb = plot_params()
from colours import plot_colours
cols = plot_colours()
import scipy.interpolate as spi
import h5py

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
    # intrinsic scatter
    plt.fill_between(ys, ((ys-alpha)/beta)-sigma, ((ys-alpha)/beta)+sigma,
                     color=cols.blue, alpha=.3, edgecolor="w")
    plt.fill_between(ys, ((ys-alpha)/beta)-2*sigma, ((ys-alpha)/beta)+2*sigma,
                     color=cols.blue, alpha=.2, edgecolor="w")
    plt.errorbar(y, x, xerr=yerr, yerr=xerr, fmt="k.", capsize=0, alpha=.5,
                 ecolor=".5", mec=".2")
    plt.plot(ys, (ys-alpha)/beta, ".2", linewidth=1)
    plt.ylim(.5, 4.)
    plt.xlim(1, 2.4)
    plt.ylabel("$\log_{10}(\\rho_{star}[\mathrm{kgm}^{-3}])$")
    plt.xlabel("$\log_{10}\mathrm{(8-hour~flicker~[ppm])}$")
    plt.savefig("flicker3")

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
    plt.clf()
    plt.hist(normed_resids, 20, histtype="stepfilled", color="w")
    plt.xlabel("Normalised residuals")
    print np.std(normed_resids)
    plt.savefig("residual_hist")

if __name__ == "__main__":

    # load and sort data
    f8, f8_err, rho, rho_err = np.genfromtxt("flickers.dat").T
    x, xerr = rho, rho_err
    y, yerr = f8, f8_err
    inds = np.argsort(x)
    x = x[inds]
    y = y[inds]
    yerr = yerr[inds]

    # load chains
    n1, c1 = np.genfromtxt("CODAchain1.txt").T
    n2, c2 = np.genfromtxt("CODAchain2.txt").T
    alpha_chain = np.concatenate((c1[1:10000], c2[1:10000]))
    beta_chain = np.concatenate((c1[10001:20000], c2[10001:20000]))
    tau_chain = np.concatenate((c1[20001:30000], c2[20001:30000]))
    samples = np.vstack((alpha_chain, beta_chain, tau_chain))
    make_flicker_plot(x, xerr, y, yerr, samples)
    make_inverse_flicker_plot(x, xerr, y, yerr, samples)
