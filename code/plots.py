import numpy as np
import matplotlib.pyplot as plt
import triangle
import scipy.interpolate as spi
import h5py
import sys
from noisy_plane import model1
from model import load_data

def interp(xold, yold, xnew, s):
    tck = spi.splrep(xold, yold, s=s)
    ynew = spi.splev(xnew, tck, der=0)
    return ynew

def fit_straight_line(x, y, yerr):
    AT = np.vstack((np.ones_like(x), x))
    C = np.eye(len(y)) * yerr**2
    ATCA = np.dot(np.dot(AT, C), AT.T)
    ATCy = np.dot(np.dot(AT, C), y)
    return np.linalg.solve(ATCA, ATCy)

def make_inverse_flicker_plot(x, xerr, y, yerr, samples, whichx, fname, ndraws,
                              fractional=False, extra=False):

    # fit straight line
    lim = 200
    a1 = fit_straight_line(x[:lim], y[:lim], yerr[:lim])
    a2 = fit_straight_line(x[lim:], y[lim:], yerr[lim:])

    assert np.shape(samples)[0] < np.shape(samples)[1], \
            "samples is wrong shape"

    # take medians
    results = np.median(samples, axis=1)
    print 'results = ', results
    beta, alpha, tau = results[:3]
    if extra:
        print np.shape(results), "shape"
        beta, alpha, tau, f = results

    # use highest likelihood samples
    lls = samples[:, -1]
    m = lls == max(lls)
    beta, alpha, tau = [samples[i, m] for i in range(np.shape(samples)[0]-1)]
    sigma = abs(tau)**.5
    if fname == "simple":
        sigma = tau
    pars = [beta, alpha, sigma]

    print alpha, beta, tau, sigma

    b_samp = np.random.choice(samples[0, :], ndraws)
    a_samp = np.random.choice(samples[1, :], ndraws)
    t_samp = np.random.choice(samples[2, :], ndraws)
    s_samp = abs(t_samp)**.5 * np.random.randn(ndraws)
    if fname == "simple"
    s_samp = t_samp
    if extra:
        f_samp = np.random.choice(samples[3, :], ndraws)

    plt.clf()
    xs = np.linspace(1., 2.4, 100)
    if whichx == "rho":
        plt.ylabel("$\log_{10}(\\rho_{\star}[\mathrm{g~cm}^{-3}])$")
        col = "#FF33CC"
        plt.text(1.55, .5, "$\log_{10} (\\rho_{\star}) \sim \mathcal{N} \
                 (\\alpha + \\beta \log_{10}(F_8), \sigma_{\\rho})$")
        plt.text(1.95, .22, "$\\alpha = %.3f$" % (alpha-3))
        plt.text(1.95, .07, "$\\beta = %.3f$" % beta)
        plt.text(1.95, -.08, "$\\sigma_{\\rho} = %.3f$" % sigma)
        plt.ylim(-2, 1)
        lines = []
        for i in range(ndraws):
            ys = model1([b_samp[i], a_samp[i]], xs)
            if fractional:
#                 plt.plot(xs, ys + ys * s_samp[i] - 3, col, alpha=.05)
                lines.append(ys + ys * np.random.randn(1)*s_samp[i]
                             - 3) #FIXME: opt
            elif extra:
#                 plt.plot(xs, f_samp[i] * ys + s_samp[i] - 3, col,
#                          alpha=.05)
                lines.append(ys*f_samp[i] + np.random.randn(1)*s_samp[i]
                             - 3) #FIXME: opt
            else:
#                 plt.plot(xs, ys + s_samp[i] - 3, col, alpha=.05)
#                 plt.plot(xs, ys + np.random.randn(1)*s_samp[i] - 3, col,
#                          alpha=.05)
                lines.append(ys + np.random.randn(1)*s_samp[i] - 3) #FIXME: opt
        plt.plot(xs, model1(pars, xs)-3, ".2", linewidth=1)
        plt.errorbar(x, y-3, xerr=xerr, yerr=xerr, fmt="k.", capsize=0,
                             alpha=.5, ecolor=".5", mec=".2")

        quantiles = np.percentile(lines, [16, 84], axis=0)
        plt.fill_between(xs, quantiles[0], quantiles[1], color=col,
                         alpha=.5)

        ys = model1(pars, xs)
        if fractional:
            plt.plot(xs, ys + ys * sigma - 3 , "k--")
            plt.plot(xs, ys - ys * sigma - 3, "k--")
        elif extra:
            plt.plot(xs, ys + f * ys + sigma - 3 , "k--")
            plt.plot(xs, ys - f * ys + sigma - 3, "k--")
        else:
            plt.plot(xs, ys + sigma - 3 , "k--")
            plt.plot(xs, ys - sigma - 3, "k--")

    elif whichx == "logg":
        plt.ylim(3, 5)
        col = "#0066CC"
        plt.ylabel("$\log_{10}(g [\mathrm{cm~s}^{-2}])$")
        plt.text(1.6, 4.7, "$\log(g) \sim \mathcal{N} \
                 (\\gamma + \\delta \log_{10}(F_8), \\sigma_g)$")
        plt.text(1.95, 4.52, "$\\gamma = %.3f$" % alpha)
        plt.text(1.95, 4.42, "$\\delta = %.3f$" % beta)
        plt.text(1.95, 4.32, "$\\sigma_g = %.3f$" % sigma)
        lines = []
        for i in range(ndraws):
            ys = model1([b_samp[i], a_samp[i]], xs)
            if fractional:
#                 plt.plot(xs, ys + ys * s_samp[i], col, alpha=.05)
                lines.append(ys + ys*np.random.randn(1)*s_samp[i]) #FIXME: opt
            elif extra:
#                 plt.plot(xs, ys + f_samp[i] * ys + s_samp[i], col,
#                          alpha=.05)
                lines.append(ys*f_samp[i] +
                             np.random.randn(1)*s_samp[i]) #FIXME: opt
            else:
#                 plt.plot(xs, ys + s_samp[i], col, alpha=.05)
#                 plt.plot(xs, ys + np.random.randn(1)*s_samp[i], col, alpha=.05)
                lines.append(ys + np.random.randn(1)*s_samp[i]) #FIXME: opt


        quantiles = np.percentile(lines, [16, 84], axis=0)
        plt.fill_between(xs, quantiles[0], quantiles[1], color=col,
                         alpha=.5)

        plt.plot(xs, model1(pars, xs), ".2", linewidth=1)
        plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="k.", capsize=0,
                     alpha=.5, ecolor=".5", mec=".2")
        ys = model1(pars, xs)

        if fractional:
            plt.plot(xs, ys + ys*sigma, "k--")
            plt.plot(xs, ys - ys*sigma, "k--")
        elif extra:
            plt.plot(xs, ys + (f * ys + sigma), "k--")
            plt.plot(xs, ys - (f * ys + sigma), "k--")
        else:
            plt.plot(xs, ys + sigma, "k--")
            plt.plot(xs, ys - sigma, "k--")

        A = np.vander(xs, 2)
        lines = np.dot(samples[:, :2], A.T)
        quantiles = np.percentile(lines, [16, 84], axis=0)
        plt.fill_between(xs, quantiles[0], quantiles[1], color="#8d44ad",
                         alpha=.5)
    plt.subplots_adjust(bottom=.1)

    plt.xlim(1, 2.4)
    plt.xlabel("$\log_{10}\mathrm{(F}_8~\mathrm{[ppm]})$")
    print "..figs/%s_vs_flicker_%s.pdf" % (whichx, fname)
    plt.savefig("../figs/%s_vs_flicker_%s.pdf" % (whichx, fname))
    plt.savefig("flicker_inv_%s_%s" % (whichx, fname))

    plt.clf()
    plt.ylim(4.6, 3.2)
    x -= 3
    xs = np.linspace(min(x), max(x), 100)
    plt.plot(10**x, y, "ko")
    ys = 1.15136-3.59637*xs-1.40002*xs**2-.22993*xs**3
    plt.plot(10**xs, ys, "m")
    plt.savefig("bastien_figureS2")

if __name__ == "__main__":

    plotpar = {'axes.labelsize': 18,
               'text.fontsize': 26,
               'legend.fontsize': 18,
               'xtick.labelsize': 18,
               'ytick.labelsize': 18,
               'text.usetex': True}
    plt.rcParams.update(plotpar)

    whichx = str(sys.argv[1]) # should be either "rho" or "logg"
    fname = str(sys.argv[2]) # mixture, f_extra, f, test, simple

    x, y, xerr, yerr = load_data(whichx, bigdata=True)

    # load chains
    with h5py.File("%s_samples_%s.h5" % (whichx, fname), "r") as f:
        samples = f["samples"][...]
    samples = samples.T
    print np.shape(samples)

    fractional, extra = False, False
    if fname == "f":
       fractional = True
    elif fname == "f_extra":
       extra = True
    make_inverse_flicker_plot(x, xerr, y, yerr, samples, whichx, fname, 500,
                              fractional=fractional, extra=extra)
