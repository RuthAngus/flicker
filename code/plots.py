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

def make_flicker_plot(x, xerr, y, yerr, samples, whichx, fname, ndraws,
                          fractional=False, extra=False):

    # use highest likelihood samples
    lls = samples[:, -1]
    m = lls == max(lls)

    if fname == "simple":
        beta, alpha, tau = [samples[i, m]
                               for i in range(np.shape(samples)[0]-1)]
    else:
        beta, alpha, tau, f = [samples[i, m]
                           for i in range(np.shape(samples)[0]-1)]
#     sigma = abs(tau)**.5
    sigma = tau
    print "parameters:"
    bp = np.percentile(samples[0, :], [16, 50, 84])
    print "beta =", beta, "-", bp[1]-bp[0], "+", bp[2]-bp[1]
    ap = np.percentile(samples[1, :], [16, 50, 84])
    print "alpha =", alpha, "-", ap[1]-ap[0], "+", ap[2]-ap[1]
    tp = np.percentile(samples[1, :], [16, 50, 84])
    print "tau =", tau, "-", tp[1]-tp[0], "+", tp[2]-tp[1]
    if fname != "simple":
        fp = np.percentile(samples[1, :], [16, 50, 84])
        print "f =", f, "-", tp[1]-tp[0], "+", tp[2]-tp[1]

    # draw samples from post
    b_samp = np.random.choice(samples[0, :], ndraws)
    a_samp = np.random.choice(samples[1, :], ndraws)
    s_samp = np.random.choice(samples[2, :], ndraws)
    f_samp = np.random.choice(samples[3, :], ndraws)

    # rho plot
    plt.clf()
    xs = np.linspace(.7, 2.5, 100)
    if whichx == "rho":
        plt.ylabel("$\log_{10}(\\rho_{\star}[\mathrm{g~cm}^{-3}])$")
        col = "#FF33CC"
        plt.text(1.4, .5, "$\log_{10} (\\rho_{\star}) \sim \mathcal{N} \
                \\left(\\alpha + \\beta \log_{10}(F_8), \
                \\sigma=\sqrt{\\sigma_{\\rho}^2+\\gamma F_8}\\right)$")
        plt.text(1.95, .24, "$\\alpha = %.3f$" % (alpha-3))
        plt.text(1.95, .09, "$\\beta = %.3f$" % beta)
        plt.text(1.95, -.06, "$\\sigma_{\\rho} = %.3f$" % sigma)
        plt.text(1.95, -.21, "$\\gamma = %.3f$" % f)
        plt.ylim(-2, 1)

        # plot line draws
        for i in range(ndraws):
            ys = model1([b_samp[i], a_samp[i]], xs)
            y3 = ys - 3
            line = ys + (np.random.randn(1)*np.median(s_samp)**2 + \
                    np.random.randn(1)*np.median(f_samp)*xs)**.5 - 3
            plt.plot(xs, line, col, alpha=.05)

        # plot best fit and data
        ym = model1([np.median(b_samp), np.median(a_samp)], xs)
        plt.plot(xs, ym - 3, ".2", linewidth=1)
        plt.xlim(min(xs), max(xs))
        plt.errorbar(x, y - 3, xerr=xerr, yerr=xerr, fmt="k.", capsize=0,
                             alpha=.5, ecolor=".5", mec=".2")
        plt.savefig("new_rho")

    # logg plot
    elif whichx == "logg":
        plt.ylim(3, 5)
        col = "#0066CC"
        plt.text(1.5, 4.7, "$\log(g) \sim \mathcal{N} \
                 \\left(\\delta + \\epsilon \log_{10}(\\rho_{\star}), \
                 \\sigma=\sqrt{\\sigma_g^2 + \zeta F_8}\\right)$")
        plt.text(1.95, 4.52, "$\\delta = %.3f$" % alpha)
        plt.text(1.95, 4.42, "$\\epsilon = %.3f$" % beta)
        plt.text(1.95, 4.32, "$\\sigma_g = %.3f$" % sigma)
        plt.text(1.95, 4.22, "$\\zeta = %.3f$" % f)
        plt.ylabel("$\log_{10}(g [\mathrm{cm~s}^{-2}])$")

        # plot line draws
#         lines = np.zeros((ndraws, len(xs)))
        lines = []
        for i in range(ndraws):
            ys = model1([b_samp[i], a_samp[i]], xs)
            line = ys + (np.random.randn(1)*np.median(s_samp)**2 + \
                    np.random.randn(1)*np.median(f_samp)*xs)**.5
            plt.plot(xs, line, col, alpha=.05)
#             lines[i, :] = line
            lines.append(line)

        plt.xlim(min(xs), max(xs))
        # plot regions
        quantiles = np.percentile(lines, [2, 16, 84, 98], axis=0)
        plt.fill_between(xs, quantiles[1], quantiles[2], color=col,
                         alpha=.4)
        plt.fill_between(xs, quantiles[0], quantiles[3], color=col,
                         alpha=.2)

        # plot best fit and data
        ym = model1([np.median(b_samp), np.median(a_samp)], xs)
        plt.plot(xs, ym, ".2", linewidth=1)
        plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="k.", capsize=0,
                     alpha=.5, ecolor=".5", mec=".2")
        plt.savefig("new_logg")

def make_inverse_flicker_plot(x, xerr, y, yerr, samples, whichx, fname, ndraws,
                              fractional=False, extra=False):

    # fit straight line
    lim = 200
    a1 = fit_straight_line(x[:lim], y[:lim], yerr[:lim])
    a2 = fit_straight_line(x[lim:], y[lim:], yerr[lim:])

    assert np.shape(samples)[0] < np.shape(samples)[1], \
            "samples is wrong shape"

    m = samples[0, :] < 0
    samples = samples[:, m]

    # use highest likelihood samples
    lls = samples[:, -1]
    m = lls == max(lls)
    if extra:
        beta, alpha, tau, f = \
                [samples[i, m] for i in range(np.shape(samples)[0]-1)]
    else:
        beta, alpha, tau = \
                [samples[i, m] for i in range(np.shape(samples)[0]-1)]
    sigma = abs(tau)**.5
    if fname == "simple":
        sigma = tau
    pars = [beta, alpha, sigma]

#     # take medians
#     results = np.median(samples, axis=1)
#     print 'results = ', results
#     beta, alpha, tau = results[:3]
#     if extra:
#         print np.shape(results), "shape"
#         beta, alpha, tau, f = results[:4]

    print alpha, beta, tau, sigma

    b_samp = np.random.choice(samples[0, :], ndraws)
    a_samp = np.random.choice(samples[1, :], ndraws)
    t_samp = np.random.choice(samples[2, :], ndraws)
    if fname == "f":
        s_samp = (abs(t_samp)**.5 - 1) * np.random.randn(ndraws) + 1
    if fname == "simple":
        s_samp = t_samp
    if fname == "test":
        s_samp = (abs(t_samp)**.5) * np.random.randn(ndraws)
    if extra:
        s_samp = (abs(t_samp)**.5 - 1) * np.random.randn(ndraws) + 1
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
        ym = model1([np.median(b_samp), np.median(a_samp)], xs)
        for i in range(ndraws):
            ys = model1([b_samp[i], a_samp[i]], xs)
            y3 = ys - 3
#             if fractional:
#                 plt.plot(xs, ym + ym * np.random.randn(1)*sigma - 3 ,
#                         "b", alpha=.05)
#                 lines.append(ys-3 + ys * (np.random.randn(1)*s_samp[i] - 1))
#                 lines.append(ym + ym * np.random.randn(1)*(sigma-1) - 3)
            if extra:
                line = y3 + np.random.randn(1)*np.median(s_samp-1) + \
                        np.random.randn(1)*np.median(f_samp)*xs
                plt.plot(xs, line, col, alpha=.05)
                lines.append(line)
            else:
#                 plt.plot(xs, ys + s_samp[i] - 3, col, alpha=.05)
#                 plt.plot(xs, ys + np.random.randn(1)*s_samp[i] - 3, col,
#                          alpha=.05)
               lines.append(ys + np.random.randn(1)*s_samp[i] - 3) #FIXME: opt
        plt.plot(xs, model1([np.median(b_samp), np.median(a_samp)], xs)-3, ".2", linewidth=1)
        plt.errorbar(x, y-3, xerr=xerr, yerr=xerr, fmt="k.", capsize=0,
                             alpha=.5, ecolor=".5", mec=".2")

        quantiles = np.percentile(lines, [2, 16, 84, 98], axis=0)
        plt.fill_between(xs, quantiles[1], quantiles[2], color=col,
                         alpha=.4)
        plt.fill_between(xs, quantiles[0], quantiles[3], color=col,
                         alpha=.2)

#         ys = model1(pars, xs)
#         if fractional:
#             plt.plot(xs, ys + ys * sigma - 3 , "k--")
#             plt.plot(xs, ys - ys * sigma - 3, "k--")
#         elif extra:
#             plt.plot(xs, ys + f * ys + sigma - 3 , "k--")
#             plt.plot(xs, ys - f * ys + sigma - 3, "k--")
#         else:
#             plt.plot(xs, ys + sigma - 3 , "k--")
#             plt.plot(xs, ys - sigma - 3, "k--")

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
        ym = model1([np.median(b_samp), np.median(a_samp)], xs)
        for i in range(ndraws):
            ys = model1([b_samp[i], a_samp[i]], xs)
#             if fractional:
#                 plt.plot(xs, ym + ym * np.random.randn(1)*sigma - 3 ,
#                         "b", alpha=.05)
#                 lines.append(ys + ys*np.random.randn(1)*s_samp[i])
            if extra:
                plt.plot(xs, ys + np.random.randn(1)*np.median(s_samp-1) +
                        np.random.randn(1)*np.median(f_samp)*xs, col, alpha=.05)
#                 plt.plot(xs, ys + np.median(f_samp-1)*np.random.randn(1) * \
#                         ys + np.median(s_samp-1)*np.random.randn(1), col, alpha=.05)
#                 plt.plot(xs, ys + np.median(f_samp)*np.random.randn(1)*ys
#                          + np.median(s_samp-1)*np.random.randn(1),
#                          col, alpha=.05)
#                 lines.append(ys + ys*np.median(f_samp)*np.random.randn(1))
                lines.append(ys + ys*np.median(f_samp)*np.random.randn(1)
                             + np.random.randn(1)*np.median(s_samp-1))
            else:
#                 plt.plot(xs, ys + s_samp[i], col, alpha=.05)
#                 plt.plot(xs, ys + np.random.randn(1)*s_samp[i], col, alpha=.05)
                lines.append(ys + np.random.randn(1)*s_samp[i])

        plt.plot(xs, model1(pars, xs), ".2", linewidth=1)
        plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="k.", capsize=0,
                     alpha=.5, ecolor=".5", mec=".2")
        ys = model1(pars, xs)

        quantiles = np.percentile(lines, [2, 16, 84, 98], axis=0)
        plt.fill_between(xs, quantiles[1], quantiles[2], color=col,
                         alpha=.4)
        plt.fill_between(xs, quantiles[0], quantiles[3], color=col,
                         alpha=.2)

#         if fractional:
#             plt.plot(xs, ys + ys*sigma, "k--")
#             plt.plot(xs, ys - ys*sigma, "k--")
#         elif extra:
#             plt.plot(xs, ys + (f * ys + sigma), "k--")
#             plt.plot(xs, ys - (f * ys + sigma), "k--")
#         else:
#             plt.plot(xs, ys + sigma, "k--")
#             plt.plot(xs, ys - sigma, "k--")

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

#     x, y, xerr, yerr = load_data(whichx, bigdata=True)
    x, y, xerr, yerr = load_data(whichx, bigdata=False)

    # load chains
    with h5py.File("%s_samples_%s.h5" % (whichx, fname), "r") as f:
        samples = f["samples"][...]
    samples = samples.T

    fractional, extra = False, False
    if fname == "f":
       fractional = True
    elif fname == "f_extra" or "short":
       extra = True
    make_flicker_plot(x, xerr, y, yerr, samples, whichx, fname, 1000,
                              fractional=fractional, extra=extra)
#     make_inverse_flicker_plot(x, xerr, y, yerr, samples, whichx, fname, 1000,
#                               fractional=fractional, extra=extra)
