# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import corner
import h5py

plotpar = {'axes.labelsize': 18,
           'text.fontsize': 14,
           'legend.fontsize': 18,
           'xtick.labelsize': 18,
           'ytick.labelsize': 18,
           'text.usetex': True}
plt.rcParams.update(plotpar)

# load data
f, ferr, r, rerr = np.genfromtxt("../data/flickers.dat").T
r -= 3  # convert rho to g/cm^3

pink = "#FF33CC"
blue = "#0066CC"

alpha, ap, am = c2
beta, bp, bm = m2
sigma, sp, sm = sig2
ys = m2[0] * xs + c2[0]
resids = f - m2[0] * r + c2[0]
plt.fill_between(xs, ys-2*np.std(resids)-2*sig2[0], ys+2*np.std(resids)+2*sig2[0], color=pink, alpha=.1)
plt.fill_between(xs, ys-np.std(resids)-sig2[0], ys+np.std(resids)+sig2[0], color=pink, alpha=.3)
plt.errorbar(r, f, xerr=rerr, yerr=ferr, fmt="k.", capsize=0, ecolor=".5", mec=".2", alpha=.5)
plt.plot(xs, ys, "k")
plt.plot(xs, ys-np.std(resids), color=pink)
plt.plot(xs, ys+np.std(resids), color=pink)
plt.ylabel("$\log_{10}(F_8)$")
plt.xlabel("$\log_{10}(\\rho_{\star}[\mathrm{g~cm}^{-3}])$")
#plt.text(-1.15, 2.25, "$\log_{10} (\\rho_{\star}) \sim \mathcal{N} \left(\\alpha_\\rho + \\beta_\\rho \log_{10}(F_8), \
#                \\sigma=\\sigma_{\\rho}\\right)$")
plt.text(-1.15, 2.25, "$\log_{10} (F_8) \sim \mathcal{N} \left(\\alpha_\\rho + \\beta_\\rho \log_{10}(\\rho_{\star}),                 \\sigma=\\sigma_{\\rho}\\right)$")
plt.text(-1.45, 1.2, "$\\alpha_\\rho = %.2f \pm %.2f$" % (np.round(alpha, 2), np.round(ap, 2)))
plt.text(-1.45, 1., "$\\beta_\\rho = %.2f \pm %.2f $" % (np.round(beta, 2), np.round(bp, 2)))
plt.text(-1.45, .8, "$\\sigma_{\\rho} = %.3f \pm %.3f $" % (np.round(sigma, 2), np.round(sp, 3)))
plt.xlim(min(r), max(r))
plt.ylim(.4, 2.5)
plt.savefig("../version1.0/flicker_vs_rho.pdf")

# load data
f, ferr, l, lerr, _, _ = np.genfromtxt("../data/log.dat").T

alpha, ap, am = c2
beta, bp, bm = m2
sigma, sp, sm = sig2
ys = m2[0] * xs + c2[0]
resids = f - m2[0] * l + c2[0]
plt.fill_between(xs, ys-2*np.std(resids)-2*sig2[0], ys+2*np.std(resids)+2*sig2[0], color=blue, alpha=.1)
plt.fill_between(xs, ys-np.std(resids)-sig2[0], ys+np.std(resids)+sig2[0], color=blue, alpha=.3)
plt.errorbar(l, f, xerr=lerr, yerr=ferr, fmt="k.", capsize=0, ecolor=".5", mec=".2", alpha=.5)
plt.plot(xs, ys, "k")
plt.plot(xs, ys-np.std(resids), color=blue)
plt.plot(xs, ys+np.std(resids), color=blue)
plt.ylabel("$\log_{10}(F_8)$")
plt.xlabel("$\log_{10}(g [\mathrm{cm~s}^{-2}])$")
plt.text(3.55, 2.25, "$\log_{10} (F_8) \sim \mathcal{N} \left(\\alpha_g + \\beta_g \log_{10}(g),                 \\sigma=\\sigma_g\\right)$")
plt.text(3.45, 1.2, "$\\alpha_g = %.2f \pm %.2f$" % (np.round(alpha, 2), np.round(ap, 2)))
plt.text(3.45, 1., "$\\beta_g = %.2f \pm %.2f $" % (np.round(beta, 2), np.round(bp, 2)))
plt.text(3.45, .8, "$\\sigma_g = %.3f \pm %.3f $" % (np.round(sigma, 2), np.round(sp, 3)))
plt.xlim(min(l), max(l))
plt.ylim(.4, 2.5)
plt.savefig("../version1.0/flicker_vs_logg.pdf")

# save samples
fi = h5py.File("logg_samples.h5", "w")
data = fi.create_dataset("samples", np.shape(sampler.chain))
data[:, :] = np.array(sampler.chain)
fi.close()
