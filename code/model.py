import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as spo

def load_data(whichx, nd=0, bigdata=True, huber=False):

    if bigdata:
        data = np.genfromtxt("../data/BIGDATA.nohuber.filtered.dat").T
        if huber:
            data = np.genfromtxt("../data/BIGDATA.filtered.dat").T
        r, rerrp, rerrm = data[7:10]
        rerrp, rerrm = rerrp/r/np.log(10), rerrm/r/np.log(10)
        r = np.log10(1000*r)
        data2 = np.genfromtxt("../data/BIGDATA.nohuber.filtered.dat").T
        logg, loggerrp, loggerrm = data2[10:13]
        f, ferr = data2[20:22]
        if whichx == "rho":
            f, ferr = data[20:22]
        ferrp, ferrm = ferr/f/np.log(10), ferr/f/np.log(10)
        f = np.log10(1000*f)
        # format data
        if nd == 0:  # parameter for using less data (zero for use all)
            nd = len(f)
        m = np.isfinite(logg[:nd])
        x = f[:nd][m]-3
        xerrp, xerrm = ferrp[:nd][m] * x, ferrm[:nd][m] * x
        y = logg[:nd][m]
        yerrp, yerrm = loggerrp[:nd][m] * y, loggerrm[:nd][m] * y
        xerr = .5*(xerrp + xerrm)
        yerr = .5*(yerrp + yerrm)
        if whichx == "rho":
            m = np.isfinite(r[:nd])
            y = r[:nd][m]
            yerrp, yerrm = rerrp[:nd][m] * y, rerrm[:nd][m] * y
            xerr = .5*(xerrp + xerrm)
            yerr = .5*(yerrp + yerrm)

    else:
        fr, frerr, r, rerr = np.genfromtxt("../data/flickers.dat").T
        fl, flerr, l, lerr, t, terr = np.genfromtxt("../data/log.dat").T
        if nd==0:
            nd = len(fr)
        x, xerr, y, yerr = fl[:nd], flerr[:nd], l[:nd], lerr[:nd]
        if whichx == "rho":
            x, xerr, y, yerr = fr[:nd], frerr[:nd], r[:nd], rerr[:nd]

    m = np.argsort(x)
    return x[m], y[m], xerr[m], yerr[m]


def model1(pars, x):
    return pars[0]*x + pars[1]

def model(pars, samples):
    xs = samples[0, :, :]
    x = np.reshape(xs, np.shape(xs)[0]* np.shape(xs)[1])
    return pars[0]*x + pars[1]

def broken_power_law1(pars, x):
    y = np.zeros_like(x)
    m = x < pars[0]
    y[m] = pars[1] + pars[2] * x[m]
    y[~m] = pars[3] + pars[4] * x[~m]
    return y

def polynomial(pars, x):
    return pars[0] + pars[1]*x + pars[2]*x**2 + pars[3]*x**3

def broken_power_law():
    xs = samples[0, :, :]
    x = np.reshape(xs, np.shape(xs)[0]* np.shape(xs)[1])
    y = np.zeros_like(x)
    m = x < pars[0]
    y[m] = pars[1] + pars[2] * x[m]
    y[~m] = pars[3] + pars[4] * x[~m]
    return y

def nll(pars, x, y, yerr, model):
    inv_sig2 = 1./yerr**2
    mod = model(pars, x)
    chi2 = -.5 * ((y-mod)**2 * inv_sig2)
    return - np.logaddexp.reduce(chi2, axis=0)

if __name__ == "__main__":

    x, y, xerr, yerr = load_data("logg")

    pars_init = [1.8, 5.6, -1, 5.95, -1.2]
    res = spo.minimize(nll, pars_init, args=(x, y, yerr, broken_power_law1))
    print res.x

    plt.clf()
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="k.", capsize=0, ecolor=".7",
                 label="bigdata")

    plt.xlim(1., 2.4)
    plt.ylim(3., 5.)
    plt.plot(x, broken_power_law1(res.x, x), "m")

    pars_init = [5.6, -1]
    res = spo.minimize(nll, pars_init, args=(x, y, yerr, model1))
    print res.x
    plt.plot(x, model1(res.x, x), "y")

    pars_init = [5.6, -1, .5, .5]
    res = spo.minimize(nll, pars_init, args=(x, y, yerr, polynomial))
    print res.x
    plt.plot(x, polynomial(res.x, x), "b")
    plt.savefig("model.pdf")
