import numpy as np
import matplotlib.pyplot as plt
import h5py

def deterministic_model(f, a, b):
    return a + f * b

def stochastic_model(f_samples, alpha, beta, tau):
#     plt.clf()
#     y = deterministic_model(f_samples, np.median(alpha), np.median(beta))
    y_samples = deterministic_model(f_samples, alpha, beta)
#     plt.hist(y_samples, 50, alpha=.5, color="b")
#     plt.hist(y, 50, alpha=.5, color="r")
    y_samples += np.random.randn(len(y_samples))*tau
#     plt.hist(y_samples, 50, alpha=.5)
#     plt.show()
#     assert 0
    return y_samples

# given a flicker, predict rho_star
def f2y(f, ferrp, ferrm, fname):
    with h5py.File("%s_samples.h5" % fname, "r") as F:
        samples = F["samples"][...]
    alpha = samples[:, 0]
    beta = samples[:, 1]
    tau = samples[:, 2]
    nsamples = len(alpha)

    # sample from gaussian uncertainties
    ferr = .5*(ferrp + ferrm)
    f_samples = f + np.random.randn(nsamples) * ferr

    predicted_value = deterministic_model(f, np.median(alpha), np.median(beta))
    print "predicted = ", predicted_value
    predicted_distribution = stochastic_model(f_samples, alpha, beta, tau)
    return predicted_distribution

if __name__ == "__main__":

    flickers, ferr, r, rerr = np.genfromtxt("data/flickers.dat").T
    for i, f in enumerate(flickers):
        rho = f2y(f, ferr[i], ferr[i], "rho")
        plt.clf()
        plt.hist(rho, 50)
        print "measured = ", r[i]
        print r[i] - predicted_distribution, "diff"
        plt.axvline(r[i], color="r")
        plt.axvline(r[i]+rerr[i], color="k")
        plt.axvline(r[i]-rerr[i], color="k")
        plt.show()
        raw_input('enter')
