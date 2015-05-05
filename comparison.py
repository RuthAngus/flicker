import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
from predictive import stochastic_model

path = "/Users/angusr/Dropbox/FlickerHackDay/kois_comparison"
kids, flicker, ferr = np.genfromtxt("%s/results.flickers.dat" % path).T

for i, fname in enumerate(fnames):
    fl = np.genfromtxt(fname).T[0]
    print fl
    assert 0

# plt.clf()
# plt.hist(rho_samples)
# plt.show()
