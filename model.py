import numpy as np

def model1(pars, x):
    return pars[0]*x + pars[1]

def model(pars, samples):
    xs = samples[0, :, :]
    x = np.reshape(xs, np.shape(xs)[0]* np.shape(xs)[1])
    return pars[0]*x + pars[1]
