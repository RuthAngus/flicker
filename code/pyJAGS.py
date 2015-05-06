import numpy as np
import matplotlib.pyplot as plt
import triangle
from params import plot_params
reb = plot_params()
from colours import plot_colours
cols = plot_colours()
import subprocess
import sys
import triangle

def make_data_file(obs_vals, obs_names, name):
    '''
    obs_vals = array of observable values. Shape = (nobs, ndim)
    obs_names = list of strings of observable names
    name = string, name of model (e.g. "flicker")
    '''
    assert np.shape(obs_vals)[0] > np.shape(obs_vals)[1], \
            "obs_vals is the wrong shape. Try transposing"
    orig_stdout = sys.stdout
    f = file('%s-data.R' % name, 'w')
    sys.stdout = f
    for i in range(len(obs_names)):
        print '"%s" <-' % obs_names[i]
        print 'c(%s)' % str(list(obs_vals[:, i]))[1:-1]
    print '"N" <-'
    print len(obs_vals)
    sys.stdout = orig_stdout
    f.close()

def make_init_file(par_names, inits, name):
    '''
    par_names = list of strings of parameter names
    inits = list of initial values
    '''
    orig_stdout = sys.stdout
    f = file('%s-inits.R' % name, 'w')
    sys.stdout = f
    for i in range(len(par_names)):
        print '"%s" <-' % par_names[i], inits[i]
    sys.stdout = orig_stdout
    f.close()

def make_execute_files(name, par_names, obs_names, nchains=2, nsteps=1000,
                 niter=10000):
    assert len(par_names) == 3

    # make R file
    orig_stdout = sys.stdout
    f = file('test.R', 'w')
    sys.stdout = f
    print 'source("../../R/Rcheck.R")'
    print 'source("line-data.R")'
    print 'm <- jags.model("%s.bug", n.chains=%s)' % (name, nchains)
    print 'update(m, %s)' % nsteps
    print '%s <- coda.samples(m, c("%s","%s","%s"), n.iter=%s' \
            % (obs_names[0], par_names[0], par_names[1], par_names[2], niter)
    print 'source("bench-test.R")'
    print 'check.fun()'
    sys.stdout = orig_stdout
    f.close()

    # make cmd file
    orig_stdout = sys.stdout
    f = file('test.cmd', 'w')
    sys.stdout = f
    print 'model in %s.bug' % name
    print 'data in %s-data.R' % name
    print 'compile, nchains(%s)' % nchains
    print 'inits in %s-inits.R' % name
    print 'initialize'
    print 'update %s' % nsteps
    for i in range(len(par_names)):
        print 'monitor set %s' % par_names[i]
    print 'update %s' % niter
    print 'coda *'
    sys.stdout = orig_stdout
    f.close()

def runJAGS(obs_vals, obs_names, par_names, inits, name):
    make_data_file(obs_vals, obs_names, name)
    make_init_file(par_names, inits, name)
    make_execute_files(name, par_names, obs_names)
    subprocess.call("JAGS test.cmd", shell=True)

def plot_results():
    n1, c1 = np.genfromtxt("CODAchain1.txt").T
    n2, c2 = np.genfromtxt("CODAchain2.txt").T

    chain1 = np.concatenate((c1[1:10000], c2[1:10000]))
    chain2 = np.concatenate((c1[10001:20000], c2[10001:20000]))
    chain3 = np.concatenate((c1[20001:30000], c2[20001:30000]))
    pars = [np.median(chain1), np.median(chain2), np.median(chain3)]
    print pars

    samples = np.vstack((chain1, chain2, chain3)).T
    fig = triangle.corner(samples)
    fig.savefig("triangle_test")


if __name__ == "__main__":

    f8, f8_err, rho, rho_err = np.genfromtxt("flickers.dat").T

    obs_vals = np.vstack((rho, rho_err, f8, f8_err)).T
    obs_names = ["xobs", "xerr", "Yobs", "Yerr"]
    par_names = ["alpha", "beta", "tau"]
    inits = [0, 0, 1]
    name = "flicker"
    runJAGS(obs_vals, obs_names, par_names, inits, name)
    plot_results()
