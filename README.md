# Flicker
HBM with flicker

The probabilistic graphical model for this project is shown below.

![pgm](https://github.com/RuthAngus/flicker/blob/master/figs/pgm.png)

`simple_flicker.py` - runs MCMC on data with hierarchical model and 2d
uncertainties using the simple version of the likelihood function.
It also samples in log space.

`main.py` - runs MCMC on data with hierarchical model and 2d uncertainties.

`extra.py` - runs MCMC on data with hierarchical model, where the jitter term
depends on y, plus 2d uncertainties.

`mixture.py` - runs MCMC on data with hierarchical mixture model, plus 2d
uncertainties.

`plots.py` - creates main flicker plot using mcmc samples.
