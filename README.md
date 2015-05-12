#
flicker
HBM with flicker

Flicker.bug contains the model shown below.

![pgm](https://github.com/RuthAngus/flicker/blob/master/pgm.png)

`main.py` - runs MCMC on data with hierarchical model and 2d uncertainties

continue by fixing the `generate_samples_log` function in `noisy_plane.py`

`simpleMCMC.py` - uncertainties on 1d, HBM with density
`MCMClogg.py` - uncertainties on 1d, HBM with logg
`logg2d.py` - uncertainties in 2d, non-HBM with logg
`rho2d.py` - uncertainties in 2d, non-HBM with rho

`plots.py` - creates main flicker plot using mcmc samples
