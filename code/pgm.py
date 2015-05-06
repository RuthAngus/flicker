from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft

# Instantiate the PGM.
pgm = daft.PGM([3.2, 4], origin=[0.3, 0.3])

# Data.
pgm.add_node(daft.Node("rho_obs", r"$\hat{\rho_n}$", 1, 1, observed=True))
pgm.add_node(daft.Node("f_obs", r"$\hat{F8_n}$", 2, 1, observed=True))

# Latent variables.
pgm.add_node(daft.Node("rho", r"$\rho_n$", 1, 1.8))
pgm.add_node(daft.Node("f", r"$F8_n$", 2, 1.8))
pgm.add_node(daft.Node("mean", r"$\mu_n$", 1.5, 2.5))

# add the plate.
pgm.add_plate(daft.Plate([0.5, 0.5, 2, 2.5], label=r"$n = 1, \cdots, N$",
    shift=-0.1))

# Model parameters
pgm.add_node(daft.Node("alpha", r"$\alpha$", 1, 3.5))
pgm.add_node(daft.Node("beta", r"$\beta$", 2, 3.5))
pgm.add_node(daft.Node("tau", r"$\tau$", 3, 1.8))

# # Add in the edges.
pgm.add_edge("rho_obs", "rho")
pgm.add_edge("rho", "mean")
pgm.add_edge("mean", "f")
pgm.add_edge("f", "f_obs")
pgm.add_edge("alpha", "mean")
pgm.add_edge("beta", "mean")
pgm.add_edge("tau", "f")

# Render and save.
pgm.render()
pgm.figure.savefig("pgm.png", dpi=150)
