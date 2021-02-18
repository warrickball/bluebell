#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import ticker
import bluebell as bb
import bluebell.plot as bbplot

# custom linestyle
dashed = (0,(5,5))
dotted = ':'

pl.style.use('paper.mpl')

np.random.seed(20)
mu = np.array([1.,1.])
C = np.array([[1., 0.5], [0.5, 1.]])
x = np.random.uniform(low=mu-2, high=mu+2, size=(20,2))

# sample doesn't include optimum
# (but presumably something close)
x0 = mu + np.array([0.05, -0.05])
chi20 = (x0-mu)@np.linalg.inv(C)@(x0-mu)
chi2 = np.array([xi@np.linalg.inv(C)@xi for xi in x-mu])
xx = x0 + (x-x0)/np.sqrt(chi2-chi20)[:,None]
dx = xx-x

bbplot.cov_ellipse(mu, C, lw=1.5, fc='none', ec='k');
pl.plot([np.nan, np.nan], [np.nan, np.nan], 'k-', lw=1.5, label=r"$\Delta\chi^2=1$")

for xi, dxi in zip(x, dx):
    pl.arrow(*(xi+0.05*dxi), *(0.9*dxi),
             width=0.01, head_width=0.05,
             length_includes_head=True, fc='k', ec='k')

pl.scatter(*x.T, c=np.sqrt(chi2), s=50, label=r"$\bm{x}$", marker='o');
pl.scatter(*xx.T, c=np.sqrt(chi2), s=50, label=r"$\bm{x}'$", marker='s');

mu_fit, C_fit = bb.MVBE(xx, tol=1e-9)
bbplot.cov_ellipse(mu_fit, C_fit, lw=1.5, fc='none', ec='C3', ls=dotted);
pl.plot([np.nan, np.nan], [np.nan, np.nan], 'C3', ls=dotted, lw=1.5, label='MVBE (initial)')

mu_fit, C_fit = bb.MVBE(xx[chi2 > chi20+0.25**2], tol=1e-9)
bbplot.cov_ellipse(mu_fit, C_fit, lw=1.5, fc='none', ec='C1', ls=dashed);
pl.plot([np.nan, np.nan], [np.nan, np.nan], 'C1', ls=(0,(5,5)), lw=1.5, label='MVBE ($\Delta\chi^2\\!>\\!0.25^2$)')

pl.xlim(mu[0]-2, mu[1]+2)
pl.ylim(mu[0]-2, mu[1]+2)
pl.legend(ncol=2, borderpad=0.2, labelspacing=0.2, handlelength=1.2,
          handletextpad=0.4, columnspacing=0.2, loc='upper left', bbox_to_anchor=(-0.01,1.01))
pl.xlabel(r"$x_1$")
pl.ylabel(r"$x_2$")

pl.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
pl.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))

pl.savefig('3-no_opt.pdf')
pl.clf()
