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

# chi^2 is noisy
chi2 = np.array([xi@np.linalg.inv(C)@xi for xi in x-mu])
chi2 += np.random.uniform(low=-0.3, high=0.3, size=len(chi2))
xx = mu + (x-mu)/np.sqrt(chi2)[:,None]
dx = xx-x

bbplot.cov_ellipse(mu, C, lw=1.5, fc='none', ec='k');
pl.plot([np.nan, np.nan], [np.nan, np.nan], 'k-', lw=1.5, label=r"$\Delta\chi^2=1$")

for xi, dxi in zip(x, dx):
    pl.arrow(*(xi+0.05*dxi), *(0.9*dxi),
             width=0.01, head_width=0.05,
             length_includes_head=True, fc='k', ec='k')

pl.scatter(x[:,0], x[:,1], c=np.sqrt(chi2), s=50, label=r"$\bm{x}$", marker='o');
pl.scatter(xx[:,0], xx[:,1], c=np.sqrt(chi2), s=50, label=r"$\bm{x}'$", marker='s');

mu_fit, C_fit = bb.MVBE(xx, tol=1e-9)
bbplot.cov_ellipse(mu_fit, C_fit, lw=1.5, fc='none', ec='C1', ls=dashed);
pl.plot([np.nan, np.nan], [np.nan, np.nan], 'C1', ls=dashed, lw=1.5, label='MVBE')

pl.xlim(mu[0]-2, mu[1]+2)
pl.ylim(mu[0]-2, mu[1]+2)
pl.legend(ncol=2, borderpad=0.2, labelspacing=0.2, handlelength=1.2,
          handletextpad=0.4, columnspacing=0.2, loc='upper left', bbox_to_anchor=(-0.01,1.01))
pl.xlabel(r"$x_1$")
pl.ylabel(r"$x_2$")

pl.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
pl.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))

pl.savefig('4-chi2_noisy.pdf')
pl.clf()
