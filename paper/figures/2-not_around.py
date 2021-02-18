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

# sample doesn't surround optimum
chi2 = np.array([xi@np.linalg.inv(C)@xi for xi in x-mu])
xx = mu + (x-mu)/np.sqrt(chi2)[:,None]
dx = xx-x
I = x[:,0] < mu[0]

bbplot.cov_ellipse(mu, C, lw=1.5, fc='none', ec='k');
pl.plot([np.nan, np.nan], [np.nan, np.nan], 'k-', lw=1.5, label=r"$\Delta\chi^2=1$")

for xi, dxi in zip(x[I], dx[I]):
    pl.arrow(*(xi+0.05*dxi), *(0.9*dxi),
             width=0.01, head_width=0.05,
             length_includes_head=True, fc='k', ec='k')

pl.scatter(*x[I].T, c=np.sqrt(chi2[I]), s=50, label=r"$\bm{x}$", marker='o',
           vmin=np.sqrt(chi2.min()), vmax=np.sqrt(chi2.max()));
pl.scatter(*xx[I].T, c=np.sqrt(chi2[I]), s=50, label=r"$\bm{x}'$", marker='s',
           vmin=np.sqrt(chi2.min()), vmax=np.sqrt(chi2.max()));

mu_fit, C_fit = bb.MVBE(np.vstack([mu, xx[I]]), tol=1e-9)
bbplot.cov_ellipse(mu_fit, C_fit, lw=1.5, fc='none', ec='C3', ls=dotted);
pl.plot([np.nan, np.nan], [np.nan, np.nan], 'C3', ls=dotted, lw=1.5, label='MVBE (initial)')

xtra = bb.vertices(mu, C_fit)
chi2 = np.array([xi@np.linalg.inv(C)@xi for xi in xtra-mu])
pl.scatter(*xtra.T, c='C1', s=50, marker='o')
xxtra = mu + (xtra-mu)/np.sqrt(chi2)[:,None]
dxtra = xxtra-xtra

for xi, dxi in zip(xtra, dxtra):
    pl.arrow(*(xi+0.05*dxi), *(0.9*dxi),
             width=0.01, head_width=0.05,
             length_includes_head=True, fc='k', ec='k')

pl.scatter(*xxtra.T, c='C1', s=50, marker='s')

mu_fit, C_fit = bb.MVBE(np.vstack([mu, xx[I], xxtra]), tol=1e-9)
bbplot.cov_ellipse(mu_fit, C_fit, lw=1.5, fc='none', ec='C1', ls=(0,(5,5)));
pl.plot([np.nan, np.nan], [np.nan, np.nan], 'C1', ls=(0,(5,5)), lw=1.5, label='MVBE (w/ vertices)')

pl.xlim(mu[0]-2, mu[1]+2)
pl.ylim(mu[0]-2, mu[1]+2)
pl.legend(ncol=2, borderpad=0.2, labelspacing=0.2, handlelength=1.2,
          handletextpad=0.4, columnspacing=0.2, loc='upper left', bbox_to_anchor=(-0.01,1.01))
pl.xlabel(r"$x_1$")
pl.ylabel(r"$x_2$")

pl.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
pl.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))

pl.savefig('2-not_around.pdf')
pl.clf()
