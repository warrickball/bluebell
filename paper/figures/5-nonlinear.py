#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as pl
import bluebell as bb
import bluebell.plot as bbplot

# custom linestyle
dashed = (0,(5,5))
dotted = ':'

pl.style.use('paper.mpl')

# nonlinear χ² based on Rosenbrock function:
# R(u,v) = (a-u)² + b²(v-u²)²
# so define
# x = [(a-u), (v-u²)]
# μ = [a, a²]
# σ = [1, 1/b]
# then
# χ² = σ¯²(x-μ)'(x-μ) = R(u,v)

np.random.seed(51)
a = 0.5
b = 5.0
mu = np.array([a, a**2])
sigma = np.array([1.0, 1.0/b])
xmin, xmax = mu[0]-2.5, mu[0]+2.5
ymin, ymax = mu[1]-2.5, mu[1]+2.5
# x = np.random.uniform(low=(xmin, ymin),
#                       high=(xmax, ymax),
#                       size=(20,2))
x = np.zeros((20,2))
# force some points to be outside
for i in range(10):
    xnew = np.array([0, 0])
    while (a-xnew[0])**2 + (xnew[1]-xnew[0])**2/b**2 <= 1:
        xnew = np.random.uniform(low=(xmin, ymin), high=(xmax, ymax))

    x[i] = xnew
# force some points to be inside
for i in range(10,20):
    xnew = np.array([10, 10])
    while (a-xnew[0])**2 + (xnew[1]-xnew[0])**2/b**2 > 1:
        xnew = np.random.uniform(low=(xmin, ymin), high=(xmax, ymax))

    x[i] = xnew
    
y = np.vstack([a-x[:,0], x[:,1]-x[:,0]**2]).T
chi2 = np.sum(y**2/sigma**2, axis=1)
xx = mu + (x-mu)/np.sqrt(chi2)[:,None]
dx = xx-x

mu_fit, C_fit = bb.MVBE(xx, tol=1e-9)

# add vertices, which doesn't actually help
# xtra = bb.vertices(mu, C_fit)
# y = np.vstack([a-xtra[:,0], xtra[:,1]-xtra[:,0]**2]).T
# chi2 = np.sum(y**2/sigma**2, axis=1)
# pl.scatter(xtra[:,0], xtra[:,1], c='r', s=50)
# xxtra = mu + (xtra-mu)/np.sqrt(chi2)[:,None]
# dxtra = xxtra-xtra

# for xi, dxi in zip(xtra, dxtra):
#     pl.arrow(*(xi+0.05*dxi), *(0.9*dxi),
#              width=0.01, head_width=0.05,
#              length_includes_head=True, fc='k', ec='k')

# pl.scatter(xxtra[:,0], xxtra[:,1], c='r')

# mu_fit, C_fit = bb.MVBE(np.vstack([mu, xx[I], xxtra]), tol=1e-9)
# bbplot.cov_ellipse(mu_fit, C_fit, lw=1.5, ls='--', ec='r');

# plot contours
u, v = np.ogrid[xmin-1:xmax+1:401j,ymin-1:ymax+1:401j]
R = (a-u)**2+b**2*(v-u**2)**2
imax = 7 # int(np.max(R**0.25))
vmin = 0
vmax = (imax-2)**2

pl.contour(u[:,0], v[0], R.T**0.5, alpha=0.25, vmin=vmin, vmax=vmax,
           levels=[i**2 for i in range(1,imax+5)]);

for xi, dxi in zip(x, dx):
    dir = dxi/np.linalg.norm(dxi)
    pl.arrow(*(xi+0.05*dir), *(dxi-0.1*dir),
             width=0.01, head_width=0.05,
             length_includes_head=True, fc='k', ec='k')

kwargs = {'vmin': vmin, 'vmax': vmax, 's': 20, 'c': np.sqrt(chi2)}
pl.scatter(*x.T, marker='o', label=r"$\bm{x}$", **kwargs)
pl.scatter(*xx.T, marker='s', label=r"$\bm{x}'$", **kwargs);

bbplot.cov_ellipse(mu_fit, C_fit, lw=1.5, ec='C1', fc='none', ls=dashed);
pl.plot([np.nan, np.nan], [np.nan, np.nan], 'C1', ls=dashed, lw=1.5, label='MVBE')

# get the "correct" value by tracing points where R=1
# R(u,v) = r² => v = u²±√(r²-(a-u)²)/b
# radical > 0 => a-r < x < a+r
r = 1
uu = np.linspace(a-r, a+r, 11)
vv = np.hstack([uu[:-1]**2 + np.sqrt(r-(a-uu[:-1])**2)/b,
                uu[-1:0:-1]**2 - np.sqrt(r-(a-uu[-1:0:-1])**2)/b,
                uu[0]**2])
uu = np.hstack([uu[:-1], uu[-1:0:-1], uu[0]])
# uu = np.hstack([uu[:-1], uu[1:]])
mu0, C0 = bb.MVBE(np.vstack([uu,vv]).T, tol=1e-9)
# pl.plot(uu, vv, 'ro')
bbplot.cov_ellipse(mu0, C0, lw=1.5, ec='k', fc='none', ls='-');
pl.plot([np.nan, np.nan], [np.nan, np.nan], 'k-', lw=1.5, label='MVBE ($\Delta\chi^2=1$)')

pl.xlim(xmin-0.3, xmax-0.6)
pl.ylim(ymin-0.2, ymax-0.15)
pl.legend(ncol=2, borderpad=0.2, labelspacing=0.2, handlelength=1.2,
          handletextpad=0.4, columnspacing=0.2, loc='lower left', bbox_to_anchor=(-0.01,-0.01))
pl.xlabel(r"$x_1$")
pl.ylabel(r"$x_2$")

pl.savefig('5-nonlinear.pdf')
