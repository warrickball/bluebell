#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as pl
import bluebell as bb
import bluebell.plot as bbplot
from scipy.interpolate import interp2d
from scipy.optimize import curve_fit

def show_mu_C(name, mu, C):
    print(''.join(['%20s']+['%16.8f']*5) % (name, mu[0], mu[1], C[0,0], C[0,1], C[1,1]))

pl.style.use('paper.mpl')

np.random.seed(44882244)
t = np.sort(np.random.uniform(low=-3, high=3, size=10))

def lorentz(t, p):
    return p[0]/(1+(t/p[1])**2)

x = np.array([1., 1.])
sigma = np.random.uniform(low=0.02, high=0.2, size=len(t))
y = lorentz(t, x) + sigma*np.random.randn(len(t))

def chi2(p):
    return np.sum((lorentz(t, p) - y)**2/sigma**2)

p0 = np.array([0.95, 1.05])
ps = [p0, p0+np.array([0.1, 0]), p0+np.array([0, 0.1])]
zs = [chi2(p) for p in ps]

a, b, c, d = 1.0, 2.0, 0.5, 0.5

s = np.array(ps)
z = np.array(zs)
for i in range(20):
    s = s[np.argsort(z)]
    z = z[np.argsort(z)]
    s0 = np.mean(s[:-1], axis=0)

    # reflect
    si = s0 + a*(s0 - s[-1])
    zi = chi2(si)
    ps.append(si + 0)
    zs.append(zi + 0)

    if zi < z[-1]:
        z[-1] = zi + 0
        s[-1] = si + 0

        if zi < z[0]:
            # expand
            si = s0 + b*(si - s0)
            zi = chi2(si)
            ps.append(si + 0)
            zs.append(zi + 0)

            # better than reflect
            if zi < z[-1]:
                z[-1] = zi + 0
                s[-1] = si + 0
                
        continue

    # contract
    si = s0 + c*(s[-1] - s0)
    zi = chi2(si)
    ps.append(si + 0)
    zs.append(zi + 0)
    
    if zi < z[-1]:
        z[-1] = zi + 0
        s[-1] = si + 0
        continue

    # no shrink
    print("WARNING: shouldn't have to shrink, stopping NM")
    break

    # shrink
    s[1:] = s[:1] + d*(s[1:]-s[:1])

# NM uncertainty
P = (s[None,:,:]+s[:,None,:])/2
NM_a0 = chi2(P[0,0])
NM_a = np.array([2*chi2(P[0,i]) - (chi2(P[i,i])+3*chi2(P[0,0]))/2 for i in range(1,3)])
NM_b = np.array([[2*(chi2(P[i,j])+chi2(P[0,0])-chi2(P[0,i])-chi2(P[0,j]))
                  for i in range(1,3)] for j in range(1,3)])
NM_Q = (s[1:]-s[0]).T

NM_x = np.linalg.lstsq(-NM_b, NM_a, rcond=None)[0]
NM_p = s[0] + NM_Q@NM_x
NM_C = NM_Q@np.linalg.inv(NM_b)@NM_Q.T

show_mu_C('NM Hessian', NM_p, NM_C)

ps = np.array(ps)
zs = np.array([chi2(ppppp) for ppppp in ps])
popt = ps[np.argmin(zs)]

mu, C = bb.estimate(ps, zs, sigma_inner=0.25, sigma_outer=3.0)

show_mu_C('bluebell', mu, C)

def func(t, *p):
    global i
    i += 1
    return lorentz(t, p)

i = 0

# let's find the true optimum ...
ropt, rcov, info, msg, ierr = \
    curve_fit(func, t, y, p0=p0, sigma=sigma, absolute_sigma=True,
              ftol=1e-10, xtol=1e-10, full_output=True)
# ... then solve to the same precision as the Nelder-Mead solution
xmin = ropt
chi2min = np.sum(info['fvec']**2)

show_mu_C('curve_fit', ropt, rcov)

i = 0
ropt, rcov, info, msg, ierr = \
    curve_fit(func, t, y, p0=p0, sigma=sigma, absolute_sigma=True,
              full_output=True,
              ftol=1e-6, # chi2(popt)-chi2min,
              xtol=1e-4) # np.min(np.abs(popt-xmin)))

show_mu_C('curve_fit', ropt, rcov)

import emcee

def lnprob(p):
    if np.any(p <= 0):
        return -np.inf
    else:
        return -0.5*chi2(p)

D = 2
W = 10
S = emcee.EnsembleSampler(W, D, lnprob)
pos = 1 + 0.1*np.random.randn(W, D)
while np.any(pos <= 0):
    pos[pos <= 0] = 1 + 0.1*np.random.randn(*pos[pos <= 0].shape)

pos, prob, state = S.run_mcmc(pos, 2000)
S.reset()
pos, prob, state = S.run_mcmc(pos, 2000)

show_mu_C('emcee', np.mean(S.flatchain, axis=0), np.cov(S.flatchain.T))

pl.hist2d(*S.flatchain.T, cmap='gray_r', bins=30);

pl.plot(*ps.T, 'o', label=r"$\bm{x}$")

bbplot.cov_ellipse(NM_p, NM_C, fc='none', ec='C0', lw=1.5)
pl.plot([np.nan, np.nan], [np.nan, np.nan], c='C0', ls='-', label='NM')

I = (0.25**2 < zs-np.min(zs)) & (zs-np.min(zs) < 3.0**2)
pp = ps[np.argmin(zs)] + (ps[I]-ps[np.argmin(zs)])/np.sqrt(zs[I]-np.min(zs))[:,None]
pl.plot(*pp.T, 's', label=r"$\bm{x}'$")
bbplot.cov_ellipse(mu, C, fc='none', ec='C1', lw=1.5)
pl.plot([np.nan, np.nan], [np.nan, np.nan], c='C1', ls='-', label='MVBE')

bbplot.cov_ellipse(ropt, rcov, fc='none', ec='C2', lw=1.5)
pl.plot([np.nan, np.nan], [np.nan, np.nan], c='C2', ls='-', label=r"$\texttt{curve\_fit}$")

I = np.abs(S.flatlnprobability.max()-S.flatlnprobability-0.5)<0.2
pl.tricontour(*S.flatchain[I].T, S.flatlnprobability[I],
              [np.max(S.flatlnprobability)-0.5],
              colors='C3', linestyles='solid');
pl.plot([np.nan, np.nan], [np.nan, np.nan], c='C3', ls='-', label='MCMC')

pl.xlabel(r"$A$")
pl.ylabel(r"$\Gamma$")
pl.axis([0.9, 1.4, 0.7, 1.25])
pl.legend(borderpad=0.2, labelspacing=0.2, handlelength=1.2,
          handletextpad=0.4, columnspacing=0.2, loc='upper right', bbox_to_anchor=(1.01,1.01))

pl.savefig('6-lorentzian.pdf')
