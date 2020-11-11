import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Ellipse

def cov_ellipse(mu, C, ax=None, **kwargs):
    """Plot an ellipse at point ``mu`` given its covariance matrix
    ``cov``.

    Parameters
    ----------
    mu: array (2,)
        Center of the ellipse

    C: array (2,2)
        The covariance matrix for the point

    ax: matplotlib.Axes, optional
        The axis to overplot on

    **kwargs : dict
        These keywords are passed to matplotlib.patches.Ellipse

    Returns
    -------
    ellipse: matplotlib Ellipse object

    Example
    -------

    .. plot::
       :include-source:

       import matplotlib.pyplot as pl
       import bluebell.plot as bbplot

       A = np.random.rand(2,2)
       C = A.T.dot(A)
       mu = np.random.rand(2)
       std = np.sqrt(np.diag(C))

       bbplot.cov_ellipse(mu, C, ec='C0', fc='none')
       pl.axis((mu + np.array([[-1.1],[1.1]])*std).T.flatten())

    """
    L, Q = np.linalg.eigh(C)
    theta = np.degrees(np.arctan2(Q[1,1], Q[1,0]))
    ellipse = Ellipse(xy=mu,
                      width =2*np.sqrt(L[1]),
                      height=2*np.sqrt(L[0]),
                      angle=theta,
                      **kwargs)

    if ax is None:
        ax = pl.gca()

    ax.add_patch(ellipse)
    return ellipse


def chi2_corner(x, chi2, vmin=None, vmax=None, names=None,
                cmap='inferno', scale='sqrt'):
    """Create a corner plot showing the 2-d projections of the sample *x*
    in with each point coloured by its relative value of *chi2*, as
    well as a top row showing 1-d plots of *chi2* for each dimension.

    Parameters
    ----------
    x: 2-d NumPy array of shape (N, D)
        Points at which function has been evaluated.
    chi2: 1-d NumPy array of shape (N,)
        Values of chi-squared for the points in ``x``.

    Returns
    -------
    fig: matplotlib figure
    ax: 2-d array of shape (D, D) of matplotlib.axes
        Array of the matplotlib axes containing the individual scatter
        plots.

    Example
    -------

    .. plot::
       :include-source:

       import matplotlib.pyplot as pl
       import bluebell as bb
       import bluebell.plot as bbplot

       D = 3
       C = np.eye(D)
       mu = np.zeros(D)
       std = np.sqrt(np.diag(C))

       x = np.random.uniform(low=mu-std, high=mu+std, size=(100,D))
       x[0] = mu
       chi2 = np.sum(x.dot(np.linalg.inv(C))*x, axis=1)

       fig, ax = bbplot.chi2_corner(x, chi2)

    """
    N = len(x[0])
    I = np.argsort(chi2)[::-1]

    if scale.lower() == 'sqrt':
        z = np.sqrt(chi2-np.nanmin(chi2))
    elif scale.lower() == 'log10':
        z = np.log10(chi2-np.nanmin(chi2)+1)
    elif scale.lower() == 'log':
        z = np.log(chi2-np.nanmin(chi2)+1)
    else:
        z = chi2

    if vmin is None:
        vmin = np.nanmin(z)

    if vmax is None:
        vmax = np.nanmax(z)

    # arrangement is:
    #   (0,0)   ...   (0,N-1)
    #    ...           ...
    # (N-1,0)   ... (N-1,N-1)

    fig, ax = pl.subplots(N, N, sharex='col', sharey='row')

    scatter_kwargs = {'s': 100, 'c': z[I],
                      'vmin': vmin, 'vmax': vmax, 'cmap': cmap}

    # populate lower triangle ...
    for i in range(1, N):
        for j in range(i):
            ax[i][j].scatter(x[I,j], x[I,i], **scatter_kwargs)

        # ... and blank upper triangle
        for j in range(i, N):
            ax[i][j].axis('off')

    # do top row
    for j in range(N):
        ax[0][j].plot(x[I,j], chi2[I], 'ko', alpha=0.5)

    # for j in range(N):
    #     for i in range(N):
    #         if ax[i][j] is None:
    #             continue
    #         else:
    #             ax[i][j].tick_params(labelleft=False, labelbottom=False,
    #                                   labeltop=False, labelright=False)

    for j in range(N):
        ax[0][j].tick_params(labeltop=True)

    for j in range(N-1):
        ax[N-1][j].tick_params(labelbottom=True)
        ax[j+1][0].tick_params(labelleft=True)

    ax[0][0].tick_params(labelleft=True)
    ax[0][-1].tick_params(labelright=True)

    #     for i in range(1,j):
    #         ax[i][j].tick_params(labelleft=False, labelbottom=False, labeltop=True, labelright=True)

    # ax[i][j].tick_params(labelbottom=True)

    if not names is None:
        for j in range(N-1):
            ax[N-1][j].set_xlabel(names[j])

        for i in range(1, N):
            ax[i][0].set_ylabel(names[i])

        for j in range(N):
            ax[0][j].set_xlabel(names[j])
            ax[0][j].xaxis.set_label_position('top')

    return fig, ax
