.. bluebell documentation master file, created by
   sphinx-quickstart on Tue Nov  3 15:59:59 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

bluebell
========
Basic linear uncertainty estimation with bounding ellipsoids
------------------------------------------------------------

``bluebell`` is a Python package for estimating the uncertainties of
parameters in least-squares problems by computing minimum-volume
bounding ellipsoids (MVBEs).  Given a set of points :math:`x` at which
some :math:`\chi^2` has been evaluated, ``bluebell`` computes the
uncertainties on the points :math:`x` by computing the MVBE around the
points, moved by their values of :math:`\chi^2-\chi^2_\mathrm{min}`
such that, if the problem is exactly linear, all the points lie on the
same ellipsoid.

.. plot::

   import matplotlib.pyplot as pl
   import bluebell as bb
   import bluebell.plot as bbplot

   np.random.seed(3816547290)
   mu = np.array([1.,1.])
   C = np.array([[1., 0.5], [0.5, 1.]])
   std = np.sqrt(np.diag(C))
   low = mu - 1.5
   high = mu + 1.5
   x = np.random.uniform(low=low, high=high, size=(20,2))

   chi2 = np.array([xi@np.linalg.inv(C)@xi for xi in x-mu])
   xx = mu + (x-mu)/np.sqrt(chi2)[:,None]
   dx = xx-x

   bbplot.cov_ellipse(mu, C, ec='k', fc='none', lw=1.5);

   for xi, dxi in zip(x, dx):
      pl.arrow(*(xi+0.05*dxi), *(0.9*dxi),
               width=0.01, head_width=0.05,
               length_includes_head=True, fc='k', ec='k')

   pl.scatter(*x.T, c=np.sqrt(chi2), s=50);
   pl.scatter(*xx.T, c=np.sqrt(chi2), s=50);

   pl.xlim(low[0], high[0])
   pl.ylim(low[1], high[1])

``bluebell`` also provides a number of support functions to work with
the points and ellipsoids, as well as some plotting functions in
``bluebell.plot``.

You can install ``bluebell`` from the Python Package Index (PyPI) with ::

    python -m pip install bluebell

perhaps also with the ``--user`` flag, depending how you administer
your system.  The `development version
<https://github.com/warrickball/bluebell>`_ is on GitHub.

.. toctree::
   :maxdepth: 2
   :caption: API

   bluebell
   bluebell.plot

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
