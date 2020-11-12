# bluebell

`bluebell` is a Python package for estimating the uncertainties of
parameters in least-squares problems by computing minimum-volume
bounding ellipsoids (MVBEs). Given a set of points *x* at which some
*χ²* has been evaluated, `bluebell` computes the uncertainties on the
points *x* by computing the MVBE around the points, moved by their
values of *χ²* such that, if the problem is exactly linear, all the
points lie on the same ellipsoid.

Install `bluebell` from PyPI with

    python -m pip install bluebell

Read the documentation on [Read the Docs](https://bluebell.readthedocs.io/).
