import numpy as np

# If you're going to edit the code, some linear algebra is useful.
# Without loss of generality, an ellipse centred at zero is defined in
# "mean-covariance" form by
#
#     x'C¯¹x = 1
#
# C¯¹ must be positive definite, so we can factorise it by
#
#     C¯¹ = UΣ²U' = (UΣ)(ΣU')
#
# using e.g. the SVD.  If we then define
#
#     B = ΣU'
#
# our ellipsoid becomes
#
#     x'C¯¹x = (Bx)'(Bx)
#
# so B maps point x on an ellipsoid to y=Bx on a sphere.
# Specifically, V' rotates the principal axes to the co-ordinate axes
# and Σ rescales them to the unit vectors.  The reverse transformation,
# B¯¹ = (ΣU')¯¹ = UΣ¯¹ maps a point on a sphere to an ellipse.
#
# The SVD and eigenvalue decomposition both factorise C¯¹.  SVD gives
# USV' with S in *decreasing* order. `eigh` gives L, Q such that QLQ¯¹
# with L in *increasing* order.  So L = S[::-1] and, up to a sign, Q =
# U[:,::-1].

def MVBE(x, tol=1e-3, maxiter=1000):
    """Return the mean and covariance matrix that define the
    minimum-volume ellipsoid that bounds the points in ``x``
    using algorithm 3.2 of `Moshtagh (2005)`_.

    .. _Moshtagh (2005): http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.116.7691

    Parameters
    ----------
    x: 2-d NumPy array of shape (N, D)
    tol: float, optional
        Tolerance parameter for precision of bounding ellipsoid (default=1e-3).
    maxiter: int, optional
        Maximum number of iterations to perform (default=1000).

    Returns
    -------
    mu: 1-d NumPy array of shape (D,)
        Mean (or center) of the ellipsoid.
    C: 2-d NumPy array of shape (D,D)
        Covariance matrix that defines the ellipsoid.

    """
    N, D = x.shape
    I = np.eye(N)
    Q = np.column_stack((x, np.ones(N))).T
    u = np.ones(N)/N
    err = np.inf
    i = 0
    while err > tol and i < maxiter:
        V = Q.dot(np.diag(u)).dot(Q.T)
        Vinv = np.linalg.inv(V)
        # g = np.array([Qi.dot(Vinv).dot(Qi) for Qi in Q.T]) # slow but definitely correct
        g = np.sum(Q*(Vinv.dot(Q)), axis=0) # fast
        j = np.argmax(g)
        du = I[j]-u
        alpha = (g[j]-(D+1))/(D+1)/(g[j]-1)
        err = np.sum((alpha*du)**2)**0.5
        u = u + alpha*du
        i += 1

    mu = u.dot(x)
    C = (x.T.dot(np.diag(u)).dot(x)-np.outer(mu, mu))*D
    return mu, C


def select_by_relative_chi2(chi2, sigma_inner=1e-9, sigma_outer=np.inf):
    """Returns a logical array that is ``True`` where ``chi2`` relative to
    its minimum is within the range defined by ``sigma_inner`` and
    ``sigma_outer``. i.e. where ``sigma_inner**2 < chi2-np.min(chi2) <
    sigma_outer**2``."""
    relative_chi2 = chi2 - np.nanmin(chi2)
    return (sigma_inner**2 < relative_chi2) & (relative_chi2 < sigma_outer**2)


def estimate(x, chi2, sigma_inner=1e-9, sigma_outer=np.inf,
             tol=1e-3, maxiter=1000):
    """Estimate uncertainties for a sample of points given their
    correspond values of chi-squared.

    Parameters
    ----------
    x: 2-d NumPy array of shape (N, D)
        Points at which chi-squared has been evaluated.
    chi2: 1-d NumPy array of shape (N,)
        Values of chi-squared for the points in ``x``.
    sigma_inner: float, optional
        Only use the points that are more than this many sigma from
        the minimum.
    sigma_outer: float, optional
        Only use the points that are less than this many sigma from
        the minimum.
    tol: float, optional
        Tolerance parameter for finding minimum-volume bounding
        ellipsoid.
    maxiter: int, optional
        Maximum number of iterations to use when finding
        minimum-volume bounding ellipsoid.

    Returns
    -------
    mu: 1-d NumPy array of shape (D,)
        Mean (or center) of the minimum-volume bounding ellipsoid
        around the rescaled points in ``x``.
    C: 2-d NumPy array of shape (D,D)
        Covariance matrix that defines the ellipsoid.
    """
    x0 = x[np.nanargmin(chi2)]
    I = np.where(select_by_relative_chi2(chi2, sigma_inner=sigma_inner, sigma_outer=sigma_outer))[0]
    z = x0 + (x[I]-x0)/np.sqrt(chi2[I,None]-np.nanmin(chi2))
    return MVBE(z[np.all(np.isfinite(z), axis=1)], tol=tol, maxiter=maxiter)


def propagate(x, y, mu, C, weights=None):
    """Propagate the uncertainties on ``x`` to other dependent variables
    ``y``.  Returns full mean and uncertainty so that the correlations
    between main parameters and dependent variables are given.

    Parameters
    ----------
    x: 2-d NumPy array
        Points at which function has been evaluated.
    y: 2-d NumPy array
        Values of function at points in ``x``.
    mu: 1-d NumPy array of shape (D,)
        Mean (or center) of the ellipsoid that characterizes the
        ``chi2`` values of the points in ``x``.
    C: 2-d NumPy array of shape (D,D)
        Covariance matrix of the ellipsoid.
    weights: 1-d NumPy array, optional
        Relative weights for the points in the fit.

    Returns
    -------
    mu: 1-d NumPy array of shape (D,)
        Mean (or center) of both the underlying parameters ``x`` and
        the dependent variables ``y``.
    C: 2-d NumPy array of shape (D,D)
        Covariance matrix of both the underlying parameters ``x`` and
        the dependent variables ``y``.

    """
    Q, B = linearize(x, y, weights=weights)
    A = np.vstack((np.eye(len(x[0])), B))
    return np.hstack((mu, Q + B.dot(mu))), (A.dot(C)).dot(A.T)


def linearize(x, y, weights=None):
    """Given a set of points ``x`` and the values ``y`` of some (vector)
    function evaluated at those points, find the parameters of the
    best-fitting affine map between ``x`` and ``y``. i.e. returns
    arrays ``Q`` and ``B`` such that ``y == Q + x.dot(B.T)``.

    Parameters
    ----------
    x: 2-d NumPy array of shape (N, D)
        Points at which function has been evaluated.
    y: 2-d NumPy array of shape (N, K)
        Values of function at points in ``x``.
    weights: 1-d NumPy array, optional
        Relative weights for the points in the fit.

    Returns
    -------
    Q: 1-d NumPy array of shape (K,)
        Constant offset of affine map from ``x`` to ``y``.
    B: 2-d NumPy array of shape (K, D)
        Matrix transformation of affine map from ``x`` to ``y``.

    """
    if weights is None:
        weights = np.ones(len(x))

    x1 = np.hstack((np.ones((len(x),1)), x))
    B = np.linalg.lstsq(x1*weights[:,None], y*weights[:,None], rcond=-1)[0]
    Q = B[0]
    B = B[1:].T

    return Q, B


def uniform_on_unit_sphere(N, D):
    """Draw ``N`` points uniformly distributed on the surface of a
    ``D``-dimensional unit sphere."""
    x = np.random.randn(N, D)
    x = x/np.sqrt(np.sum(x**2, axis=1)).reshape((-1,1))
    return x


def sphere_to_ellipsoid(x, mu, C):
    U, S, VT = np.linalg.svd(np.linalg.inv(C))
    return mu + x.dot(np.diag(np.sqrt(1./S))).dot(VT) # (VΣ¯¹x)' = x'Σ¯¹V'


def uniform_on_ellipsoid(mu, C, N):
    """Draw ``N`` points uniformly distributed on the surface of the
    ellipsoid define by mean ``mu`` and covariance ``C``."""
    # https://math.stackexchange.com/a/982833
    # μ² = (dS'/dS)² = (abc…)²((x/a)²+(y/b)²+(z/c)²+…),
    # where (x,y,z,…) are points on sphere
    # μ²_max = (abc…)²(1/a)² = (bc…)² assuming a is shortest axis
    # μ²/μ²_max = ((x/a)²+(y/b)²+(z/c)²+…)/(1/a²)
    def iterate():
        x = uniform_on_unit_sphere(10000, len(mu))
        area = np.sum(x**2*S, axis=1)**0.5
        return sphere_to_ellipsoid(x[area/area_max > np.random.rand(len(x))], mu, C)

    U, S, VT = np.linalg.svd(np.linalg.inv(C))
    area_max = S[0]**0.5
    y_all = iterate()
    while len(y_all) < N:
        y_all = np.vstack((y_all, iterate()))

    return y_all[:N]


def uniform_in_ellipsoid(mu, C, N):
    """Draw ``N`` points uniformly distributed inside the ellipsoid define
    by mean ``mu`` and covariance ``C``."""
    x = mu.reshape((1,-1))
    std = np.sqrt(np.diag(C))
    while len(x) < N+1:
        # x_n = mu - std + 2.*std*np.random.rand(len(mu))
        # if is_in_ellipsoid(x_n, mu, C): x.append(x_n)
        x_n = (mu - std).reshape((1,-1)) + 2.*std.reshape((1,-1))*np.random.rand(1000, len(mu))
        x = np.vstack((x, x_n[is_in_ellipsoid(x_n, mu, C)]))

    return np.array(x)[:N]


def vertices(mu, C):
    """Return the ``2*D`` points at the ends of each axis of the ellipsoid
    defined by mean ``mu`` and covariance ``C``."""
    # should return VΣ¯¹[I,-I]
    # transpose is [I;-I]Σ¯¹V'
    D = len(C)
    U, S, VT = np.linalg.svd(C) # or S, VT = np.linalg.eigh(C)
    x = np.diag(np.sqrt(S)).dot(VT)
    return mu + np.vstack([x, -x])


def is_in_ellipsoid(x, mu, C):
    """Returns ``True`` if the point(s) in ``x`` are inside the ellipsoid
    defined by mean ``mu`` and covariance ``C``."""
    return np.sum((x-mu).T*(np.linalg.inv(C).dot((x-mu).T)), axis=0) < 1


def is_in_simplex(x, s):
    """Returns ``True`` if the 1-d array ``x`` is in the simplex ``s``,
    where ``s`` is a 2-d array with dimensions ``(D+1,D)`` (each row is a
    point)."""
    D = len(s[0])
    A = np.ones((D+1, D+1))
    A[:-1] = s.T
    try:
        return np.all(np.linalg.solve(A, np.hstack([x, 1.])) >= 0)
    except np.linalg.LinAlgError:
        return False


def discard(x, iterations=1):
    """Drop points that don't help to define the bounding ellipsoid."""
    D = len(x[0])
    I = np.ones(len(x), dtype=bool)
    for iteration in range(iterations):
        s = np.random.permutation(x[I])[:D+1] # construct a random simplex
        for i, xi in enumerate(x):
            if not I[i]:
                continue
            elif np.any(np.all(xi == s, axis=1)):
                continue
            else:
                I[i] = not is_in_simplex(xi, s) # discard points inside that simplex

        if sum(I) == D+1:
            break

    return x[I]


def grid(N, lower, upper):
    """Create a regular grid with `N` points between `lower` and
    `upper`."""
    D = len(upper)
    N = N*np.ones(D, dtype=int)
    grid = np.vstack(list(map(np.ravel, np.meshgrid(*[np.linspace(lower[i], upper[i], N[i]) for i in range(D)])))).T
    return grid
