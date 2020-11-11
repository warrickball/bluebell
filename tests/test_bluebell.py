#!/usr/bin/env python3

import numpy as np
import bluebell as bb
import unittest

def get_random_mu_and_C(D):
    mu = np.random.uniform(low=-1., high=1., size=(D))
    C = np.random.uniform(low=-1., high=1., size=(D,D))
    C = C@C.T # positive symmetric
    return mu, C

class TestBluebell(unittest.TestCase):
    def test_MVBE_unit_sphere(self):
        for D in range(2, 9):
            x = np.vstack([np.eye(D), -np.eye(D)])
            mu, C = bb.MVBE(x, tol=1e-6)
            np.testing.assert_allclose(mu, np.zeros(D))
            np.testing.assert_allclose(C, np.eye(D))

    def test_MVBE_ellipsoid(self):
        for D in range(2, 9):
            mu0, C0 = get_random_mu_and_C(D)
            x = bb.extrema(mu0, C0)
            mu1, C1 = bb.MVBE(x)
            np.testing.assert_allclose(mu0, mu1)
            np.testing.assert_allclose(C0, C1)

    def test_uniform_in_ellipsoid(self):
        for D in range(2, 9):
            mu, C = get_random_mu_and_C(D)
            x = bb.uniform_in_ellipsoid(mu, C, 100)
            self.assertTrue(np.all(bb.is_in_ellipsoid(x, mu, C)))

    def test_uniform_on_ellipsoid(self):
        for D in range(2, 9):
            mu, C = get_random_mu_and_C(D)
            x = bb.uniform_on_ellipsoid(mu, C, 100)
            self.assertTrue(np.all(bb.is_in_ellipsoid(x, mu, 1.0001*C)))
            self.assertFalse(np.any(bb.is_in_ellipsoid(x, mu, 0.9999*C)))

    def test_sphere_to_ellipsoid(self):
        for D in range(2, 9):
            mu, C = get_random_mu_and_C(D)
            x = bb.sphere_to_ellipsoid(bb.uniform_on_unit_sphere(100, D),
                                       mu, C)
            self.assertTrue(np.all(bb.is_in_ellipsoid(x, mu, 1.0001*C)))
            self.assertFalse(np.any(bb.is_in_ellipsoid(x, mu, 0.9999*C)))

    def test_linearize(self):
        for D in range(2, 9):
            x = np.random.uniform(size=(100, D))
            c0 = np.random.uniform(size=(D-1,))
            A0 = np.random.uniform(size=(D-1, D))
            y = c0 + x.dot(A0.T)
            c1, A1 = bb.linearize(x, y)
            np.testing.assert_allclose(c0, c1)
            np.testing.assert_allclose(A0, A1)

    def test_propagate(self):
        for D in range(3,5):
            mu = np.zeros(D)
            C = np.eye(D)
            x = np.random.uniform(size=(100,D))
            for K in range(1,D):
                Q = np.ones(K)
                B = np.hstack([np.eye(K,K), np.zeros((K,D-K))])
                y = Q + x.dot(B.T)
                mu_prop, C_prop = bb.propagate(x, y, mu, C)
                np.testing.assert_allclose(mu_prop[:D], mu)
                np.testing.assert_allclose(C_prop[:D,:D], C)
                np.testing.assert_allclose(mu_prop[D:], Q)
                np.testing.assert_allclose(C_prop[D:,D:], np.eye(K),
                                           atol=1e-12)

    def test_known_in_simplex(self):
        # define a simplex by origin plus unit vectors
        # then we know certain points must be inside
        for D in range(2, 5):
            s = np.vstack([np.zeros(D),
                           np.eye(D)])
            for si in s:
                self.assertTrue(bb.is_in_simplex(si, s))

            self.assertFalse(bb.is_in_simplex(np.ones(D), s))
            for si in -s[1:]:
                self.assertFalse(bb.is_in_simplex(si, s))

            x = np.random.rand(10, D)
            truth = [bb.is_in_simplex(xi, s) for xi in x]
            self.assertTrue(np.all(truth == (np.sum(x, axis=1) <= 1)))


    def test_known_discard(self):
        for D in range(2, 5):
            x = np.vstack([np.eye(D), -np.eye(D), np.zeros(D)])
            mu, C = bb.MVBE(x, tol=1e-6, maxiter=10000)
            np.testing.assert_allclose(mu, np.zeros(D), atol=1e-3)
            np.testing.assert_allclose(C, np.eye(D), atol=1e-3)

            x = bb.discard(x, iterations=5)
            mu, C = bb.MVBE(x, tol=1e-6, maxiter=10000)
            np.testing.assert_allclose(mu, np.zeros(D), atol=1e-3)
            np.testing.assert_allclose(C, np.eye(D), atol=1e-3)
            
            
    def test_random_discard(self):
        for D in range(2, 5):
            x = np.random.uniform(low=-1.0, high=1.0, size=(100, D))
            mu0, C0 = bb.MVBE(x, tol=1e-6, maxiter=10000)
            x = bb.discard(x, iterations=4)
            mu1, C1 = bb.MVBE(x, tol=1e-6, maxiter=10000)
            np.testing.assert_allclose(mu0, mu1, atol=1e-3)
            np.testing.assert_allclose(C0, C1, atol=1e-3)
                

if __name__ == '__main__':
    unittest.main()
