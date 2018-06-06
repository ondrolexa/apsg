# -*- coding: utf-8 -*-


from __future__ import division
import numpy as np
from scipy.stats import uniform
from scipy.special import gamma as gamma_fun
from scipy.special import iv as modified_bessel_2ndkind
from scipy.special import ivp as modified_bessel_2ndkind_derivative
from scipy.stats import norm as gauss


def sind(x):
    return np.sin(np.deg2rad(x))


def cosd(x):
    return np.cos(np.deg2rad(x))


def tand(x):
    return np.tan(np.deg2rad(x))


def asind(x):
    return np.rad2deg(np.arcsin(x))


def acosd(x):
    return np.rad2deg(np.arccos(x))


def atand(x):
    return np.rad2deg(np.arctan(x))


def atan2d(x1, x2):
    return np.rad2deg(np.arctan2(x1, x2))


def getldd(x, y):
    return (atan2d(x, y) % 360, 90 - 2 * asind(np.sqrt((x * x + y * y) / 2)))


def getfdd(x, y):
    return (atan2d(-x, -y) % 360, 2 * asind(np.sqrt((x * x + y * y) / 2)))


def l2v(azi, inc):
    return np.array(
        [
            np.atleast_1d(cosd(azi) * cosd(inc)),
            np.atleast_1d(sind(azi) * cosd(inc)),
            np.atleast_1d(sind(inc)),
        ]
    )


def p2v(azi, inc):
    return np.array(
        [
            np.atleast_1d(-cosd(azi) * sind(inc)),
            np.atleast_1d(-sind(azi) * sind(inc)),
            np.atleast_1d(cosd(inc)),
        ]
    )


def v2l(u):
    n = u / np.sqrt(np.sum(u * u, axis=0))
    ix = n[2] < 0
    n.T[ix] = -n.T[ix]
    azi = atan2d(n[1], n[0]) % 360
    inc = asind(n[2])
    return azi, inc


def v2p(u):
    n = u / np.sqrt(np.sum(u * u, axis=0))
    ix = n[2] < 0
    n.T[ix] = -n.T[ix]
    azi = (atan2d(n[1], n[0]) + 180) % 360
    inc = 90 - asind(n[2])
    return azi, inc


def l2xy(azi, inc):
    r = np.sqrt(2) * sind(45 - inc / 2)
    return r * sind(azi), r * cosd(azi)


def rodrigues(k, v, theta):
    return (
        v * cosd(theta)
        + np.cross(k.T, v.T).T * sind(theta)
        + k * np.dot(k.T, v) * (1 - cosd(theta))
    )


def angle_metric(u, v):
    return np.degrees(np.arccos(np.abs(np.dot(u, v))))


def eformat(f, prec):
    s = "{:e}".format(f)
    m, e = s.split("e")
    return "{:.{:d}f}E{:0d}".format(float(m), prec, int(e))


# ############################################################################
# Following counting routines are from Joe Kington's mplstereonet
# https://github.com/joferkington/mplstereonet
# ############################################################################


def _kamb_radius(n, sigma):
    """Radius of kernel for Kamb-style smoothing."""
    a = sigma ** 2 / (float(n) + sigma ** 2)
    return 1 - a


def _kamb_units(n, radius):
    """Normalization function for Kamb-style counting."""
    return np.sqrt(n * radius * (1 - radius))


# ############################################################################
# All of the following kernel functions return an _unsummed_ distribution and
# a normalization factor.
# ############################################################################


def _exponential_kamb(cos_dist, sigma=3):
    """Kernel function from Vollmer for exponential smoothing."""
    n = float(cos_dist.size)
    f = 2 * (1.0 + n / sigma ** 2)
    count = np.exp(f * (cos_dist - 1))
    units = np.sqrt(n * (f / 2.0 - 1) / f ** 2)
    return count, units


def _linear_inverse_kamb(cos_dist, sigma=3):
    """Kernel function from Vollmer for linear smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, sigma)
    f = 2 / (1 - radius)
    # cos_dist = cos_dist[cos_dist >= radius]
    count = f * (cos_dist - radius)
    count[cos_dist < radius] = 0
    return count, _kamb_units(n, radius)


def _square_inverse_kamb(cos_dist, sigma=3):
    """Kernel function from Vollemer for inverse square smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, sigma)
    f = 3 / (1 - radius) ** 2
    # cos_dist = cos_dist[cos_dist >= radius]
    count = f * (cos_dist - radius) ** 2
    count[cos_dist < radius] = 0
    return count, _kamb_units(n, radius)


def _kamb_count(cos_dist, sigma=3):
    """Original Kamb kernel function (raw count within radius)."""
    n = float(cos_dist.size)
    dist = _kamb_radius(n, sigma)
    # count = (cos_dist >= dist)
    count = np.array(cos_dist >= dist, dtype=float)
    return count, _kamb_units(n, dist)


def _schmidt_count(cos_dist, sigma=None):
    """Schmidt (a.k.a. 1%) counting kernel function."""
    radius = 0.01
    count = (1 - cos_dist) <= radius
    # To offset the count.sum() - 0.5 required for the kamb methods...
    count = 0.5 / count.size + count
    return count, cos_dist.size * radius


class KentDistribution(object):
    """
    The algorithms here are partially based on methods described in:
    [The Fisher-Bingham Distribution on the Sphere, John T. Kent
    Journal of the Royal Statistical Society. Series B (Methodological)
    Vol. 44, No. 1 (1982), pp. 71-80 Published by: Wiley
    Article Stable URL: http://www.jstor.org/stable/2984712]

    Implementation by Daniel Fraenkel
    https://github.com/edfraenkel/kent_distribution
    """

    minimum_value_for_kappa = 1E-6

    @staticmethod
    def create_matrix_H(theta, phi):
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [
                    np.sin(theta) * np.cos(phi),
                    np.cos(theta) * np.cos(phi),
                    -np.sin(phi),
                ],
                [np.sin(theta) * np.sin(phi), np.cos(theta) * np.sin(phi), np.cos(phi)],
            ]
        )

    @staticmethod
    def create_matrix_Ht(theta, phi):
        return np.transpose(KentDistribution.create_matrix_H(theta, phi))

    @staticmethod
    def create_matrix_K(psi):
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(psi), -np.sin(psi)],
                [0.0, np.sin(psi), np.cos(psi)],
            ]
        )

    @staticmethod
    def create_matrix_Kt(psi):
        return np.transpose(KentDistribution.create_matrix_K(psi))

    @staticmethod
    def create_matrix_Gamma(theta, phi, psi):
        H = KentDistribution.create_matrix_H(theta, phi)
        K = KentDistribution.create_matrix_K(psi)
        return np.inner(H, np.transpose(K))

    @staticmethod
    def create_matrix_Gammat(theta, phi, psi):
        return np.transpose(KentDistribution.create_matrix_Gamma(theta, phi, psi))

    @staticmethod
    def spherical_coordinates_to_gammas(theta, phi, psi):
        Gamma = KentDistribution.create_matrix_Gamma(theta, phi, psi)
        gamma1 = Gamma[:, 0]
        gamma2 = Gamma[:, 1]
        gamma3 = Gamma[:, 2]
        return (gamma1, gamma2, gamma3)

    @staticmethod
    def gamma1_to_spherical_coordinates(gamma1):
        theta = np.arccos(gamma1[0])
        phi = np.arctan2(gamma1[2], gamma1[1])
        return (theta, phi)

    @staticmethod
    def gammas_to_spherical_coordinates(gamma1, gamma2):
        (theta, phi) = KentDistribution.gamma1_to_spherical_coordinates(gamma1)
        Ht = KentDistribution.create_matrix_Ht(theta, phi)
        u = np.inner(Ht, np.reshape(gamma2, (1, 3)))
        psi = np.arctan2(u[2][0], u[1][0])
        return (theta, phi, psi)

    def __init__(self, gamma1, gamma2, gamma3, kappa, beta):
        self.gamma1 = np.array(gamma1, dtype=np.float64)
        self.gamma2 = np.array(gamma2, dtype=np.float64)
        self.gamma3 = np.array(gamma3, dtype=np.float64)
        self.kappa = float(kappa)
        self.beta = float(beta)

        (
            self.theta,
            self.phi,
            self.psi,
        ) = KentDistribution.gammas_to_spherical_coordinates(self.gamma1, self.gamma2)

        for gamma in (gamma1, gamma2, gamma3):
            assert len(gamma) == 3

        self._cached_rvs = np.array([], dtype=np.float64)
        self._cached_rvs.shape = (0, 3)

    def __repr__(self):
        return "kent(%s, %s, %s, %s, %s)" % (
            self.theta,
            self.phi,
            self.psi,
            self.kappa,
            self.beta,
        )

    @property
    def Gamma(self):
        return self.create_matrix_Gamma(self.theta, self.phi, self.psi)

    def normalize(self, cache=dict(), return_num_iterations=False):
        """
        Returns the normalization constant of the Kent distribution.
        The proportional error may be expected not to be greater than
        1E-11.
        """

        (k, b) = (self.kappa, self.beta)
        if not (k, b) in cache:
            G = gamma_fun
            I = modified_bessel_2ndkind
            result = 0.0
            j = 0
            if b == 0.0:
                result = (0.5 * k) ** (-2 * j - 0.5) * I(2 * j + 0.5, k)
                result /= G(j + 1)
                result *= G(j + 0.5)
            else:

                while True:
                    a = np.exp(
                        np.log(b) * 2 * j + np.log(0.5 * k) * (-2 * j - 0.5)
                    ) * I(2 * j + 0.5, k)
                    a /= G(j + 1)
                    a *= G(j + 0.5)
                    result += a

                    j += 1
                    if abs(a) < abs(result) * 1E-12 and j > 5:
                        break

            cache[k, b] = 2 * np.pi * result
        if return_num_iterations:
            return (cache[k, b], j)
        else:
            return cache[k, b]

    def log_normalize(self, return_num_iterations=False):
        """
        Returns the logarithm of the normalization constant.
        """

        if return_num_iterations:
            (normalize, num_iter) = self.normalize(return_num_iterations=True)
            return (np.log(normalize), num_iter)
        else:
            return np.log(self.normalize())

    def pdf_max(self, normalize=True):
        return np.exp(self.log_pdf_max(normalize))

    def log_pdf_max(self, normalize=True):
        """
        Returns the maximum value of the log(pdf)
        """

        if self.beta == 0.0:
            x = 1
        else:
            x = self.kappa * 1.0 / (2 * self.beta)
        if x > 1.0:
            x = 1
        fmax = self.kappa * x + self.beta * (1 - x ** 2)
        if normalize:
            return fmax - self.log_normalize()
        else:
            return fmax

    def pdf(self, xs, normalize=True):
        """
        Returns the pdf of the kent distribution for 3D vectors that
        are stored in xs which must be an array of N x 3 or N x M x 3
        N x M x P x 3 etc.

        The code below shows how points in the pdf can be evaluated. An integral is
        calculated using random points on the sphere to determine wether the pdf is
        properly normalized.

        >>> from numpy.random import seed
        >>> from scipy.stats import norm as gauss
        >>> seed(666)
        >>> num_samples = 400000
        >>> xs = gauss(0, 1).rvs((num_samples, 3))
        >>> xs = divide(xs, reshape(norm(xs, 1), (num_samples, 1)))
        >>> assert abs(4*pi*average(kent(1.0, 1.0, 1.0, 4.0,  2.0).pdf(xs)) - 1.0) < 0.01
        >>> assert abs(4*pi*average(kent(1.0, 2.0, 3.0, 4.0,  2.0).pdf(xs)) - 1.0) < 0.01
        >>> assert abs(4*pi*average(kent(1.0, 2.0, 3.0, 4.0,  8.0).pdf(xs)) - 1.0) < 0.01
        >>> assert abs(4*pi*average(kent(1.0, 2.0, 3.0, 16.0, 8.0).pdf(xs)) - 1.0) < 0.01
        """

        return np.exp(self.log_pdf(xs, normalize))

    def log_pdf(self, xs, normalize=True):
        """
        Returns the log(pdf) of the kent distribution.
        """

        axis = len(np.shape(xs)) - 1
        g1x = np.sum(self.gamma1 * xs, axis)
        g2x = np.sum(self.gamma2 * xs, axis)
        g3x = np.sum(self.gamma3 * xs, axis)
        (k, b) = (self.kappa, self.beta)

        f = k * g1x + b * (g2x ** 2 - g3x ** 2)
        if normalize:
            return f - self.log_normalize()
        else:
            return f

    def pdf_prime(self, xs, normalize=True):
        """
        Returns the derivative of the pdf with respect to kappa and beta.
        """

        return self.pdf(xs, normalize) * self.log_pdf_prime(xs, normalize)

    def log_pdf_prime(self, xs, normalize=True):
        """
        Returns the derivative of the log(pdf) with respect to kappa and beta.
        """

        axis = len(np.shape(xs)) - 1
        g1x = np.sum(self.gamma1 * xs, axis)
        g2x = np.sum(self.gamma2 * xs, axis)
        g3x = np.sum(self.gamma3 * xs, axis)
        (k, b) = (self.kappa, self.beta)

        dfdk = g1x
        dfdb = g2x ** 2 - g3x ** 2
        df = np.array([dfdk, dfdb])
        if normalize:
            return np.transpose(np.transpose(df) - self.log_normalize_prime())
        else:
            return df

    def normalize_prime(self, cache=dict(), return_num_iterations=False):
        """
        Returns the derivative of the normalization factor with respect
        to kappa and beta.
        """

        (k, b) = (self.kappa, self.beta)
        if not (k, b) in cache:
            G = gamma_fun
            I = modified_bessel_2ndkind
            dIdk = lambda v, z: modified_bessel_2ndkind_derivative(v, z, 1)
            (dcdk, dcdb) = (0.0, 0.0)
            j = 0
            if b == 0:
                dcdk = (
                    G(j + 0.5)
                    / G(j + 1)
                    * ((-0.5 * j - 0.125) * k ** (-2 * j - 1.5))
                    * I(2 * j + 0.5, k)
                )
                dcdk += (
                    G(j + 0.5)
                    / G(j + 1)
                    * (0.5 * k) ** (-2 * j - 0.5)
                    * dIdk(2 * j + 0.5, k)
                )

                dcdb = 0.0
            else:
                while True:
                    dk = (
                        (-1 * j - 0.25)
                        * np.exp(np.log(b) * 2 * j + np.og(0.5 * k) * (-2 * j - 1.5))
                        * I(2 * j + 0.5, k)
                    )
                    dk += np.exp(
                        np.log(b) * 2 * j + np.log(0.5 * k) * (-2 * j - 0.5)
                    ) * dIdk(2 * j + 0.5, k)
                    dk /= G(j + 1)
                    dk *= G(j + 0.5)

                    db = (
                        2
                        * j
                        * np.exp(
                            np.log(b) * (2 * j - 1) + np.log(0.5 * k) * (-2 * j - 0.5)
                        )
                        * I(2 * j + 0.5, k)
                    )
                    db /= G(j + 1)
                    db *= G(j + 0.5)
                    dcdk += dk
                    dcdb += db

                    j += 1
                    if (
                        abs(dk) < abs(dcdk) * 1E-12
                        and abs(db) < abs(dcdb) * 1E-12
                        and j > 5
                    ):
                        break

            # print("dc", dcdk, dcdb, "(", k, b)

            cache[k, b] = 2 * np.pi * np.array([dcdk, dcdb])
        if return_num_iterations:
            return (cache[k, b], j)
        else:
            return cache[k, b]

    def log_normalize_prime(self, return_num_iterations=False):
        """
        Returns the derivative of the logarithm of the normalization factor.
        """

        if return_num_iterations:
            (normalize_prime, num_iter) = self.normalize_prime(
                return_num_iterations=True
            )
            return (normalize_prime / self.normalize(), num_iter)
        else:
            return self.normalize_prime() / self.normalize()

    def log_likelihood(self, xs):
        """
        Returns the log likelihood for xs.
        """

        retval = self.log_pdf(xs)
        return np.sum(retval, len(shape(retval)) - 1)

    def log_likelihood_prime(self, xs):
        """
        Returns the derivative with respect to kappa and beta of the log
        likelihood for xs.
        """

        retval = self.log_pdf_prime(xs)
        if len(shape(retval)) == 1:
            return retval
        else:
            return np.sum(retval, len(shape(retval)) - 1)

    def _rvs_helper(self):
        num_samples = 10000
        xs = gauss(0, 1).rvs((num_samples, 3))
        xs = np.divide(xs, np.reshape(np.linalg.norm(xs, axis=1), (num_samples, 1)))
        pvalues = self.pdf(xs, normalize=False)
        fmax = self.pdf_max(normalize=False)
        return xs[uniform(0, fmax).rvs(num_samples) < pvalues]

    def rvs(self, n_samples=None):
        """
        Returns random samples from the Kent distribution by rejection sampling.
        May become inefficient for large kappas.

        The returned random samples are 3D unit vectors.
        If n_samples == None then a single sample x is returned with shape (3,)
        If n_samples is an integer value N then N samples are returned in an array with shape (N, 3)
        """

        num_samples = 1 if n_samples is None else n_samples
        rvs = self._cached_rvs
        while len(rvs) < num_samples:
            new_rvs = self._rvs_helper()
            rvs = np.concatenate([rvs, new_rvs])
        if n_samples is None:
            self._cached_rvs = rvs[1:]
            return rvs[0]
        else:
            self._cached_rvs = rvs[num_samples:]
            retval = rvs[:num_samples]
            return retval
