import math
import numpy as np

from scipy.special import gamma as gamma_fun
from scipy.special import iv as modified_bessel_2ndkind
from scipy.special import ivp as modified_bessel_2ndkind_derivative
from scipy.stats import uniform
from scipy.stats import norm as gauss
from scipy.optimize import minimize_scalar


def vonMisesFisher(mu, kappa, num_samples):
    """Generate N samples from von Mises Fisher
    distribution around center mu in R^N with concentration kappa.

    Adopted from https://github.com/jasonlaska/spherecluster
    """

    def _sample_weight(kappa):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        b = 2 / (np.sqrt(4.0 * kappa**2 + 4) + 2 * kappa)
        x = (1.0 - b) / (1.0 + b)
        c = kappa * x + 2 * np.log(1 - x**2)

        while True:
            z = np.random.beta(1, 1)
            w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + 2 * np.log(1.0 - x * w) - c >= np.log(u):
                return w

    def _sample_orthonormal_to(mu):
        """Sample point on sphere orthogonal to mu."""
        v = np.random.randn(mu.shape[0])
        proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
        orthto = v - proj_mu_v
        return orthto / np.linalg.norm(orthto)

    result = np.zeros((num_samples, 3))
    for nn in range(num_samples):
        # sample offset from center (on sphere) with spread kappa
        w = _sample_weight(kappa)

        # sample a point v on the unit sphere that's orthogonal to mu
        v = _sample_orthonormal_to(mu)

        # compute new point
        result[nn, :] = v * np.sqrt(1.0 - w**2) + w * mu

    return result


def estimate_k(features):
    # objective function to be minimized
    def obj_fun(k):
        W = np.exp(
            k * (np.abs(np.dot(np.asarray(features), np.asarray(features).T)))
        ) * (k / (4 * math.pi * math.sinh(k + 1e-9)))
        np.fill_diagonal(W, 0.0)
        return -np.log(W.sum(axis=0)).sum()

    if len(features) > 1:
        return minimize_scalar(obj_fun, bounds=(0.1, len(features)), method="bounded").x
    else:
        return 1


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

    minimum_value_for_kappa = 1e-6

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
        if (k, b) not in cache:
            G = gamma_fun
            Imb2 = modified_bessel_2ndkind
            result = 0.0
            j = 0
            if b == 0.0:
                result = (0.5 * k) ** (-2 * j - 0.5) * Imb2(2 * j + 0.5, k)
                result /= G(j + 1)
                result *= G(j + 0.5)
            else:

                while True:
                    a = np.exp(
                        np.log(b) * 2 * j + np.log(0.5 * k) * (-2 * j - 0.5)
                    ) * Imb2(2 * j + 0.5, k)
                    a /= G(j + 1)
                    a *= G(j + 0.5)
                    result += a

                    j += 1
                    if abs(a) < abs(result) * 1e-12 and j > 5:
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
        fmax = self.kappa * x + self.beta * (1 - x**2)
        if normalize:
            return fmax - self.log_normalize()
        else:
            return fmax

    def pdf(self, xs, normalize=True):
        """
        Returns the pdf of the kent distribution for 3D vectors that
        are stored in xs which must be an array of N x 3 or N x M x 3
        N x M x P x 3 etc.
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

        f = k * g1x + b * (g2x**2 - g3x**2)
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

        dfdk = g1x
        dfdb = g2x**2 - g3x**2
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
        if (k, b) not in cache:
            G = gamma_fun
            Imb2 = modified_bessel_2ndkind
            (dcdk, dcdb) = (0.0, 0.0)
            j = 0
            if b == 0:
                dcdk = (
                    G(j + 0.5)
                    / G(j + 1)
                    * ((-0.5 * j - 0.125) * k ** (-2 * j - 1.5))
                    * Imb2(2 * j + 0.5, k)
                )
                dcdk += (
                    G(j + 0.5)
                    / G(j + 1)
                    * (0.5 * k) ** (-2 * j - 0.5)
                    * modified_bessel_2ndkind_derivative(2 * j + 0.5, k, 1)
                )

                dcdb = 0.0
            else:
                while True:
                    dk = (
                        (-1 * j - 0.25)
                        * np.exp(np.log(b) * 2 * j + np.og(0.5 * k) * (-2 * j - 1.5))
                        * Imb2(2 * j + 0.5, k)
                    )
                    dk += np.exp(
                        np.log(b) * 2 * j + np.log(0.5 * k) * (-2 * j - 0.5)
                    ) * modified_bessel_2ndkind_derivative(2 * j + 0.5, k, 1)
                    dk /= G(j + 1)
                    dk *= G(j + 0.5)

                    db = (
                        2
                        * j
                        * np.exp(
                            np.log(b) * (2 * j - 1) + np.log(0.5 * k) * (-2 * j - 0.5)
                        )
                        * Imb2(2 * j + 0.5, k)
                    )
                    db /= G(j + 1)
                    db *= G(j + 0.5)
                    dcdk += dk
                    dcdb += db

                    j += 1
                    if (
                        abs(dk) < abs(dcdk) * 1e-12
                        and abs(db) < abs(dcdb) * 1e-12
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
        return np.sum(retval, len(np.shape(retval)) - 1)

    def log_likelihood_prime(self, xs):
        """
        Returns the derivative with respect to kappa and beta of the log
        likelihood for xs.
        """

        retval = self.log_pdf_prime(xs)
        if len(np.shape(retval)) == 1:
            return retval
        else:
            return np.sum(retval, len(np.shape(retval)) - 1)

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
