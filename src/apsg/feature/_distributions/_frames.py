"""
Distributions of the orientations of pairs (foliation and lineation).

A ``Pair`` is a frame ``R = [l, m, n]`` (lineation, ``m = n x l``, foliation
normal). Reversing the lineation or the normal gives the same pair, so a pair
is a frame up to 180 degree rotations about its three axes.
"""

import numpy as np
from scipy import special, stats
from scipy.optimize import minimize

from apsg.feature._distributions._axial import Bingham, bingham_ellipse
from apsg.feature._distributions._base import (
    OrientationDistribution,
    as_rng,
    finite_param,
)
from apsg.feature._distributions._special import lisle_log_norm, restricted_eigen
from apsg.feature._geodata import Lineation, Pair
from apsg.helpers._helper import check_count
from apsg.math._vector import Vector3


def pair_arrays(data):
    """Return unit lineations and foliation normals of pairs as ``(N, 2, 3)``.

    Args:
        data: ``Pair``, ``PairSet`` (or ``FaultSet``), ``(N, 2, 3)`` array of
            ``[l, n]`` or ``(N, 6)`` array of ``[n, l]`` (the layout of
            ``np.asarray(pair)``).
    """
    from apsg.feature._container import PairSet

    if isinstance(data, PairSet):
        L, N = np.array(data.lvec, dtype=float), np.array(data.fvec, dtype=float)
    elif isinstance(data, Pair):
        L = np.atleast_2d(np.asarray(data.lvec, dtype=float))
        N = np.atleast_2d(np.asarray(data.fvec, dtype=float))
    else:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 3 and arr.shape[1:] == (2, 3):
            L, N = arr[:, 0], arr[:, 1]
        else:
            arr = np.atleast_2d(arr)
            if arr.shape[1] < 6:
                raise ValueError(f"Expected pairs, got array of shape {arr.shape}")
            N, L = arr[:, :3], arr[:, 3:6]
    L = L / np.linalg.norm(L, axis=1)[:, None]
    N = N / np.linalg.norm(N, axis=1)[:, None]
    return np.stack([L, N], axis=1)


def lisle_moments(X):
    """Lisle's (1989) tensor of pairs and the moments of its eigenvectors.

    Args:
        X: ``(N, 2, 3)`` array of unit lineations and normals.

    Returns:
        tuple: eigenvalues (descending, sum 0) and eigenvectors (rows) of
        ``mean(l l' - n n')``, and the matrix ``m_ij = mean((v_i' A v_j)^2)``
        with ``A = l l' - n n'`` used by the confidence ellipses.
    """
    L, N = X[:, 0], X[:, 1]
    T = (L.T @ L - N.T @ N) / len(X)
    taus, vecs = np.linalg.eigh(T)
    taus, vecs = taus[::-1], vecs[:, ::-1].T
    PL, PN = L @ vecs.T, N @ vecs.T
    A = PL[:, :, None] * PL[:, None, :] - PN[:, :, None] * PN[:, None, :]
    return taus, vecs, np.mean(A**2, axis=0)


def lisle_ellipses(taus, vecs, moments, n, level=0.95):
    """Confidence ellipses of the mean lineation and the mean foliation normal.

    Large-sample (central limit theorem) regions of the first and the last
    eigenvector of Lisle's tensor, as ``bingham_ellipse`` for axes.

    Returns:
        tuple: ``ConfidenceRegion`` of the lineation and of the normal.
    """
    axes = [Lineation(v) for v in vecs]
    lin = bingham_ellipse(taus, axes, moments, n, level, which=0)
    fol = bingham_ellipse(taus, axes, moments, n, level, which=2)
    return lin, fol


class Lisle(OrientationDistribution):
    """Lisle distribution of pairs (foliation and lineation).

    The density with respect to the uniform distribution of pairs is
    ``exp(l' L l - n' L n) / c(kappa1, kappa2)``, with the lineation `l` and the
    foliation normal `n` in the frame of the mean pair and
    ``L = diag(kappa1 + kappa2, kappa2, 0)``. ``kappa1`` controls the spread of
    the lineation within the foliation (rotation about the normal), ``kappa2``
    the rotation of the foliation about the lineation.

    It is the exponential family whose sufficient statistic is Lisle's (1989)
    tensor ``sum(l l' - n n')``, so its maximum-likelihood mean pair is Lisle's
    mean (eigenvectors of the tensor). Reversing `l` or `n` does not change the
    density.

    Args:
        mean: mean ``Pair``. Default ``Pair()``.
        kappa1: concentration of the lineation within the foliation. Default 1.
        kappa2: concentration of the foliation about the lineation. Default 1.

    The normalizer is evaluated by quadrature (``lisle_log_norm``). Sampling
    draws the lineation from its marginal by rejection from a ``Bingham``
    envelope and the normal from its axial von Mises conditional.
    """

    dim = "pair"
    axial = True
    n_params = 5
    name = "lisle"
    feature = Pair

    def __init__(self, mean=None, kappa1=1, kappa2=1):
        mean = Pair() if mean is None else mean
        X = pair_arrays(mean if isinstance(mean, Pair) else Pair(mean))[0]
        l, n = X
        self._frame = np.array([l, np.cross(n, l), n])
        kappa1 = finite_param("kappa1", kappa1)
        kappa2 = finite_param("kappa2", kappa2)
        if not (kappa1 >= 0 and kappa2 >= 0):
            raise ValueError("kappa1 and kappa2 must be non-negative")
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self._lam = np.array([self.kappa1 + self.kappa2, self.kappa2, 0.0])
        self._log_norm = lisle_log_norm(self.kappa1, self.kappa2)

    @property
    def params(self):
        return {"mean": self.mean, "kappa1": self.kappa1, "kappa2": self.kappa2}

    @property
    def mean(self):
        return Pair(Vector3(self._frame[2]), Vector3(self._frame[0]))

    @classmethod
    def as_array(cls, x):
        return pair_arrays(x)

    @staticmethod
    def _is_single(x):
        return isinstance(x, Pair) or np.ndim(x) == 1

    def logpdf_array(self, X):
        Lc, Nc = X[:, 0] @ self._frame.T, X[:, 1] @ self._frame.T
        return (Lc**2) @ self._lam - (Nc**2) @ self._lam - self._log_norm

    def rvs_array(self, n, rng):
        lam = self._lam
        envelope = Bingham((1, 0, 0), (0, 1, 0), -self.kappa1, -lam[0])
        L = np.empty((0, 3))
        while len(L) < n:
            m = max(2 * (n - len(L)), 16)
            cand = envelope.rvs_array(m, rng)
            _, s, d = restricted_eigen(lam, cand)
            accept = np.exp(-(s - d) / 2) * special.i0e(d / 2)
            L = np.concatenate([L, cand[rng.uniform(size=m) < accept]])
        L = L[:n]
        # normal in the plane perpendicular to l, along the eigenvectors (a, b)
        # of L restricted to it: n' L n = const + (mu1 - mu2) / 2 cos(2 phi)
        P = np.eye(3) - L[:, :, None] * L[:, None, :]
        w, V = np.linalg.eigh(P @ np.diag(lam) @ P)
        along = np.abs(np.einsum("nij,ni->nj", V, L))
        order = np.argsort(along, axis=1)[:, :2]
        idx = np.arange(n)[:, None]
        mu, vecs = w[idx, order], np.transpose(V, (0, 2, 1))[idx, order]
        swap = mu[:, 0] < mu[:, 1]
        a = np.where(swap[:, None], vecs[:, 1], vecs[:, 0])
        b = np.where(swap[:, None], vecs[:, 0], vecs[:, 1])
        conc = np.maximum(np.abs(mu[:, 0] - mu[:, 1]) / 2, 1e-12)
        phi = stats.vonmises.rvs(conc, loc=np.pi, random_state=rng) / 2
        N = np.cos(phi)[:, None] * a + np.sin(phi)[:, None] * b
        return np.stack([L @ self._frame, N @ self._frame], axis=1)

    def rvs(self, n=100, *, rng=None, feature=None, name="Default"):
        """Return random sample of pairs.

        Args:
            n: sample size. Default 100.

        Keyword Args:
            rng: None, an integer seed or a ``numpy.random.Generator``.
            feature: ``Pair`` (default) or a subclass without extra data.
            name: name of the returned set. Default 'Default'.

        Returns:
            PairSet: the sample, with random signs of lineations and normals.
        """
        from apsg.feature._container import PairSet

        feature = Pair if feature is None else feature
        if not issubclass(feature, Pair) or feature.__shape__ != Pair.__shape__:
            raise TypeError(f"{feature.__name__} is not a pair feature")
        n = check_count(n)
        rng = as_rng(rng)
        X = self.rvs_array(n, rng)
        X = X * rng.choice([-1.0, 1.0], size=(n, 2, 1))
        return PairSet([feature(Vector3(x[1]), Vector3(x[0])) for x in X], name=name)

    @classmethod
    def _fit(cls, X, method="mle", **options):
        if method != "mle":
            raise ValueError(f"Unknown fitting method {method!r}")
        taus, vecs, moments = lisle_moments(X)

        def nll(p):
            k1, k2 = np.exp(p)
            return -((k1 + k2) * taus[0] + k2 * taus[1] - lisle_log_norm(k1, k2))

        # concentrated data: 1 - tau1 and 1 + tau3 are small
        start = [
            np.log(1 / max(1 - taus[0], 1e-3)),
            np.log(1 / max(1 + taus[2], 1e-3)),
        ]
        res = minimize(nll, start, method="Nelder-Mead", options=_NM_OPTIONS)
        k1, k2 = np.exp(res.x)
        params = {
            "mean": Pair(Vector3(vecs[2]), Vector3(vecs[0])),
            "kappa1": float(k1),
            "kappa2": float(k2),
        }
        stats_ = {"eigenvalues": tuple(taus), "axes": tuple(vecs), "moments": moments}
        return params, bool(res.success), stats_

    def mean_confidence(self, level=0.95, which="lin"):
        """Confidence region of the mean lineation or the mean foliation normal.

        Args:
            level: confidence level. Default 0.95.
            which: "lin" for the lineation or "fol" for the foliation normal.
                Default "lin".

        Returns:
            ConfidenceRegion: ellipse, see ``lisle_ellipses``.
        """
        if self.fit_info is None:
            raise ValueError("Confidence region needs a distribution fitted to data")
        if which not in ("lin", "fol"):
            raise ValueError("which must be 'lin' or 'fol'")
        s = self.fit_info.stats
        lin, fol = lisle_ellipses(
            s["eigenvalues"], s["axes"], s["moments"], self.fit_info.n, level
        )
        return lin if which == "lin" else fol


_NM_OPTIONS = {"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000}
