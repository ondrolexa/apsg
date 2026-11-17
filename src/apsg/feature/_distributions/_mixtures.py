"""Finite mixtures of orientation distributions fitted by expectation-maximization."""

import numpy as np
from scipy.special import logsumexp

from apsg.feature._distributions._axial import Bingham
from apsg.feature._distributions._base import (
    FitInfo,
    OrientationDistribution,
    as_rng,
)
from apsg.feature._distributions._special import bingham_fit_kappas, bingham_log_norm
from apsg.feature._geodata import Lineation
from apsg.helpers._helper import check_count, check_weights


class BinghamMixture(OrientationDistribution):
    """Mixture of Bingham distributions of axes.

    The density is ``sum_k pi_k f_k(x)`` with ``Bingham`` components ``f_k``
    and mixing weights ``pi_k``. Each component is an elliptical cluster or a
    girdle, so the mixture models fracture sets of different shape and
    concentration (Yamaji & Sato, 2011), and the soft memberships
    (``responsibilities``) carry the overlap between the sets.

    Args:
        components: sequence of ``Bingham`` distributions.
        weights: mixing weights, normalized to sum 1. Default None, all equal.

    Fit the mixture with ``BinghamMixture.fit(data, k)`` and choose the number
    of components with ``BinghamMixture.select(data)``.

    Note:
        The components are ordered by decreasing weight when fitted, an
        arbitrary labelling, and the number of components `k` is an argument of
        ``fit``: ``linset.fit("bingham_mixture", k=3)`` or
        ``Vector3Set.fit_mixture`` (``k`` chosen by BIC) and
        ``Vector3Set.cluster_bingham`` (with the assignment of the data).

    References:
        Yamaji, A. & Sato, K. (2011). Clustering of fracture orientations using
            a mixed Bingham distribution and its application to paleostress
            analysis from dike or vein orientations. Journal of Structural
            Geology 33, 1148-1157.
    """

    dim = 3
    axial = True
    name = "bingham_mixture"
    feature = Lineation
    weighted = True

    def __init__(self, components, weights=None):
        components = tuple(components)
        if not components:
            raise ValueError("A mixture needs at least one component")
        if not all(isinstance(c, Bingham) for c in components):
            raise TypeError("The components must be Bingham distributions")
        if weights is None:
            weights = np.ones(len(components))
        self._components = components
        self._weights = check_weights(weights, len(components))

    @property
    def components(self):
        """tuple: the ``Bingham`` components."""
        return self._components

    @property
    def weights(self):
        """numpy.ndarray: mixing weights, summing to 1."""
        return self._weights

    @property
    def n_params(self):
        k = len(self._components)
        return 5 * k + k - 1

    @property
    def params(self):
        return {
            "components": self._components,
            "weights": tuple(float(w) for w in self._weights),
        }

    # density and sampling

    def _log_components(self, X):
        """``(N, K)`` array of ``log(pi_k f_k(x))``."""
        with np.errstate(divide="ignore"):
            logw = np.log(self._weights)
        return np.column_stack(
            [lw + c.logpdf_array(X) for lw, c in zip(logw, self._components)]
        )

    def logpdf_array(self, X):
        return logsumexp(self._log_components(X), axis=1)

    def rvs_array(self, n, rng):
        counts = rng.multinomial(n, self._weights)
        X = np.concatenate(
            [c.rvs_array(m, rng) for c, m in zip(self._components, counts) if m]
        )
        return X[rng.permutation(len(X))]

    # memberships

    def responsibilities(self, data):
        """Posterior probability of each component for every datum.

        Args:
            data: feature set or ``(N, 3)`` array of axes.

        Returns:
            numpy.ndarray: ``(N, K)`` array whose rows sum to 1.
        """
        lp = self._log_components(self.as_array(data))
        return np.exp(lp - logsumexp(lp, axis=1, keepdims=True))

    def labels(self, data):
        """Index of the most probable component of every datum.

        Args:
            data: feature set or ``(N, 3)`` array of axes.

        Returns:
            numpy.ndarray: ``(N,)`` integer array.
        """
        return np.argmax(self.responsibilities(data), axis=1)

    def split(self, data):
        """Split `data` into one feature set per component by ``labels``.

        Args:
            data: feature set of axes.

        Returns:
            tuple: ``K`` feature sets, the sets of the components in order
            (an empty set when no datum is closest to a component).
        """
        labels = self.labels(data)
        return tuple(data[np.flatnonzero(labels == k)] for k in range(len(self)))

    def __len__(self):
        return len(self._components)

    # fitting

    @classmethod
    def fit(
        cls,
        data,
        k=2,
        *,
        weights=None,
        n_init=10,
        tol=1e-6,
        max_iter=200,
        min_count=4.0,
        rng=None,
    ):
        """Fit a mixture of `k` Bingham distributions by EM.

        The E-step computes the memberships of the data, the M-step fits each
        component by weighted maximum likelihood (``Bingham.fit(weights=...)``)
        and sets the mixing weights to the total memberships. The iteration
        stops when the relative change of the log-likelihood is below `tol`.
        The likelihood has local maxima, so the EM is restarted `n_init` times
        (the first start from well separated data, the others from random
        data) and the best result is kept.

        Args:
            data: feature set (``LineationSet``, ``FoliationSet``, ...) or
                ``(N, 3)`` array of axes.
            k: number of components. Default 2.

        Keyword Args:
            weights: optional non-negative weight of each datum, e.g. Terzaghi
                weights; the likelihood is the weighted one. Default None.
            n_init: number of EM starts. Default 10.
            tol: relative tolerance of the log-likelihood. Default 1e-6.
            max_iter: maximum number of iterations of one start. Default 200.
            min_count: a start is discarded when a component has fewer
                effective data than this. Default 4.
            rng: None, an integer seed or a ``numpy.random.Generator``.

        Returns:
            BinghamMixture: the components are ordered by decreasing mixing
            weight; ``fit_info`` holds the (weighted) log-likelihood, ``n``,
            ``n_eff`` and in ``stats`` the number of ``iterations``, of
            ``starts`` and of ``converged_starts``.

        Raises:
            ValueError: if no start gives a valid fit, e.g. for too few data.
        """
        k = check_count(k, "k", minimum=1)
        n_init = check_count(n_init, "n_init", minimum=1)
        X = cls.as_array(data)
        n = len(X)
        if n < max(2, k):
            raise ValueError(f"A mixture of {k} components needs at least {k} data")
        w = np.full(n, 1.0 / n) if weights is None else check_weights(weights, n)
        n_eff = float(1.0 / np.sum(w**2))
        rng = as_rng(rng)

        if k == 1:
            dist = Bingham.fit(X, weights=w)
            mixture = cls([dist])
            mixture.fit_info = FitInfo(
                n=n,
                method="em",
                loglik=float(n * np.sum(w * mixture.logpdf_array(X))),
                converged=bool(dist.fit_info.converged),
                stats={"iterations": 0, "starts": 1, "converged_starts": 1},
                n_eff=n_eff,
            )
            return mixture

        best, converged_starts = None, 0
        for start in range(n_init):
            seeds = _seeds(X, w, k, rng, farthest=start == 0)
            result = _em(X, w, k, seeds, tol, max_iter, min_count)
            if result is None:
                continue
            converged_starts += result[3]
            # a converged start beats an unconverged one, then likelihood
            if best is None or (result[3], result[1]) > (best[3], best[1]):
                best = result
        if best is None:
            raise ValueError(
                f"Cannot fit a mixture of {k} Bingham distributions: no start "
                "gave components with enough data (try fewer components)"
            )
        comps, loglik, iterations, converged = best
        order = np.argsort([-p for p in comps[1]])
        mixture = cls([comps[0][i] for i in order], np.asarray(comps[1])[order])
        mixture.fit_info = FitInfo(
            n=n,
            method="em",
            loglik=float(n * loglik),
            converged=bool(converged),
            stats={
                "iterations": iterations,
                "starts": n_init,
                "converged_starts": converged_starts,
            },
            n_eff=n_eff,
        )
        return mixture

    @classmethod
    def select(cls, data, k_range=range(1, 6), *, criterion="bic", **options):
        """Fit mixtures with several numbers of components and pick the best.

        Args:
            data: feature set or ``(N, 3)`` array of axes.
            k_range: numbers of components to try. Default 1 to 5.

        Keyword Args:
            criterion: "bic" (default) or "aic", the criterion to minimize.
            **options: passed to ``fit``, e.g. ``weights``, ``n_init``, ``rng``.

        Returns:
            tuple: the best ``BinghamMixture`` and a list of dicts with the
            ``k``, ``loglik``, ``aic`` and ``bic`` of every fit that succeeded
            (too many components for the data are skipped).
        """
        if criterion not in ("bic", "aic"):
            raise ValueError(f"criterion must be 'bic' or 'aic', got {criterion!r}")
        table, fits = [], []
        for k in k_range:
            try:
                m = cls.fit(data, k, **options)
            except ValueError:
                continue
            X = cls.as_array(data)
            ll = m.fit_info.loglik
            aic = 2 * m.n_params - 2 * ll
            bic = m.n_params * np.log(len(X)) - 2 * ll
            table.append({"k": k, "loglik": ll, "aic": aic, "bic": bic})
            fits.append(m)
        if not fits:
            raise ValueError("No number of components could be fitted")
        best = int(np.argmin([row[criterion] for row in table]))
        return fits[best], table

    # serialization

    def to_json(self):
        return {
            "datatype": type(self).__name__,
            "args": (),
            "kwargs": {
                "components": [c.to_json() for c in self._components],
                "weights": [float(w) for w in self._weights],
            },
        }

    @classmethod
    def from_json(cls, obj_json):
        kwargs = obj_json["kwargs"]
        return cls(
            [Bingham.from_json(c) for c in kwargs["components"]], kwargs["weights"]
        )

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return len(self) == len(other) and (
            np.allclose(self._weights, other._weights)
            and all(a == b for a, b in zip(self._components, other._components))
        )

    def __hash__(self):
        return hash(type(self))

    def __repr__(self):
        parts = ", ".join(
            f"{w:.2f} * {c!r}" for w, c in zip(self._weights, self._components)
        )
        return f"BinghamMixture({parts})"


def _seeds(X, w, k, rng, farthest):
    """Indices of `k` data to start the components from."""
    if not farthest:
        p = w / w.sum()
        nonzero = int(np.count_nonzero(p))
        return rng.choice(len(X), size=min(k, nonzero), replace=False, p=p)
    idx = [int(np.argmax(w))]
    closeness = np.abs(X @ X[idx[0]])
    while len(idx) < k:
        idx.append(int(np.argmin(closeness)))
        closeness = np.maximum(closeness, np.abs(X @ X[idx[-1]]))
    return np.array(idx)


def _em(X, w, k, seeds, tol, max_iter, min_count):
    """One EM run; returns ``(components, weights), loglik, iterations, converged``
    or None when a component degenerates.

    The iteration works on arrays only: the weighted scatter matrices of all
    components come from one ``einsum``, their concentrations from
    ``bingham_fit_kappas`` (Newton, warm-started from the previous iteration),
    and ``Bingham`` objects are built once at the end. It is the weighted
    maximum-likelihood fit of ``Bingham._fit`` for every component.
    """
    if len(seeds) < k:
        return None
    # hard assignment to the nearest seed axis, softened so no component is empty
    nearest = np.argmax(np.abs(X @ X[seeds].T), axis=1)
    R = np.full((len(X), k), 0.1 / (k - 1))
    R[np.arange(len(X)), nearest] = 0.9
    previous = [None] * k
    prev = -np.inf
    converged = False
    for iteration in range(1, max_iter + 1):
        W = w[:, None] * R
        total = W.sum(axis=0)
        count = total**2 / np.sum(W**2, axis=0)
        if np.any(total <= 0) or np.any(count < min_count):
            return None
        scatter = np.einsum("nk,ni,nj->kij", W / total, X, X)
        taus, vecs = np.linalg.eigh(scatter)
        # rows of every frame: mean, major and minor axis (descending eigenvalues)
        frames = vecs[:, :, ::-1].transpose(0, 2, 1)
        taus = taus[:, ::-1]
        kappas = []
        for j in range(k):
            km, kn, _ = bingham_fit_kappas(taus[j, 1], taus[j, 2], previous[j])
            if not (np.isfinite(km) and np.isfinite(kn)) or kn < -1e10:
                return None
            kappas.append((km, kn))
        previous = kappas
        proj = np.einsum("ni,kji->nkj", X, frames[:, 1:, :]) ** 2
        lp = np.column_stack(
            [
                np.log(total[j])
                + km * proj[:, j, 0]
                + kn * proj[:, j, 1]
                - bingham_log_norm(km, kn)
                for j, (km, kn) in enumerate(kappas)
            ]
        )
        norm = logsumexp(lp, axis=1)
        R = np.exp(lp - norm[:, None])
        loglik = float(np.sum(w * norm))
        if abs(loglik - prev) <= tol * (abs(loglik) + tol):
            converged = True
            break
        prev = loglik
    comps = [
        Bingham(frames[j, 0], frames[j, 1], km, kn) for j, (km, kn) in enumerate(kappas)
    ]
    return (comps, [float(p) for p in total]), loglik, iteration, converged
