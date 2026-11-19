from dataclasses import dataclass

import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as tdist

from apsg.feature._hypothesis import TestResult
from apsg.math._vector import Vector3


def jelinek_statistics(tensors, level=0.95, normalize=False, anisoft=False):
    """Mean symmetric tensor and confidence ellipses of its principal axes.

    Linear perturbation method of Jelinek (1978), which builds upon Hext (1963)
    matrix statistics. Each tensor is rotated to the principal frame of the mean
    tensor. The deviation of principal axis `i` towards axis `j` is, to first
    order, ``K_ij / (k_i - k_j)`` where `K_ij` is off-diagonal element of the
    rotated tensor and `k` are eigenvalues of the mean tensor. Confidence
    region of the mean is derived from Hotelling's T² for the sample covariance
    of these deviations.

    Args:
        tensors (ndarray): array of shape (n, 3, 3) of symmetric tensors.
        level (float): confidence level. Default 0.95.
        normalize (bool): divide each tensor by its absolute mean value
            ``abs(trace) / 3`` before averaging. Default False.
        anisoft (bool): multiply the tilts of the principal axes by an extra
            factor ``(n - 1) / n`` (the covariance by ``((n - 1) / n)²``), as done by
            the jelinekstat and TomoFab packages, which reproduces their published
            values. It shrinks the ellipses (by about 9-12 % in angle for n = 8)
            and lowers their coverage below the nominal level. Default False.

    Returns:
        dict: ``mean`` (3x3 mean tensor), ``eigenvalues`` (descending),
        ``ellipses`` (list of three dicts with keys ``mu`` (principal axis),
        ``axes`` (unit vectors of the ellipse axes in the tangent plane) and
        ``gamma`` (semi-angles in degrees, corresponding to ``axes``)),
        ``n``, ``level``, ``normalize`` and ``anisoft``. Index in ``ellipses`` is the
        principal axis of the mean tensor (0 is the largest eigenvalue).
        ``intervals`` holds the ``(3, 2)`` confidence intervals of the
        eigenvalues (Student's t, Jelínek, 1978, eq. 35), ``anisotropy_test``
        the ``TestResult`` of ``anisotropy_test`` (for ``n > 5``) and, for
        positive definite mean tensors, ``shape`` the parameters of
        ``jelinek_shape``.

    References:
        Hext, G.R. (1963). The estimation of second-order tensors, with related
            tests and designs. Biometrika 50, 353-373.
        Jelinek, V. (1978). Statistical processing of anisotropy of magnetic
            susceptibility measured on groups of specimens. Studia Geophysica
            et Geodaetica 22, 50-62.
    """
    tensors = np.asarray(tensors, dtype=float)
    n = len(tensors)
    if n < 3:
        raise ValueError("At least 3 tensors are required")
    if normalize:
        scale = np.abs(np.trace(tensors, axis1=1, axis2=2)) / 3
        if np.any(scale <= 1e-12 * np.abs(tensors).max()):
            raise ValueError("Cannot normalize tensors with zero trace")
        tensors = tensors / scale[:, None, None]
    mean = tensors.mean(axis=0)
    k, P = np.linalg.eigh(mean)
    k, P = k[::-1], P[:, ::-1]
    rotated = np.einsum("ki,nkl,lj->nij", P, tensors, P)
    tol = 1e-9 * np.abs(k).max()
    # critical value of Hotelling's T² for a 2-D mean (the two tilts of an
    # axis) from n samples: T² = 2 (n - 1) / (n - 2) F(2, n - 2)
    T0 = 2 * (n - 1) / (n - 2) * fdist.ppf(level, 2, n - 2)
    scale = ((n - 1) / n) ** 2 if anisoft else 1.0
    ellipses = []
    for i in range(3):
        j, l = [m for m in range(3) if m != i]
        dj, dl = k[i] - k[j], k[i] - k[l]
        if abs(dj) <= tol or abs(dl) <= tol:
            axes, gamma = (P[:, j], P[:, l]), (90.0, 90.0)
        else:
            # covariance of the first-order tilts of axis i towards j and l;
            # its eigenvectors orient the ellipse, T0 * eigenvalues / n give
            # the squared tangents of its semi-angles
            W = np.cov(np.vstack([rotated[:, i, j] / dj, rotated[:, i, l] / dl]))
            lam, U = np.linalg.eigh(W)
            lam, U = np.clip(lam[::-1], 0, None), U[:, ::-1]
            axes = tuple(U[0, m] * P[:, j] + U[1, m] * P[:, l] for m in range(2))
            gamma = tuple(np.degrees(np.arctan(np.sqrt(scale * T0 * lam / n))))
        ellipses.append({"mu": P[:, i], "axes": axes, "gamma": gamma})
    result = {
        "mean": mean,
        "eigenvalues": k,
        "ellipses": ellipses,
        "n": n,
        "level": level,
        "normalize": normalize,
        "anisoft": anisoft,
    }
    # t intervals of the principal values (Jelinek, 1978, eq. 35)
    half = tdist.ppf((1 + level) / 2, n - 1) * np.sqrt(
        np.var(np.diagonal(rotated, axis1=1, axis2=2), axis=0, ddof=1) / n
    )
    result["intervals"] = np.column_stack([k - half, k + half])
    if n > 5:
        result["anisotropy_test"] = anisotropy_test(tensors, level)
    if k[-1] > 0:
        result["shape"] = jelinek_shape(k)
    return result


def _deviatoric(K):
    """Five orthonormal components of the deviatoric part of symmetric tensors."""
    K = np.asarray(K, dtype=float)
    r2 = np.sqrt(2)
    return np.column_stack(
        [
            (K[:, 0, 0] - K[:, 1, 1]) / r2,
            (K[:, 0, 0] + K[:, 1, 1] - 2 * K[:, 2, 2]) / np.sqrt(6),
            r2 * K[:, 0, 1],
            r2 * K[:, 1, 2],
            r2 * K[:, 0, 2],
        ]
    )


def _trace_normalized(tensors):
    """Divide each tensor by ``abs(trace) / 3``."""
    scale = np.abs(np.trace(tensors, axis1=1, axis2=2)) / 3
    if np.any(scale <= 1e-12 * np.abs(tensors).max()):
        raise ValueError("Cannot normalize tensors with zero trace")
    return tensors / scale[:, None, None]


def anisotropy_test(tensors, level=0.95):
    """Hotelling's T² test that the mean of symmetric tensors is isotropic.

    The deviatoric part of each tensor is expressed by its five components in
    an orthonormal basis of traceless symmetric matrices; under the null
    hypothesis (mean tensor a multiple of the identity) their mean is zero and
    ``T² (n - 5) / (5 (n - 1))`` follows ``F(5, n - 5)`` (Jelínek, 1978, eq.
    44-46). Hotelling's T² does not depend on the choice of the basis.

    Args:
        tensors: ``(n, 3, 3)`` array of symmetric tensors, ``n > 5``.
        level: confidence level. Default 0.95.

    Returns:
        TestResult: ``reject()`` is True when the mean tensor is anisotropic.

    References:
        Jelínek, V. (1978). Statistical processing of anisotropy of magnetic
            susceptibility measured on groups of specimens. Studia Geophysica
            et Geodaetica 22, 50-62.
    """
    K = np.asarray(tensors, dtype=float)
    n = len(K)
    if n <= 5:
        raise ValueError(f"The anisotropy test needs more than 5 tensors, got {n}")
    X = _deviatoric(K)
    xbar = X.mean(axis=0)
    T2 = float(n * xbar @ np.linalg.pinv(np.cov(X, rowvar=False)) @ xbar)
    scale = 5 * (n - 1) / (n - 5)
    return TestResult(
        name="Jelinek anisotropy",
        statistic=T2,
        p_value=float(fdist.sf(T2 / scale, 5, n - 5)),
        df=(5, n - 5),
        critical=float(scale * fdist.ppf(level, 5, n - 5)),
        level=level,
        source="Jelinek (1978), eq. 44-46",
    )


def jelinek_shape(eigenvalues):
    """Anisotropy and shape parameters of a positive definite tensor.

    The parameters of Jelínek (1981) for the principal values ``k1 >= k2 >= k3``
    of e.g. a magnetic susceptibility tensor: lineation ``L = k1 / k2``,
    foliation ``F = k2 / k3``, anisotropy degree ``P = k1 / k3``, corrected
    anisotropy degree ``Pj = exp(sqrt(2 sum((eta_i - eta)^2)))`` with ``eta_i =
    ln(k_i)`` and ``eta`` their mean, shape parameter ``T = (2 eta_2 - eta_1 -
    eta_3) / (eta_1 - eta_3)`` (-1 prolate, 0 neutral, +1 oblate) and ``U = (2
    k2 - k1 - k3) / (k1 - k3)``.

    Note:
        The principal values are used directly. ``Ellipsoid.P_j`` and
        ``Ellipsoid.T`` take the principal stretches, the square roots of the
        eigenvalues of a finite strain tensor, so for a susceptibility tensor
        ``Ellipsoid.P_j`` is the square root of ``Pj`` (``T`` is the same). The
        corrected anisotropy degree of Borradaile (2003, eq. 10.24), centred on
        the logarithm of the arithmetic mean, differs from Jelínek's.

    Args:
        eigenvalues: three positive principal values, in any order.

    Returns:
        dict: with keys ``L``, ``F``, ``P``, ``Pj``, ``T`` and ``U``.

    References:
        Jelínek, V. (1981). Characterization of the magnetic fabric of rocks.
            Tectonophysics 79, T63-T67, eq. 2-4, 7, 9.
    """
    k = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    if k.shape != (3,) or not np.all(np.isfinite(k)) or k[-1] <= 0:
        raise ValueError(f"Three positive principal values are required, got {k}")
    eta = np.log(k)
    spread = eta[0] - eta[2]
    return {
        "L": float(k[0] / k[1]),
        "F": float(k[1] / k[2]),
        "P": float(k[0] / k[2]),
        "Pj": float(np.exp(np.sqrt(2 * np.sum((eta - eta.mean()) ** 2)))),
        "T": float((2 * eta[1] - eta[0] - eta[2]) / spread) if spread > 0 else 0.0,
        "U": float((2 * k[1] - k[0] - k[2]) / (k[0] - k[2])) if spread > 0 else 0.0,
    }


@dataclass(frozen=True, eq=False)
class TensorBootstrap:
    """Bootstrap of the mean of symmetric tensors (Constable and Tauxe, 1990).

    Attributes:
        mean: ``(3, 3)`` mean tensor of the sample.
        eigenvalues: ``(n_resamples, 3)`` eigenvalues of the bootstrapped mean
            tensors, descending.
        eigenvectors: ``(n_resamples, 3, 3)`` their eigenvectors in columns,
            each turned to the side of the corresponding axis of `mean`.
        ellipses: tuple of three dicts, one per principal axis (0 is the largest
            eigenvalue), with ``mu`` (mean bootstrapped axis), ``axes`` (the two
            semi-axes of the ellipse, perpendicular to ``mu``), ``gamma``
            (semi-angles in degrees) and ``which``, the layout of
            ``jelinek_statistics``.
        intervals: ``(3, 2)`` percentile intervals of the eigenvalues.
        level: confidence level of `ellipses` and `intervals`.
        n: number of tensors (or None for a parametric bootstrap).
        n_resamples: number of bootstrap samples.
        normalize: whether tensors were divided by their mean.
    """

    mean: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    ellipses: tuple
    intervals: np.ndarray
    level: float
    n: int | None
    n_resamples: int
    normalize: bool

    def __repr__(self):
        return (
            f"TensorBootstrap(n={self.n}, n_resamples={self.n_resamples}, "
            f"level={self.level})"
        )

    def shape_parameters(self):
        """Parameters of Jelinek (1981) of every bootstrapped mean tensor.

        Returns:
            dict: arrays ``L``, ``F``, ``P``, ``Pj`` and ``T`` of length
            ``n_resamples`` (``nan`` where an eigenvalue is not positive).
        """
        k = self.eigenvalues
        with np.errstate(all="ignore"):
            eta = np.log(np.where(k > 0, k, np.nan))
            spread = eta[:, 0] - eta[:, 2]
            return {
                "L": k[:, 0] / k[:, 1],
                "F": k[:, 1] / k[:, 2],
                "P": k[:, 0] / k[:, 2],
                "Pj": np.exp(
                    np.sqrt(2 * np.sum((eta - eta.mean(axis=1)[:, None]) ** 2, axis=1))
                ),
                "T": (2 * eta[:, 1] - eta[:, 0] - eta[:, 2]) / spread,
            }

    def parameter_intervals(self):
        """Percentile intervals (at `level`) of `shape_parameters`.

        Returns:
            dict: name -> ``(lower, upper)``.
        """
        a = 100 * (1 - self.level) / 2
        return {
            name: tuple(np.nanpercentile(v, [a, 100 - a]))
            for name, v in self.shape_parameters().items()
        }


def _axis_region(directions, level):
    """Kent-type ellipse containing `level` of the unit vectors `directions`.

    The mean direction is the normalised resultant (length ``mu``), the
    semi-axes follow from the second moments ``sigma_2^2, sigma_3^2`` of the
    components perpendicular to it: ``arcsin(sigma sqrt(g))`` with ``g = -2
    ln(1 - level) / mu^2`` (Constable and Tauxe, 1990, eq. 13-15), 90 degrees
    when the sine reaches 1.
    """
    resultant = directions.mean(axis=0)
    mu = np.linalg.norm(resultant)
    m = resultant / mu
    t = directions - np.outer(directions @ m, m)
    lam, U = np.linalg.eigh(t.T @ t / len(t))
    lam, U = lam[::-1], U[:, ::-1]
    g = -2 * np.log(1 - level) / mu**2
    gamma = np.degrees(np.arcsin(np.minimum(1, np.sqrt(np.clip(lam[:2], 0, None) * g))))
    return m, (U[:, 0], U[:, 1]), gamma


def _summarize_bootstrap(means, ref, level, n, normalize):
    """`TensorBootstrap` of the bootstrapped mean tensors `means` ``(B, 3, 3)``."""
    k, V = np.linalg.eigh(means)
    k, V = k[:, ::-1], V[:, :, ::-1]
    _, Vref = np.linalg.eigh(ref)
    Vref = Vref[:, ::-1]
    sign = np.sign(np.einsum("bki,ki->bi", V, Vref))
    sign[sign == 0] = 1
    V = V * sign[:, None, :]
    ellipses = []
    for i in range(3):
        mu, axes, gamma = _axis_region(V[:, :, i], level)
        ellipses.append(
            {
                "mu": Vector3(mu),
                "axes": tuple(Vector3(a) for a in axes),
                "gamma": tuple(float(x) for x in gamma),
                "which": i,
            }
        )
    a = 100 * (1 - level) / 2
    intervals = np.percentile(k, [a, 100 - a], axis=0).T
    return TensorBootstrap(
        mean=ref,
        eigenvalues=k,
        eigenvectors=V,
        ellipses=tuple(ellipses),
        intervals=intervals,
        level=level,
        n=n,
        n_resamples=len(means),
        normalize=normalize,
    )


def tensor_bootstrap(tensors, n_resamples=1000, level=0.95, normalize=False, rng=None):
    """Nonparametric bootstrap of the mean of symmetric tensors.

    The tensors (optionally divided by their mean ``abs(trace) / 3``, as for
    anisotropy of magnetic susceptibility) are resampled with replacement and
    the eigenvectors and eigenvalues of each mean are collected. Unlike the
    linear perturbation method of Jelinek (1978), it does not assume that the
    errors of the principal values and directions are small and normal, which
    fails when two eigenvalues are close (Constable and Tauxe, 1990). The
    ellipse of an axis contains `level` of the bootstrapped axes.

    Note:
        The region describes the bootstrap distribution; for small samples it
        covers the true axis less often than `level` (about 82 % at level 0.95
        for 8 specimens with close eigenvalues, in simulation, against 91 % of
        the T² region of ``jelinek_statistics``), so treat it as a measure of
        spread rather than a guaranteed confidence region.

    Args:
        tensors: ``(n, 3, 3)`` array of symmetric tensors, ``n >= 2``.
        n_resamples: number of bootstrap samples. Default 1000.
        level: confidence level. Default 0.95.
        normalize: divide each tensor by ``abs(trace) / 3``. Default False.
        rng: None, an integer seed or a ``numpy.random.Generator``. With None
            numpy's global random state seeds the generator.

    Returns:
        TensorBootstrap: eigenvalues, eigenvectors and confidence regions.

    References:
        Constable, C., Tauxe, L. (1990). The bootstrap for magnetic
            susceptibility tensors. J. Geophys. Res. 95(B6), 8383-8395.
    """
    from apsg.feature._distributions._base import as_rng

    K = np.asarray(tensors, dtype=float)
    if K.ndim != 3 or K.shape[1:] != (3, 3):
        raise ValueError("tensors must have shape (n, 3, 3)")
    n = len(K)
    if n < 2:
        raise ValueError("The bootstrap needs at least 2 tensors")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    if int(n_resamples) < 10:
        raise ValueError("n_resamples must be at least 10")
    if normalize:
        K = _trace_normalized(K)
    K = (K + K.transpose(0, 2, 1)) / 2
    rng = as_rng(rng)
    idx = rng.integers(0, n, size=(int(n_resamples), n))
    means = K[idx].mean(axis=1)
    return _summarize_bootstrap(means, K.mean(axis=0), level, n, normalize)


def tensor_error_bootstrap(
    tensor,
    sigma,
    n_resamples=1000,
    level=0.95,
    normalize=True,
    rng=None,
    rotation=None,
):
    """Parametric bootstrap of one tensor with normal measurement errors.

    Each of the six independent components of the tensor gets independent
    normal noise of standard deviation `sigma` (the standard error of the
    instrument), and the noisy tensors are divided by their mean when
    `normalize` is True, which makes the noise of the normalised components
    non-normal (Constable and Tauxe, 1990). The result describes the
    uncertainty of the axes and eigenvalues of one specimen.

    Args:
        tensor: symmetric ``(3, 3)`` tensor.
        sigma: standard deviation of the components, in the units of `tensor`.
        n_resamples: number of simulated tensors. Default 1000.
        level: confidence level. Default 0.95.
        normalize: divide each tensor by ``abs(trace) / 3``. Default True.
        rng: None, an integer seed or a ``numpy.random.Generator``.
        rotation: optional 3x3 rotation matrix ``R``; the noise is added in the
            frame of `tensor` and the simulated tensors are rotated, ``R T R'``,
            e.g. to geographic coordinates.

    Returns:
        TensorBootstrap: ``eigenvalues`` and ``eigenvectors`` of the simulated
        tensors, with ``n`` None.
    """
    from apsg.feature._distributions._base import as_rng

    T = np.asarray(tensor, dtype=float)
    if T.shape != (3, 3):
        raise ValueError("tensor must have shape (3, 3)")
    if not sigma >= 0:
        raise ValueError("sigma must be non-negative")
    if int(n_resamples) < 10:
        raise ValueError("n_resamples must be at least 10")
    rng = as_rng(rng)
    T = (T + T.T) / 2
    noise = rng.normal(scale=sigma, size=(int(n_resamples), 3, 3))
    noise = np.triu(noise) + np.triu(noise, 1).transpose(0, 2, 1)
    sims = T + noise
    ref = T
    if rotation is not None:
        R = np.asarray(rotation, dtype=float)
        sims = R @ sims @ R.T
        ref = R @ T @ R.T
    if normalize:
        sims = _trace_normalized(sims)
        ref = ref / (np.trace(T) / 3)
    return _summarize_bootstrap(sims, ref, level, None, normalize)


def compare_tensor_means(
    a, b, level=0.95, normalize=False, method="hotelling", n_resamples=2000, rng=None
):
    """Test whether two samples of symmetric tensors have the same mean tensor.

    Hotelling's T² of the difference of the two sample means is formed from
    the five deviatoric components of the tensors (the same orthonormal basis
    as in ``anisotropy_test``), with the pooled covariance matrix. It compares
    the shape and orientation of the tensors; the mean ``trace / 3`` is not
    among the components, so differences of the bulk susceptibility do not
    matter. Two ways to judge the statistic:

    - ``"hotelling"``: ``T² (n - 6) / (5 (n - 2))`` follows ``F(5, n - 6)``,
      ``n = n_a + n_b``, for normal components with a common covariance matrix
      (the two-sample extension of Jelinek's test of isotropy, 1978, eq.
      44-46). The components of normalised tensors are not exactly normal
      (Constable and Tauxe, 1990).
    - ``"permutation"``: the same statistic against its distribution under
      random reassignment of the tensors to the two samples, which makes no
      assumption about the distribution.

    Args:
        a: ``(n_a, 3, 3)`` array of symmetric tensors.
        b: ``(n_b, 3, 3)`` array of symmetric tensors, ``n_a + n_b >= 7``.
        level: confidence level. Default 0.95.
        normalize: divide each tensor by ``abs(trace) / 3`` first, as usual for
            anisotropy of magnetic susceptibility of specimens with different
            bulk susceptibility. Default False.
        method: ``"hotelling"`` (default) or ``"permutation"``.
        n_resamples: number of permutations. Default 2000.
        rng: None, an integer seed or a ``numpy.random.Generator``.

    Returns:
        TestResult: ``reject()`` is True when the mean tensors differ.
    """
    from scipy.stats import f as fdist

    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    for K in (A, B):
        if K.ndim != 3 or K.shape[1:] != (3, 3):
            raise ValueError("tensors must have shape (n, 3, 3)")
    n1, n2 = len(A), len(B)
    if n1 < 2 or n2 < 2 or n1 + n2 < 7:
        raise ValueError(
            "The comparison needs at least 2 tensors per sample and 7 in total, "
            f"got {n1} and {n2}"
        )
    if method not in ("hotelling", "permutation"):
        raise ValueError(f"Unknown method {method!r}")
    if normalize:
        A, B = _trace_normalized(A), _trace_normalized(B)
    X1, X2 = _deviatoric(A), _deviatoric(B)
    n = n1 + n2

    def t2(x1, x2):
        d = x1.mean(axis=0) - x2.mean(axis=0)
        S = (
            np.cov(x1, rowvar=False) * (len(x1) - 1)
            + np.cov(x2, rowvar=False) * (len(x2) - 1)
        ) / (n - 2)
        return float(n1 * n2 / n * d @ np.linalg.pinv(S) @ d)

    stat = t2(X1, X2)
    if method == "hotelling":
        scale = 5 * (n - 2) / (n - 5 - 1)
        return TestResult(
            name="Jelinek two-sample tensor",
            statistic=stat,
            p_value=float(fdist.sf(stat / scale, 5, n - 6)),
            df=(5, n - 6),
            critical=float(scale * fdist.ppf(level, 5, n - 6)),
            level=level,
            source="Hotelling T² of the deviatoric components (Jelinek, 1978, eq. 44-46)",
            extra={"n": (n1, n2)},
        )
    from apsg.feature._distributions._base import as_rng

    if int(n_resamples) < 99:
        raise ValueError("n_resamples must be at least 99")
    rng = as_rng(rng)
    X = np.vstack([X1, X2])
    count = 0
    stats = np.empty(int(n_resamples))
    for i in range(int(n_resamples)):
        perm = rng.permutation(n)
        stats[i] = t2(X[perm[:n1]], X[perm[n1:]])
    count = int(np.sum(stats >= stat))
    return TestResult(
        name="Jelinek two-sample tensor (permutation)",
        statistic=stat,
        p_value=(1 + count) / (1 + len(stats)),
        df=None,
        critical=float(np.quantile(stats, level)),
        level=level,
        source="permutation test of Hotelling T² of the deviatoric components",
        extra={"n": (n1, n2), "n_resamples": len(stats)},
    )
