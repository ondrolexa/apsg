import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as tdist

from apsg.feature._hypothesis import TestResult


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
    r2 = np.sqrt(2)
    X = np.column_stack(
        [
            (K[:, 0, 0] - K[:, 1, 1]) / r2,
            (K[:, 0, 0] + K[:, 1, 1] - 2 * K[:, 2, 2]) / np.sqrt(6),
            r2 * K[:, 0, 1],
            r2 * K[:, 1, 2],
            r2 * K[:, 0, 2],
        ]
    )
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
