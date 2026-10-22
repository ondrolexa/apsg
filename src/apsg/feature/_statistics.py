import numpy as np
from scipy.stats import f as fdist


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
    return {
        "mean": mean,
        "eigenvalues": k,
        "ellipses": ellipses,
        "n": n,
        "level": level,
        "normalize": normalize,
        "anisoft": anisoft,
    }
