"""
Distributions of rotations (directed orthogonal frames).
"""

import numpy as np
from scipy import special
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
from scipy.spatial.transform import Rotation

from apsg.feature._distributions._base import OrientationDistribution, as_rng
from apsg.feature._tensor3 import Rotation3
from apsg.helpers._helper import check_count


def rotation_arrays(data):
    """Return rotations as an ``(N, 3, 3)`` array of rotation matrices.

    Args:
        data: ``Rotation3``, ``RotationSet``, sequence of ``Rotation3`` or array
            of shape ``(3, 3)`` or ``(N, 3, 3)``.
    """
    if isinstance(data, Rotation3):
        X = np.asarray(data, dtype=float)[None]
    else:
        X = np.asarray([np.asarray(r, dtype=float) for r in data], dtype=float)
        if X.shape == (3, 3):
            X = X[None]
    if X.ndim != 3 or X.shape[1:] != (3, 3):
        raise ValueError(f"Expected rotation matrices, got shape {X.shape}")
    eye = np.einsum("nji,njk->nik", X, X)
    if not np.allclose(eye, np.eye(3), atol=1e-8):
        raise ValueError("Rotation matrices must be orthogonal")
    if np.any(np.linalg.det(X) < 0):
        raise ValueError("Rotation matrices must have determinant 1")
    return X


def proper_svd(A):
    """Proper singular value decomposition ``A = U diag(d) V'``.

    ``U`` and ``V`` are rotations; the last singular value takes the sign of
    ``det(A)``, so ``d1 >= d2 >= |d3|``.

    Returns:
        tuple: ``U``, signed singular values ``d`` and ``V``.
    """
    U, s, Vt = np.linalg.svd(np.asarray(A, dtype=float))
    du, dv = np.linalg.det(U), np.linalg.det(Vt)
    U[:, 2] *= du
    Vt[2] *= dv
    return U, s * np.array([1, 1, du * dv]), Vt.T


def matrix_fisher_log_norm(psi):
    """Log of the normalizer of the matrix Fisher distribution.

    ``c(Psi) = 1/2 int_-1^1 I0((psi_1 - psi_2)(1 - u) / 2) I0((psi_1 + psi_2)(1
    + u) / 2) exp(psi_3 u) du`` for the signed singular values ``psi`` of
    ``F``, the normalizer with respect to the uniform distribution of rotations
    (Kent, Ganeiber & Mardia, 2013; López-Custodio, 2025, eq. 44).
    """
    p1, p2, p3 = psi
    a, b = (p1 - p2) / 2, (p1 + p2) / 2

    def log_integrand(u):
        x, y = a * (1 - u), b * (1 + u)
        return (
            np.log(special.ive(0, x)) + abs(x) + np.log(special.ive(0, y)) + abs(y)
        ) + p3 * u

    peak = max(log_integrand(-1.0), log_integrand(1.0), log_integrand(0.0))
    value = quad(lambda u: np.exp(log_integrand(u) - peak), -1, 1, limit=200)[0]
    return float(np.log(value / 2) + peak)


class MatrixFisher(OrientationDistribution):
    """Matrix Fisher distribution of rotations.

    The density with respect to the uniform (Haar) distribution of rotations
    is ``exp(tr(F' R)) / c(F)`` (Kent, Ganeiber & Mardia, 2013, eq. 4.2). With
    the proper singular value decomposition ``F = U diag(psi) V'`` the mode is
    the rotation ``M = U V'`` and ``psi_1 >= psi_2 >= |psi_3|`` are the
    concentrations; ``F = kappa M`` is the isotropic case.

    Rotations describe directed orthogonal frames, e.g. a lineation with sense
    and a foliation normal with a defined side. Axial frames such as ``Pair``
    have four rotations each and need the ``Lisle`` distribution or Lisle's
    tensor instead.

    Args:
        F: the 3x3 parameter matrix. Default ``I`` (mode the identity,
            concentrations 1).

    The fit is the maximum-likelihood estimate: the mode from the proper
    singular value decomposition of the mean rotation matrix ``U diag(d) V'``
    (with signed singular values ``d``; Mardia & Jupp, 2000, eq. 13.2.16), the
    concentrations solving ``d log c / d psi_i = d_i``. Sampling maps a
    Bingham distribution of unit quaternions to rotations, drawn by rejection
    from an angular central Gaussian envelope (Kent, Ganeiber & Mardia, 2013,
    sec. 5.3).

    References:
        Kent, J.T., Ganeiber, A.M. & Mardia, K.V. (2013). A new method to
            simulate the Bingham and related distributions in directional data
            analysis with applications. arXiv:1310.8110.
        Mardia, K.V. & Jupp, P.E. (2000). Directional Statistics. Wiley.
    """

    dim = "rotation"
    axial = False
    n_params = 9
    name = "matrix_fisher"
    feature = Rotation3

    def __init__(self, F=None):
        F = np.eye(3) if F is None else np.asarray(F, dtype=float)
        if F.shape != (3, 3) or not np.all(np.isfinite(F)):
            raise ValueError(f"F must be a finite 3x3 matrix, got {F}")
        self.F = F
        self._U, self._psi, self._V = proper_svd(F)
        self._log_norm = matrix_fisher_log_norm(self._psi)

    @classmethod
    def from_mode(cls, mode, kappa=1):
        """Isotropic matrix Fisher distribution ``F = kappa M`` about `mode`."""
        return cls(kappa * np.asarray(Rotation3(mode), dtype=float))

    @property
    def params(self):
        return {"F": self.F}

    @property
    def mean(self):
        """Rotation3: the mode ``U V'``."""
        return Rotation3(self._U @ self._V.T)

    @property
    def concentrations(self):
        """ndarray: signed singular values ``psi`` of ``F``."""
        return self._psi.copy()

    @classmethod
    def as_array(cls, x):
        return rotation_arrays(x)

    @staticmethod
    def _is_single(x):
        return isinstance(x, Rotation3) or np.ndim(x) == 2

    def logpdf_array(self, X):
        return np.einsum("ij,nij->n", self.F, X) - self._log_norm

    def rvs_array(self, n, rng):
        # tr(Psi R(q)) = q' B q for unit quaternions q = (w, x, y, z)
        p1, p2, p3 = self._psi
        B = np.array(
            [p1 + p2 + p3, p1 - p2 - p3, -p1 + p2 - p3, -p1 - p2 + p3], dtype=float
        )
        lam = B.max() - B  # exp(-q' A q), A = diag(lam) >= 0
        b = brentq(lambda b: np.sum(1 / (b + 2 * lam)) - 1, 1e-12, 4.0)
        omega = 1 + 2 * lam / b
        log_m = -(4 - b) / 2 + 2 * np.log(4 / b)
        Q = np.empty((0, 4))
        while len(Q) < n:
            m = max(2 * (n - len(Q)), 16)
            Y = rng.standard_normal((m, 4)) / np.sqrt(omega)
            Y = Y / np.linalg.norm(Y, axis=1)[:, None]
            log_ratio = -(Y**2 @ lam) + 2 * np.log(Y**2 @ omega) - log_m
            Q = np.concatenate([Q, Y[np.log(rng.uniform(size=m)) < log_ratio]])
        R0 = Rotation.from_quat(Q[:n], scalar_first=True).as_matrix()
        return self._U @ R0 @ self._V.T

    def rvs(self, n=100, *, rng=None, feature=None, name="Default"):
        """Return random sample of rotations.

        Args:
            n: sample size. Default 100.

        Keyword Args:
            rng: None, an integer seed or a ``numpy.random.Generator``.
            feature: ignored, the sample holds ``Rotation3``.
            name: name of the returned set. Default 'Default'.

        Returns:
            RotationSet: the sample.
        """
        from apsg.feature._container import RotationSet

        X = self.rvs_array(check_count(n), as_rng(rng))
        return RotationSet([Rotation3(x) for x in X], name=name)

    @classmethod
    def _fit(cls, X, method="mle", **options):
        if method != "mle":
            raise ValueError(f"Unknown fitting method {method!r}")
        U, d, V = proper_svd(X.mean(axis=0))

        def nll(psi):
            return -(psi @ d - matrix_fisher_log_norm(psi))

        # concentrated data: 1 - d_i ~ (sum of the other two inverse psi) / 2
        start = 1 / np.maximum(1 - d, 1e-3)
        res = minimize(nll, start, method="Nelder-Mead", options=_NM_OPTIONS)
        F = U @ np.diag(res.x) @ V.T
        return {"F": F}, bool(res.success), {"singular_values": tuple(d)}


_NM_OPTIONS = {"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000}
