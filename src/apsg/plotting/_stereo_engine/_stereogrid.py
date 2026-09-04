"""``StereoGrid``: density/statistics on a uniform spherical grid.

Deliberately carries no projection/hemisphere concept and returns raw NED
vectors (not apsg ``Lineation`` objects) from ``min_at``/``max_at`` --
apsg's own ``apsg.plotting.StereoGrid`` (``_stereogrid.py``, one level up)
wraps this class and re-wraps those into ``Lineation`` for apsg users.
Projecting and drawing this class's ``values`` is ``StereonetAxes.contour``'s
job (see ``_axes.py``).
"""

import numpy as np

from ._utils import _as_vectors

try:
    import scipy.special

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

__all__ = ["StereoGrid"]


def _golden_section_spiral(n):
    """``n`` NED unit vectors, Golden Section Spiral -- apsg's
    ``Vector3Set.gss`` (``apsg/feature/_container.py``), formula copied
    verbatim (isotropic distribution, so the NED axis labeling is moot)."""
    inc = np.pi * (3 - np.sqrt(5))
    off = 2.0 / n
    k = np.arange(n)
    y = k * off - 1.0 + off / 2.0
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, None))
    phi = k * inc
    return np.column_stack([np.cos(phi) * r, y, np.sin(phi) * r])


def _fibonacci_spiral(n):
    """``n`` NED unit vectors, Spherical Fibonacci Spiral -- apsg's
    ``Vector3Set.sfs``, formula copied verbatim."""
    phi = (1.0 + np.sqrt(5)) / 2.0
    i2 = 2 * np.arange(n) - n + 1
    theta = 2 * np.pi * i2 / phi
    sp = i2 / n
    cp = np.sqrt((n + i2) * (n - i2)) / n
    return np.column_stack([cp * np.sin(theta), cp * np.cos(theta), sp])


def _real_sph_harm(n, m, polar, azimuthal):
    """Real spherical harmonic Y_n^m, from the complex form."""
    if hasattr(scipy.special, "sph_harm_y"):
        y_complex = scipy.special.sph_harm_y(n, abs(m), polar, azimuthal)
    else:  # pragma: no cover -- legacy scipy fallback
        y_complex = scipy.special.sph_harm(abs(m), n, azimuthal, polar)
    if m < 0:
        return np.sqrt(2) * y_complex.imag
    if m == 0:
        return y_complex.real
    return np.sqrt(2) * y_complex.real


def _evaluate_odf(coefficients, polar, azimuthal):
    odf_values = np.zeros_like(polar, dtype=float)
    for (n, m), c in coefficients.items():
        odf_values += c * _real_sph_harm(n, m, polar, azimuthal)
    return odf_values


class StereoGrid:
    """Values on a uniform grid of points over the whole sphere -- used to
    calculate continuous functions on the sphere, e.g. a density
    distribution of orientation data, for contour plotting.

    Args:
        grid_n (int): number of grid points. Default 2000 (matches apsg).
        grid_type (str): "gss" (Golden Section Spiral) or "sfs" (Spherical
            Fibonacci Spiral). Default "gss" (matches apsg).
    """

    def __init__(self, grid_n=2000, grid_type="gss"):
        self.grid_n = grid_n
        if grid_type == "gss":
            self.grid = _golden_section_spiral(grid_n)
        elif grid_type == "sfs":
            self.grid = _fibonacci_spiral(grid_n)
        else:
            raise ValueError("grid_type must be 'gss' or 'sfs'")
        self.grid_type = grid_type
        self.values = np.zeros(grid_n, dtype=float)
        self.calculated = False
        self.features = None

    def __repr__(self):
        info = ""
        if self.calculated:
            info = (
                f"\nMaximum: {self.max():.4f} at {self.max_at()}"
                f"\nMinimum: {self.min():.4f} at {self.min_at()}"
            )
        return f"StereoGrid ({self.grid_type}, {self.grid_n} points)" + info

    def min(self):
        """Return the minimum value of the grid."""
        return float(self.values.min())

    def max(self):
        """Return the maximum value of the grid."""
        return float(self.values.max())

    def min_at(self):
        """Return the NED unit vector where the grid is minimal."""
        return self.grid[self.values.argmin()].copy()

    def max_at(self):
        """Return the NED unit vector where the grid is maximal."""
        return self.grid[self.values.argmax()].copy()

    def calculate_density(self, features, method="sph", **kwargs):
        """Calculate density distribution of vectors from `features`.

        Both methods report the *same* statistic: standard deviations
        above the value expected under a uniform (random) distribution --
        i.e. "level=3" means "3 sigma", regardless of which method
        computed it.

        Args:
            features: array-like of shape (N, 3) -- NED unit vectors (axial
                data, e.g. poles or lines).

        Keyword Args:
            method (str): "sph" for the spherical-harmonics ODF method
                (requires scipy), or "kamb" for the modified-Kamb method
                with exponential smoothing (pure numpy). Default "sph".
            sigma (float): controls how much to smooth, for either method.
                Default 3 (Kamb 1959, Vollmer 1995). For "kamb", sets the
                kernel concentration `k = 2(1 + n/sigma^2)` directly. For
                "sph", determines the auto default for `n_max` (below).
            n_max (int): maximum harmonic degree, i.e. the angular
                resolution (for "sph"). Must be even. Default: derived
                from `sigma` and the sample size. Coefficients are always
                Lanczos-tapered by degree to suppress Gibbs-phenomenon
                ringing.
            trimzero (bool): if True, exact-zero values are bumped to a
                tiny positive number so a zero contour isn't drawn.
                Default True.
        """
        features = _as_vectors(features)
        self.features = features
        n = len(features)

        sigma = kwargs.get("sigma", None)
        if sigma is None:
            sigma = 3
        k = 2 * (1.0 + n / sigma**2)

        if method == "sph":
            if not _HAS_SCIPY:
                raise ImportError(
                    "calculate_density(method='sph') requires scipy; "
                    "install it, or use method='kamb' instead"
                )
            n_max = kwargs.get("n_max", None)
            if n_max is None:
                n_max = int(2 * round(1.15 * np.sqrt(k)))
                n_max = max(2, min(n_max, 40))
            if n_max % 2 != 0:
                raise ValueError("n_max must be an even integer for axial data.")
            polar_f = np.arccos(np.clip(features[:, 2], -1.0, 1.0))
            azimuthal_f = np.arctan2(features[:, 1], features[:, 0])
            coeffs = {}
            for deg in range(0, n_max + 1, 2):
                # Lanczos sigma factor tapers higher-degree coefficients to
                # suppress Gibbs-phenomenon ringing near concentrated peaks.
                taper = np.sinc(deg / (n_max + 1))
                for m in range(-deg, deg + 1):
                    y_nm = _real_sph_harm(deg, m, polar_f, azimuthal_f)
                    coeffs[(deg, m)] = taper * np.sum(y_nm) / n
            c_0_0 = coeffs[(0, 0)]
            y_0_0 = 1.0 / np.sqrt(4 * np.pi)
            mean_odf_value = c_0_0 * y_0_0
            mud_coeffs = {k_: c / mean_odf_value for k_, c in coeffs.items()}
            polar_g = np.arccos(np.clip(self.grid[:, 2], -1.0, 1.0))
            azimuthal_g = np.arctan2(self.grid[:, 1], self.grid[:, 0])
            # mean=1 (MUD) exactly, by construction. Converted to "sigma
            # above uniform" below: under the uniform (null) hypothesis
            # Var[odf(g)] = sum_{l=2,4,...,n_max} weight(l)^2 * (2l+1) / n,
            # the same everywhere on the sphere -- deliberately *not*
            # clipped to >=0 first, this is a genuine two-sided Z.
            odf = _evaluate_odf(mud_coeffs, polar_g, azimuthal_g)
            sd_null = np.sqrt(
                sum(
                    np.sinc(ll / (n_max + 1)) ** 2 * (2 * ll + 1)
                    for ll in range(2, n_max + 1, 2)
                )
                / n
            )
            self.values = (odf - 1.0) / sd_null
        elif method == "kamb":
            # scale is Vollmer's (1995) normalizer that makes
            # cnt.sum(axis=1)/scale have SD 1 under the uniform/null
            # distribution -- but its *mean* under that null is `sigma`
            # itself, not 0. Subtracting `sigma` directly gives a proper Z
            # statistic (mean 0, SD 1 under uniform data). Not clipped, for
            # the same reason `sph` isn't.
            scale = np.sqrt(n * (k / 2.0 - 1) / k**2)
            cnt = np.exp(k * (np.abs(self.grid @ features.T) - 1))
            self.values = cnt.sum(axis=1) / scale - sigma
        else:
            raise ValueError(f"Unknown method {method!r}, expected 'sph' or 'kamb'")

        if kwargs.get("trimzero", True):
            self.values[self.values == 0] = np.finfo(float).tiny
        self.calculated = True

    def apply_func(self, func, *args, **kwargs):
        """Calculate values of a user-defined function on the sphere.

        Args:
            func (callable): called as ``func(grid_point, *args, **kwargs)``
                for each grid point (a NED unit-vector ``(3,)`` array),
                must return a scalar.
        """
        for i in range(self.grid_n):
            self.values[i] = func(self.grid[i], *args, **kwargs)
        self.calculated = True

    def angmech(self, fault_normals, slip_vectors, method="classic", **kwargs):
        """Angelier-Mechler dihedra method for fault-slip data.

        Args:
            fault_normals: array-like of shape (N, 3) -- NED unit vectors,
                normal to each fault plane.
            slip_vectors: array-like of shape (N, 3) -- NED unit vectors,
                slip direction on each fault plane.

        Keyword Args:
            method (str): "classic" assigns +/-1 to individual positions;
                "probability" additionally weights by a maximum-likelihood
                estimate. Default "classic".
        """
        normals = _as_vectors(fault_normals)
        slips = _as_vectors(slip_vectors)
        if len(normals) != len(slips):
            raise ValueError("fault_normals and slip_vectors must be the same length")

        values = np.zeros(self.grid_n, dtype=float)
        for normal, slip in zip(normals, slips):
            dot_n = self.grid @ normal
            dot_l = self.grid @ slip
            dist = 2.0 * (np.sign(dot_n) == np.sign(dot_l)) - 1.0
            if method == "probability":
                lprob = 1 - np.abs(2 * (np.abs(dot_l) - 0.5))
                fprob = 1 - np.abs(2 * (np.abs(dot_n) - 0.5))
                dist = dist * lprob * fprob
            elif method != "classic":
                raise ValueError(
                    f"Unknown method {method!r}, expected 'classic' or 'probability'"
                )
            values += dist
        self.values = values
        self.calculated = True
