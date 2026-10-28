"""Sets of ellipses, ellipsoids, orientation tensors, stress tensors and
rotations."""

import numpy as np
from scipy import linalg as spla

from apsg.config import stress_sign
from apsg.feature._container._base import FeatureSet, _per_item
from apsg.feature._container._vectorsets import Vector2Set, Vector3Set
from apsg.feature._statistics import (
    jelinek_statistics,
)
from apsg.feature._tensor2 import Ellipse, OrientationTensor2, Stress2
from apsg.feature._tensor3 import (
    Ellipsoid,
    OrientationTensor3,
    Rotation3,
    Stress3,
)
from apsg.helpers._helper import check_count
from apsg.helpers._math import atan2d, cosd, sind
from apsg.math._vector import Vector3


def _mean_tensor(tensor_set, level, normalize, anisoft):
    """Mean tensor and principal axes confidence ellipses of a tensor set.

    Shared implementation of ``EllipsoidSet.mean_tensor`` and
    ``Stress3Set.mean_tensor``.
    """
    tensor_cls = tensor_set.__feature_class__
    result = jelinek_statistics(
        np.array([np.asarray(t) for t in tensor_set]),
        level=level,
        normalize=normalize,
        anisoft=anisoft,
    )
    eigenvalues = result["eigenvalues"]
    if issubclass(tensor_cls, (Stress2, Stress3)):
        # internal matrices are mechanical; report eigenvalues in the active convention
        eigenvalues = stress_sign() * eigenvalues
    ellipses = tuple(
        {
            "mu": Vector3(e["mu"]),
            "axes": tuple(Vector3(a) for a in e["axes"]),
            "gamma": tuple(float(g) for g in e["gamma"]),
            "which": which,
        }
        for which, e in enumerate(result["ellipses"])
    )
    stress = issubclass(tensor_cls, (Stress2, Stress3))
    intervals = result["intervals"]
    if stress and stress_sign() < 0:
        intervals = -intervals[:, ::-1]
    out = {
        "mean": tensor_cls._from_canonical(result["mean"]),
        "eigenvalues": eigenvalues,
        "ellipses": ellipses,
        "intervals": intervals,
        "n": result["n"],
        "level": result["level"],
        "normalize": result["normalize"],
        "anisoft": result["anisoft"],
    }
    if "anisotropy_test" in result:
        out["anisotropy_test"] = result["anisotropy_test"]
    if "shape" in result and not stress:
        out["shape"] = result["shape"]
    return out


class EllipseSet(FeatureSet):
    """
    Class to store set of ``Ellipse`` features.
    """

    __feature_class__ = Ellipse

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, Ellipse) for obj in data):
            raise TypeError("Data must be instances of Ellipse")

    def __repr__(self):
        return f"E2({len(self)}) {self.name}"

    S1 = _per_item("S1", "Return the array of maximum principal stretches.")
    S2 = _per_item("S2", "Return the array of minimum principal stretches.")
    e1 = _per_item("e1", "Return the maximum natural principal strains.")
    e2 = _per_item("e2", "Return the array of minimum natural principal strains.")
    ar = _per_item("ar", "Return the array of axial ratios.")
    orientation = _per_item(
        "orientation", "Return the array of orientations of the maximum eigenvector."
    )
    e12 = _per_item(
        "e12", "Return the array of differences between natural principal strains."
    )

    def transform(self, F):
        """Return transformation of all features ``EllipseSet`` by matrix 'F'.

        Args:
            F: Transformation matrix. Array-like value e.g. ``DeformationGradient3``

        Returns:
            EllipseSet: The transformed feature set.
        """
        return type(self)([e.transform(F) for e in self], name=self.name)

    def rfphi(self) -> Ellipse:
        """Return strain ellipse from the Rf/phi analytical vector-mean method.

        Doubles each ellipse orientation (ellipses have 180° periodicity) and
        averages ``ln(Rf)`` in that doubled-angle Cartesian space before
        converting back, following the standard Rf/phi vector-mean technique:

            x_i = ln(Rf_i) * cos(2 * phi_i)
            y_i = ln(Rf_i) * sin(2 * phi_i)
            Rs = exp(sqrt(mean(x)**2 + mean(y)**2))
            phi_s = atan2d(mean(y), mean(x)) / 2

        Returns:
            Ellipse: strain ellipse with axial ratio ``Rs`` and orientation ``phi_s``.
        """
        phi2 = np.radians(2 * self.orientation)
        x = np.log(self.ar) * np.cos(phi2)
        y = np.log(self.ar) * np.sin(phi2)
        xbar, ybar = x.mean(), y.mean()
        Rs = np.exp(np.sqrt(xbar**2 + ybar**2))
        phi_s = atan2d(ybar, xbar) / 2
        c, s = cosd(phi_s), sind(phi_s)
        R = np.array([[c, -s], [s, c]])
        return Ellipse(R @ np.diag([Rs, 1 / Rs]) @ R.T)

    def robin(self) -> Ellipse:
        """Return mean strain ellipse using the log-matrix method of Robin (1977).

        Averages the matrix logarithms of all tensors in this set and maps the
        result back with the matrix exponential (the log-Euclidean tensor mean),
        avoiding the distortion a naive linear average of the tensors themselves
        would introduce for anisotropic/high-strain ellipses.

        Returns:
            Ellipse: mean strain ellipse.
        """
        mean_log = np.mean([spla.logm(np.asarray(e)) for e in self], axis=0)
        return Ellipse(spla.expm(mean_log).real)


class OrientationTensor2Set(EllipseSet):
    """
    Class to store set of ``OrientationTensor2`` features.
    """

    __feature_class__ = OrientationTensor2

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, OrientationTensor2) for obj in data):
            raise TypeError("Data must be instances of OrientationTensor2")

    def __repr__(self):
        return f"M2({len(self)}) {self.name}"


class EllipsoidSet(FeatureSet):
    """
    Class to store set of ``Ellipsoid`` features.
    """

    __feature_class__ = Ellipsoid

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, Ellipsoid) for obj in data):
            raise TypeError("Data must be instances of Ellipsoid")

    def __repr__(self):
        return f"E({len(self)}) {self.name}"

    kind = _per_item("kind", "Return the array of descriptive types of ellipsoids.")
    strength = _per_item("strength", "Return the array of the Woodcock strength.")
    shape = _per_item("shape", "Return the array of the Woodcock shape.")
    S1 = _per_item("S1", "Return the array of maximum principal stretches.")
    S2 = _per_item("S2", "Return the array of middle principal stretches.")
    S3 = _per_item("S3", "Return the array of minimum principal stretches.")
    e1 = _per_item("e1", "Return the array of the maximum natural principal strain.")
    e2 = _per_item("e2", "Return the array of the middle natural principal strain.")
    e3 = _per_item("e3", "Return the array of the minimum natural principal strain.")
    Rxy = _per_item("Rxy", "Return the array of the Rxy ratios.")
    Ryz = _per_item("Ryz", "Return the array of the Ryz ratios.")
    e12 = _per_item("e12", "Return the array of the e1 - e2 values.")
    e13 = _per_item("e13", "Return the array of the e1 - e3 values.")
    e23 = _per_item("e23", "Return the array of the e2 - e3 values.")
    k = _per_item("k", "Return the array of the strain symmetries.")
    d = _per_item("d", "Return the array of the strain intensities.")
    K = _per_item("K", "Return the array of the strain symmetries K (Ramsay, 1983).")
    D = _per_item("D", "Return the array of the strain intensities D (Ramsay, 1983).")
    r = _per_item("r", "Return the array of the strain intensities (Watterson, 1968).")
    goct = _per_item(
        "goct", "Return the array of the natural octahedral unit shears (Nadai, 1963)."
    )
    eoct = _per_item(
        "eoct", "Return the array of the natural octahedral unit strains (Nadai, 1963)."
    )
    lode = _per_item("lode", "Return the array of Lode parameters (Lode, 1926).")
    P_j = _per_item(
        "P_j", "Return the array of corrected anisotropy degrees (Jelínek, 1981)."
    )
    T = _per_item("T", "Return the array of shape parameters (Jelínek, 1981).")
    P = _per_item("P", "Return the array of Point indexes (Vollmer, 1990).")
    G = _per_item("G", "Return the array of Girdle indexes (Vollmer, 1990).")
    R = _per_item("R", "Return the array of Random indexes (Vollmer, 1990).")
    B = _per_item("B", "Return the array of Cylindricity indexes (Vollmer, 1990).")
    Intensity = _per_item(
        "Intensity", "Return the array of Intensity indexes (Lisle, 1985)."
    )
    MAD_l = _per_item(
        "MAD_l",
        "Return maximum angular deviation (MAD) of linearly distributed vectors.",
    )
    MAD_p = _per_item(
        "MAD_p",
        "Return maximum angular deviation (MAD) of planarly distributed vectors.",
    )
    MAD = _per_item("MAD", "Return approximate deviation according to shape")

    def mean_tensor(self, level=0.95, normalize=False, anisoft=False):
        """Return mean tensor and confidence ellipses of its principal axes.

        Uses the classic linear perturbation method of Jelinek (1978), which
        builds upon Hext (1963) matrix statistics. Each tensor is rotated to
        the principal frame of the mean tensor and the sample covariance of the
        first-order angular deviations of each principal axis towards the other
        two axes gives the confidence ellipse (Hotelling's T² with
        ``F(2, n - 2)``).

        Jelinek, V. (1978). Statistical processing of anisotropy of magnetic
            susceptibility measured on groups of specimens. Studia Geophysica
            et Geodaetica, 22, 50-62.
        Hext, G.R. (1963). The estimation of second-order tensors, with related
            tests and designs. Biometrika, 50, 353-373.

        Note:
            The covariance is the unbiased sample covariance and the region is
            the exact Hotelling T² region (in simulation its coverage matches
            the nominal level). Use ``anisoft=True`` to reproduce
            implementations (jelinekstat, TomoFab) that additionally scale the
            covariance by ``((n - 1) / n)²``.

        Args:
            level (float): confidence level. Default 0.95 for 95 %.
            normalize (bool): divide each tensor by its absolute mean value
                ``abs(trace) / 3`` before averaging, as usual for anisotropy
                of magnetic susceptibility. Default False. Fails for tensors
                with zero trace.
            anisoft (bool): multiply the tilts of the principal axes by an extra
                factor ``(n - 1) / n`` (the covariance by ``((n - 1) / n)²``),
                which shrinks the ellipses (by about 9-12 % in angle for n = 8) and
                lowers their coverage below the nominal level. Default False.

        Returns:
            dict: with keys ``mean`` (mean tensor), ``eigenvalues`` (of the
            mean tensor, descending), ``ellipses`` (tuple of three dicts, one
            for each principal axis of the mean tensor, 0 is the largest
            eigenvalue, with keys ``mu`` (principal axis), ``axes`` (tuple of
            the two ellipse axes, perpendicular to ``mu``), ``gamma`` (tuple
            of the corresponding semi-angles in degrees) and ``which``),
            ``intervals`` (``(3, 2)`` confidence intervals of the eigenvalues,
            Jelínek, 1978, eq. 35), ``anisotropy_test`` (``TestResult`` of
            Jelínek's T² test of isotropy, for more than 5 tensors), ``shape``
            (``jelinek_shape`` parameters of the mean, for positive definite
            ellipsoids), ``n`` (sample size), ``level``, ``normalize`` and
            ``anisoft``.

        Raises:
            ValueError: for less than 3 tensors, or when ``normalize`` is used
                on tensors with zero trace.
        """
        self._require(3, "mean_tensor")
        return _mean_tensor(self, level, normalize, anisoft)

    def transform(self, F):
        """Return transformation of all features ``EllipsoidSet`` by matrix 'F'.

        Args:
            F: Transformation matrix. Array-like value e.g. ``DeformationGradient3``

        Returns:
            EllipsoidSet: The transformed feature set.
        """
        return type(self)([e.transform(F) for e in self], name=self.name)

    def robin(self) -> Ellipsoid:
        """Return mean strain ellipsoid using the log-matrix method of Robin (1977).

        Averages the matrix logarithms of all tensors in this set and maps the
        result back with the matrix exponential (the log-Euclidean tensor mean),
        avoiding the distortion a naive linear average of the tensors themselves
        would introduce for anisotropic/high-strain ellipsoids.

        Returns:
            Ellipsoid: mean strain ellipsoid.
        """
        mean_log = np.mean([spla.logm(np.asarray(e)) for e in self], axis=0)
        return Ellipsoid(spla.expm(mean_log).real)


class OrientationTensor3Set(EllipsoidSet):
    """
    Class to store set of ``OrientationTensor3`` features.
    """

    __feature_class__ = OrientationTensor3

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, OrientationTensor3) for obj in data):
            raise TypeError("Data must be instances of OrientationTensor3")

    def __repr__(self):
        return f"M({len(self)}) {self.name}"


class Stress2Set(FeatureSet):
    """
    Class to store set of ``Stress2`` features.
    """

    __feature_class__ = Stress2

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, Stress2) for obj in data):
            raise TypeError("Data must be instances of Stress2")

    def __repr__(self):
        return f"Sig2({len(self)}) {self.name}"

    mean_stress = _per_item("mean_stress", "Return the array of mean stresses.")
    sigma1 = _per_item(
        "sigma1", "Return the array of the maximum principal stress (max compressive)."
    )
    sigma2 = _per_item("sigma2", "Return the array of the minimum principal stress.")

    @property
    def sigma1dir(self) -> Vector2Set:
        """Return Vector2Set of unit length vector in direction of maximum."""

        return Vector2Set([e.sigma1dir for e in self])

    @property
    def sigma2dir(self) -> Vector2Set:
        """Return Vector2Set of unit length vector in direction of minimum."""

        return Vector2Set([e.sigma2dir for e in self])

    I1 = _per_item("I1", "Return the array of first invariants.")
    I2 = _per_item("I2", "Return the array of second invariants.")
    I3 = _per_item("I3", "Return the array of third invariants.")


class Stress3Set(FeatureSet):
    """
    Class to store set of ``Stress3`` features.
    """

    __feature_class__ = Stress3

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, Stress3) for obj in data):
            raise TypeError("Data must be instances of Stress3")

    def __repr__(self):
        return f"Sig({len(self)}) {self.name}"

    mean_stress = _per_item("mean_stress", "Return the array of mean stresses.")
    sigma1 = _per_item(
        "sigma1", "Return the array of the maximum principal stress (max compressive)."
    )
    sigma2 = _per_item(
        "sigma2",
        "Return the array of the intermediate principal stress (max compressive).",
    )
    sigma3 = _per_item(
        "sigma3", "Return the array of the minimum principal stress (max tensile)."
    )

    @property
    def sigma1dir(self) -> Vector3Set:
        """Return Vector3Set of unit length vector in direction of maximum."""

        return Vector3Set([e.sigma1dir for e in self])

    @property
    def sigma2dir(self) -> Vector3Set:
        """Return Vector3Set of unit length vector in direction of intermediate."""

        return Vector3Set([e.sigma2dir for e in self])

    @property
    def sigma3dir(self) -> Vector3Set:
        """Return Vector3Set of unit length vector in direction of minimum."""

        return Vector3Set([e.sigma3dir for e in self])

    I1 = _per_item("I1", "Return the array of first invariants.")
    I2 = _per_item("I2", "Return the array of second invariants.")
    I3 = _per_item("I3", "Return the array of third invariants.")
    stress_ratio = _per_item(
        "stress_ratio", "Return the array of stress ratios R = (σ2 - σ3) / (σ1 - σ3)."
    )

    def mean_tensor(self, level=0.95, normalize=False, anisoft=False):
        """Return mean tensor and confidence ellipses of its principal axes.

        Uses the classic linear perturbation method of Jelinek (1978), which
        builds upon Hext (1963) matrix statistics. Each tensor is rotated to
        the principal frame of the mean tensor and the sample covariance of the
        first-order angular deviations of each principal axis towards the other
        two axes gives the confidence ellipse (Hotelling's T² with
        ``F(2, n - 2)``).

        Jelinek, V. (1978). Statistical processing of anisotropy of magnetic
            susceptibility measured on groups of specimens. Studia Geophysica
            et Geodaetica, 22, 50-62.
        Hext, G.R. (1963). The estimation of second-order tensors, with related
            tests and designs. Biometrika, 50, 353-373.

        Note:
            The covariance is the unbiased sample covariance and the region is
            the exact Hotelling T² region (in simulation its coverage matches
            the nominal level). Use ``anisoft=True`` to reproduce
            implementations (jelinekstat, TomoFab) that additionally scale the
            covariance by ``((n - 1) / n)²``.

        Args:
            level (float): confidence level. Default 0.95 for 95 %.
            normalize (bool): divide each tensor by its absolute mean value
                ``abs(trace) / 3`` before averaging, as usual for anisotropy
                of magnetic susceptibility. Default False. Fails for tensors
                with zero trace.
            anisoft (bool): multiply the tilts of the principal axes by an extra
                factor ``(n - 1) / n`` (the covariance by ``((n - 1) / n)²``),
                which shrinks the ellipses (by about 9-12 % in angle for n = 8) and
                lowers their coverage below the nominal level. Default False.

        Returns:
            dict: with keys ``mean`` (mean tensor), ``eigenvalues`` (of the
            mean tensor, descending; for stress sets the sign follows
            ``apsg_conf.stress_convention``, index-aligned with ``ellipses``),
            ``ellipses`` (tuple of three dicts, one
            for each principal axis of the mean tensor, 0 is the largest
            eigenvalue, with keys ``mu`` (principal axis), ``axes`` (tuple of
            the two ellipse axes, perpendicular to ``mu``), ``gamma`` (tuple
            of the corresponding semi-angles in degrees) and ``which``),
            ``intervals`` (``(3, 2)`` confidence intervals of the eigenvalues,
            Jelínek, 1978, eq. 35), ``anisotropy_test`` (``TestResult`` of
            Jelínek's T² test of isotropy, for more than 5 tensors), ``shape``
            (``jelinek_shape`` parameters of the mean, for positive definite
            ellipsoids), ``n`` (sample size), ``level``, ``normalize`` and
            ``anisoft``.

        Raises:
            ValueError: for less than 3 tensors, or when ``normalize`` is used
                on tensors with zero trace.
        """
        self._require(3, "mean_tensor")
        return _mean_tensor(self, level, normalize, anisoft)


class RotationSet(FeatureSet):
    """
    Class to store set of ``Rotation3`` features, e.g. directed orthogonal
    frames.
    """

    __feature_class__ = Rotation3

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, Rotation3) for obj in data):
            raise TypeError("Data must be instances of Rotation3")

    def __repr__(self):
        return f"R({len(self)}) {self.name}"

    def __array__(self, dtype=None, copy=None):
        return np.array([np.asarray(r) for r in self], dtype=dtype)

    def mean(self):
        """Mean rotation, the rotation closest to the mean rotation matrix.

        The polar (orthogonal) part of the mean matrix ``U diag(d) V'``, ``U
        V'`` with the proper singular value decomposition (Mardia & Jupp, 2000,
        eq. 13.2.16); the maximum-likelihood mode of the ``MatrixFisher``
        distribution.

        Returns:
            Rotation3: the mean rotation.
        """
        from apsg.feature._distributions._rotations import proper_svd

        self._require(1, "mean")
        U, _, V = proper_svd(np.asarray(self).mean(axis=0))
        return Rotation3(U @ V.T)

    def angle(self, other=None):
        """Rotation angles in degrees between the rotations and `other`.

        Args:
            other: ``Rotation3`` or ``RotationSet`` of the same length. Default
                None gives the angles of the rotations themselves.

        Returns:
            ndarray: angles in degrees.
        """
        X = np.asarray(self)
        if isinstance(other, RotationSet):
            self._check_same_length(other, "angle")
            X = np.einsum("nji,njk->nik", np.asarray(other), X)
        elif other is not None:
            X = np.einsum("ji,njk->nik", np.asarray(other, dtype=float), X)
        cos = (np.trace(X, axis1=1, axis2=2) - 1) / 2
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    @classmethod
    def random(cls, n=100, name="Default"):
        """Return ``RotationSet`` of uniformly distributed random rotations."""
        from scipy.spatial.transform import Rotation

        from apsg.feature._distributions._base import legacy_rng

        n = check_count(n)
        R = Rotation.random(n, random_state=legacy_rng()).as_matrix()
        return cls([Rotation3(r) for r in R], name=name)
