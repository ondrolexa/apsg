"""Grain models of magnetic fabrics: kinematic coefficients, demagnetising factors
and the susceptibility tensor of a population of grains.

Plain numpy functions of numbers and arrays; apsg objects only appear in the
return values of ``ams_from_orientation`` and ``ams_from_grains``.
"""

import numpy as np

from apsg.feature._tensor3 import Ellipsoid

_KINDS = ("prolate", "oblate", "cylinder")


def kinematic_K(I1, I2):
    """Kinematic coefficient ``K = 1 - I1 / I2`` of an axial grain.

    ``I1`` is the kinematic moment about the symmetry axis and ``I2`` the one
    about an axis perpendicular to it (Willis, 1977, eq. 8 and 11), both per
    unit density: ``I_i = int (p_j^2 + p_k^2) dV``.

    Args:
        I1: moment about the symmetry axis.
        I2: moment about a transverse axis.

    Returns:
        float: ``K`` in [-1, 1].
    """
    I1, I2 = float(I1), float(I2)
    if I1 <= 0 or I2 <= 0:
        raise ValueError("kinematic moments must be positive")
    K = 1 - I1 / I2
    if not -1 <= K <= 1:
        raise ValueError(f"the moments give K = {K}, outside [-1, 1]")
    return K


def grain_K(kind, ratio):
    """Kinematic coefficient ``K`` of a grain of given shape.

    ``K`` scales the rate of deformation in the pseudo-velocity-gradient of
    the grain (``VelocityGradient3.pseudo``): ``K = 1`` for a needle, ``-1``
    for a plate, 0 for a sphere.

    - ``"prolate"``: spheroid with axial ratio ``ratio = a / b >= 1``, Jeffery's
      (1922) ``(r^2 - 1) / (r^2 + 1)``.
    - ``"oblate"``: spheroid with ``ratio = b / c >= 1`` (diameter over
      thickness), ``-(r^2 - 1) / (r^2 + 1)``.
    - ``"cylinder"``: right circular cylinder with ``ratio = L / R`` (length
      over radius), ``K = 1 - (R^2 / 2) / (R^2 / 4 + L^2 / 12)``; ``L / R =
      2 sqrt(3)`` gives 0.6 and ``3 sqrt(3)`` gives 0.8 (Willis, 1977, p. 891).

    Args:
        kind: ``"prolate"``, ``"oblate"`` or ``"cylinder"``.
        ratio: axial ratio, ``>= 1`` for spheroids, ``> 0`` for a cylinder
            (``inf`` allowed).

    Returns:
        float: ``K``.

    References:
        Jeffery, G.B. (1922). The motion of ellipsoidal particles immersed in a
            viscous fluid. Proc. R. Soc. London A 102, 161-179.
        Willis, D.G. (1977). A kinematic model of preferred orientation. GSA
            Bulletin 88, 883-894.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    r = float(ratio)
    if np.isnan(r) or r <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if kind == "cylinder":
        if np.isinf(r):
            return 1.0
        return kinematic_K(1 / 2, 1 / 4 + r**2 / 12)
    if r < 1:
        raise ValueError(f"the axial ratio of a spheroid must be >= 1, got {ratio}")
    if np.isinf(r):
        k = 1.0
    else:
        k = (r**2 - 1) / (r**2 + 1)
    return k if kind == "prolate" else -k


def osborn_factors(ratio, kind="prolate"):
    """Demagnetising factors ``(N_a, N_b, N_c)`` of a spheroid (Osborn, 1945).

    ``a`` is the symmetry axis, so ``N_a`` is the factor along the long axis
    of a prolate and along the pole of an oblate spheroid. They sum to 1 and
    tend to 1/3 for a sphere.

    Args:
        ratio: axial ratio ``>= 1`` (long over short axis).
        kind: ``"prolate"`` (default) or ``"oblate"``.

    Returns:
        tuple of float: ``(N_a, N_b, N_c)`` with ``N_b = N_c``.

    References:
        Osborn, J.A. (1945). Demagnetizing factors of the general ellipsoid.
            Physical Review 67, 351-357.
    """
    if kind not in ("prolate", "oblate"):
        raise ValueError(f"kind must be 'prolate' or 'oblate', got {kind!r}")
    m = float(ratio)
    if not m >= 1 or np.isnan(m):
        raise ValueError(f"ratio must be >= 1, got {ratio}")
    if m < 1 + 1e-6:
        return (1 / 3, 1 / 3, 1 / 3)
    m21 = m * m - 1
    s = np.sqrt(m21)
    if kind == "prolate":
        al = np.log((m + s) / (m - s))
        Na = (m / (2 * s) * al - 1) / m21
        Nb = m / (2 * m21) * (m - al / (2 * s))
        return (Na, Nb, Nb)
    ar = np.arcsin(s / m)
    Na = 0.5 / m21 * (m * m / s * ar - 1)
    Nc = m * m / m21 * (1 - ar / s)
    return (Nc, Na, Na)


def grain_susceptibilities(k, ratio, kind="prolate"):
    """Principal susceptibilities of a shape-anisotropic grain.

    A spheroid of intrinsic susceptibility ``k`` (SI, dimensionless) has
    ``K_i = k / (1 + k N_i)`` along its axes (demagnetising factors ``N_i``,
    ``osborn_factors``).

    Args:
        k: intrinsic susceptibility, ``>= 0``.
        ratio: axial ratio ``>= 1``.
        kind: ``"prolate"`` (default) or ``"oblate"``.

    Returns:
        tuple of float: ``(K_axis, K_transverse)``, along the symmetry axis
        (long axis of a prolate grain, pole of an oblate one) and
        perpendicular to it.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    Na, Nb, _ = osborn_factors(ratio, kind)
    return (k / (1 + k * Na), k / (1 + k * Nb))


def grain_anisotropy(k, ratio, kind="prolate"):
    """Anisotropy degree ``P_g = K_max / K_min`` of a shape-anisotropic grain.

    For magnetite (``k = 16.6``) with axial ratio 1.2 this gives 1.2044
    (the SUSIE ``compPg`` value, Jezek and Hrouda, 2007).

    Args:
        k: intrinsic susceptibility.
        ratio: axial ratio ``>= 1``.
        kind: ``"prolate"`` (default) or ``"oblate"``.

    Returns:
        float: ``P_g >= 1``.
    """
    if k == 0:
        return 1.0
    Ka, Kb = grain_susceptibilities(k, ratio, kind)
    return max(Ka, Kb) / min(Ka, Kb)


def _orientation_matrix(orientation):
    """Orientation tensor (trace 1) as an array from a tensor, array, ``ACG`` or
    set of axes."""
    if hasattr(orientation, "ortensor"):
        orientation = orientation.ortensor()
    T = np.asarray(orientation, dtype=float)
    if T.shape != (3, 3) or not np.all(np.isfinite(T)):
        raise ValueError("orientation must be a finite 3x3 tensor")
    tr = np.trace(T)
    if tr <= 0:
        raise ValueError("orientation tensor must have a positive trace")
    return (T + T.T) / (2 * tr)


def ams_from_orientation(orientation, K, delta):
    """Susceptibility tensor of one population of uniaxial grains.

    ``k = K I + Delta T`` with ``T`` the orientation tensor (trace 1) of the
    grains' symmetry axes (long axes of prolate and poles of oblate grains),
    ``K`` the transverse susceptibility and ``Delta`` the grain anisotropy
    ``K_axis - K_transverse``: positive for prolate, negative for oblate
    grains (Jezek and Hrouda, 2002a, eq. 3-5).

    Args:
        orientation: orientation tensor, 3x3 array (normalised to trace 1),
            ``ACG`` distribution (exact ``E[u u']``) or a set of axes.
        K: susceptibility perpendicular to the grain axis.
        delta: ``K_axis - K_transverse``.

    Returns:
        Ellipsoid: the susceptibility tensor.

    References:
        Jezek, J., Hrouda, F. (2002). Software for modeling the magnetic
            fabric of rocks in a deformation. Computers & Geosciences 28,
            1061-1068, eq. 3-5.
    """
    T = _orientation_matrix(orientation)
    return Ellipsoid(K * np.eye(3) + delta * T)


def ams_from_grains(populations):
    """Susceptibility tensor of several populations of uniaxial grains.

    ``k = I sum(w_i K_i) + sum(w_i Delta_i T_i)`` with the weights ``w_i``
    normalised to sum 1 (Jezek and Hrouda, 2002a, eq. 6).

    Args:
        populations: iterable of ``(weight, orientation, K, delta)``, see
            ``ams_from_orientation``.

    Returns:
        Ellipsoid: the susceptibility tensor.
    """
    pops = list(populations)
    if not pops:
        raise ValueError("at least one population is required")
    w = np.array([p[0] for p in pops], dtype=float)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    w = w / w.sum()
    k = np.zeros((3, 3))
    for wi, (_, orientation, K, delta) in zip(w, pops):
        k += wi * np.asarray(ams_from_orientation(orientation, K, delta))
    return Ellipsoid(k)


def _skew(w):
    """Skew-symmetric matrices ``[w]x`` of vectors ``w`` of shape ``(..., 3)``."""
    z = np.zeros(w.shape[:-1])
    return np.stack(
        [
            np.stack([z, -w[..., 2], w[..., 1]], -1),
            np.stack([w[..., 2], z, -w[..., 0]], -1),
            np.stack([-w[..., 1], w[..., 0], z], -1),
        ],
        -2,
    )


def jeffery_rotations(L, semi_axes, R0, times, rtol=1e-9):
    """Rotation of rigid ellipsoids in a homogeneous viscous flow (Jeffery, 1922).

    The velocity gradient ``L`` (``v = L x``) is split into the rate of
    deformation ``E`` and the spin ``W`` with vorticity vector ``Omega``
    (``W_ij = -e_ijk Omega_k``). An ellipsoid with semi-axes ``a_1, a_2, a_3``
    turns about its axis ``i`` with the angular velocity

    ``omega_i = Omega_i + (a_j^2 - a_k^2) / (a_j^2 + a_k^2) E_jk``

    (``i, j, k`` cyclic, in the axes of the ellipsoid; Jeffery, 1922, eq. 37),
    so its axes ``p_i`` obey ``dp_i/dt = omega x p_i``. For a spheroid it
    reduces to the model of Willis (1977) with ``K = (a^2 - b^2) / (a^2 +
    b^2)``; a sphere turns with ``Omega``. Grains are rigid and do not
    interact.

    Args:
        L: 3x3 velocity gradient.
        semi_axes: three positive semi-axes of the ellipsoid (the axes ``i`` of
            the rotation matrices follow this order).
        R0: ``(n, 3, 3)`` initial rotation matrices of the grains, columns are
            the grain axes in the coordinates of the flow.
        times: increasing times, the first is the time of `R0`.
        rtol: relative tolerance of the integrator.

    Returns:
        ndarray: rotation matrices of shape ``(len(times), n, 3, 3)``.

    References:
        Jeffery, G.B. (1922). The motion of ellipsoidal particles immersed in a
            viscous fluid. Proc. R. Soc. London A 102, 161-179.
    """
    from scipy.integrate import solve_ivp

    L = np.asarray(L, dtype=float)
    a = np.asarray(semi_axes, dtype=float)
    R0 = np.asarray(R0, dtype=float)
    t = np.asarray(times, dtype=float)
    if L.shape != (3, 3) or not np.all(np.isfinite(L)):
        raise ValueError("L must be a finite 3x3 matrix")
    if a.shape != (3,) or np.any(a <= 0):
        raise ValueError("semi_axes must be three positive numbers")
    if R0.ndim == 2:
        R0 = R0[None]
    if R0.ndim != 3 or R0.shape[1:] != (3, 3):
        raise ValueError("R0 must have shape (n, 3, 3)")
    if t.ndim != 1 or len(t) < 1 or np.any(np.diff(t) <= 0):
        raise ValueError("times must be increasing")
    n = len(R0)
    E = (L + L.T) / 2
    W = (L - L.T) / 2
    Om = np.array([W[2, 1], W[0, 2], W[1, 0]])
    a2 = a**2
    c = np.array(
        [
            (a2[1] - a2[2]) / (a2[1] + a2[2]),
            (a2[2] - a2[0]) / (a2[2] + a2[0]),
            (a2[0] - a2[1]) / (a2[0] + a2[1]),
        ]
    )
    pairs = ((1, 2), (2, 0), (0, 1))

    def rhs(_, y):
        R = y.reshape(n, 3, 3)
        Eb = np.einsum("nki,kl,nlj->nij", R, E, R)
        Ob = np.einsum("nki,k->ni", R, Om)
        wb = Ob + c * np.stack([Eb[:, j, k] for j, k in pairs], axis=1)
        wl = np.einsum("nij,nj->ni", R, wb)
        return (_skew(wl) @ R).ravel()

    if len(t) == 1:
        return R0[None].copy()
    sol = solve_ivp(
        rhs,
        (t[0], t[-1]),
        R0.ravel(),
        t_eval=t,
        method="DOP853",
        rtol=rtol,
        atol=rtol * 1e-2,
    )
    if not sol.success:  # pragma: no cover
        raise RuntimeError(sol.message)
    return sol.y.T.reshape(len(t), n, 3, 3)


def ams_from_rotations(R, K):
    """Mean susceptibility tensor of grains with orientations `R`.

    ``k = mean(R diag(K) R')`` for grains whose axes (columns of ``R``) have the
    principal susceptibilities ``K = (K_1, K_2, K_3)``.

    Args:
        R: ``(n, 3, 3)`` rotation matrices (or one 3x3).
        K: three principal susceptibilities of a grain.

    Returns:
        Ellipsoid: the mean susceptibility tensor.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim == 2:
        R = R[None]
    K = np.asarray(K, dtype=float)
    if K.shape != (3,):
        raise ValueError("K must be three principal susceptibilities")
    return Ellipsoid(np.einsum("nij,j,nkj->ik", R, K, R) / len(R))


def orientation_tensor_from_ams(tensor, Pc, kind):
    """Orientation tensor of uniaxial grains from the AMS ellipsoid.

    Inversion of Hrouda and Schulmann (1990) for a rock whose susceptibility is
    carried by grains of one kind, with the anisotropy degree ``Pc`` of a
    grain: ``k_i = Kbar (A O_i + B)`` for the principal values ``O_i`` of the
    orientation tensor (trace 1), with ``A = 3 (1 - Pc) / (2 Pc + 1)``, ``B =
    3 Pc / (2 Pc + 1)`` for oblate and ``A = 3 (Pc - 1) / (Pc + 2)``, ``B =
    3 / (Pc + 2)`` for prolate grains (eq. 6, 7, 10). It is solved directly, so
    the misprinted ``O_2`` and ``o_12`` of the paper (eq. 16) do not enter. The
    eigenvectors are those of the tensor; for oblate grains (poles) the
    orientation tensor is largest where the susceptibility is smallest.

    Note:
        The model applies to minerals with a known uniaxial anisotropy, such as
        biotite or muscovite; not to magnetite, whose grains are triaxial and
        have varying ``Pc``. Magnitudes of such fabrics are lower than
        directly measured ones (Hrouda and Schulmann, 1990).

    Args:
        tensor: symmetric 3x3 susceptibility tensor (positive definite).
        Pc: anisotropy degree of the grains, ``> 1``.
        kind: ``"prolate"`` or ``"oblate"``.

    Returns:
        OrientationTensor3: the orientation tensor of the grain axes.

    Raises:
        ValueError: if the tensor is not compatible with such grains (a
            negative eigenvalue of the orientation tensor).

    References:
        Hrouda, F., Schulmann, K. (1990). Conversion of the magnetic
            susceptibility tensor into the orientation tensor in some rocks.
            Physics of the Earth and Planetary Interiors 63, 71-77.
    """
    from apsg.feature._tensor3 import OrientationTensor3

    if kind not in ("prolate", "oblate"):
        raise ValueError(f"kind must be 'prolate' or 'oblate', got {kind!r}")
    if not Pc > 1:
        raise ValueError(f"Pc must be greater than 1, got {Pc}")
    T = np.asarray(tensor, dtype=float)
    if T.shape != (3, 3) or not np.all(np.isfinite(T)):
        raise ValueError("tensor must be a finite 3x3 matrix")
    k, V = np.linalg.eigh((T + T.T) / 2)
    if k[0] <= 0:
        raise ValueError("the susceptibility tensor must be positive definite")
    if kind == "oblate":
        A, B = 3 * (1 - Pc) / (2 * Pc + 1), 3 * Pc / (2 * Pc + 1)
    else:
        A, B = 3 * (Pc - 1) / (Pc + 2), 3 / (Pc + 2)
    s = k.sum() / (A + 3 * B)
    O = (k / s - B) / A
    if O.min() < -1e-9:
        raise ValueError(
            f"the tensor is not compatible with {kind} grains of Pc = {Pc}: "
            f"the orientation tensor would have eigenvalues {np.round(O, 4)}"
        )
    O = np.clip(O, 0, None)
    return OrientationTensor3((V * O) @ V.T / O.sum())
