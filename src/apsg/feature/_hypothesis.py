"""
Hypothesis tests of orientation data.

The tests are plain numpy functions of unit vector arrays (or of their summary
statistics) returning a ``TestResult``; the feature sets and distributions wrap
them. Abbreviations of the references:

- MJ: Mardia, K.V. & Jupp, P.E. (2000). Directional Statistics. Wiley.
- FLE: Fisher, N.I., Lewis, T. & Embleton, B.J.J. (1987). Statistical Analysis
  of Spherical Data. Cambridge University Press.
- F93: Fisher, N.I. (1993). Statistical Analysis of Circular Data. Cambridge
  University Press.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# keys of the dicts returned by uniformity_test() before 2.1, mapped to
# TestResult attributes
_LEGACY_KEYS = {
    "statistic": lambda r: r.statistic,
    "p_value": lambda r: r.p_value,
    "critical_value": lambda r: r.critical,
    "uniform": lambda r: not r.reject(r.level),
}


@dataclass(frozen=True)
class TestResult:
    """Result of a hypothesis test.

    Attributes:
        name: name of the test.
        statistic: value of the test statistic.
        p_value: p-value, None when the test only has critical values.
        df: degrees of freedom of the reference distribution, if any.
        critical: critical value of the statistic at `level`, if known.
        level: confidence level the critical value refers to.
        source: reference of the test.
        extra: further statistics, e.g. the resultant length ``R``.
    """

    __test__ = False  # not a pytest test class

    name: str
    statistic: float
    p_value: float | None = None
    df: object = None
    critical: float | None = None
    level: float = 0.95
    source: str = ""
    extra: dict = field(default_factory=dict)

    def reject(self, level=None):
        """Return True if the null hypothesis is rejected at confidence `level`.

        Args:
            level: confidence level. Default None uses the level of the test.
        """
        level = self.level if level is None else level
        if self.p_value is not None:
            return bool(self.p_value < 1 - level)
        if level != self.level:
            raise ValueError(
                f"{self.name} has a critical value for level {self.level} only"
            )
        return bool(self.statistic > self.critical)

    def as_dict(self):
        """Return the result as a dict."""
        return {
            "name": self.name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "df": self.df,
            "critical": self.critical,
            "level": self.level,
            "reject": self.reject(),
        } | self.extra

    def __getitem__(self, key):
        # dict access of the uniformity_test() results before 2.1
        if key in _LEGACY_KEYS:
            value = _LEGACY_KEYS[key](self)
        elif key in self.extra:
            value = self.extra[key]
        else:
            raise KeyError(key)
        warnings.warn(
            "uniformity_test() returns a TestResult since apsg 2.1; use its "
            f"attributes (statistic, p_value, reject(), extra) instead of [{key!r}]",
            FutureWarning,
            stacklevel=2,
        )
        return value

    def __repr__(self):
        p = "" if self.p_value is None else f", p_value={self.p_value:.4g}"
        return f"TestResult({self.name}: statistic={self.statistic:.4g}{p})"


def rayleigh_test(X, level=0.95):
    """Rayleigh test of uniformity against a unimodal alternative.

    The statistic is ``S = p n Rbar^2`` for `n` unit vectors in `p` dimensions,
    asymptotically chi-square with `p` degrees of freedom; the p-value is that of
    the modified statistic ``S* = (1 - 1/(2n)) S + S^2 / (2n(p + 2))``, which
    has error O(1/n^2) (MJ eq. 10.4.4-10.4.6, eq. 6.3.3 for the circle; FLE
    sec. 5.3.1(i)). The test is not consistent against axial alternatives.

    Args:
        X: ``(n, p)`` array of unit vectors.
        level: confidence level. Default 0.95.

    Returns:
        TestResult: with ``extra`` keys ``R`` (resultant length), ``Rbar`` and
        ``modified`` (``S*``).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    R = float(np.linalg.norm(X.sum(axis=0)))
    S = p * R**2 / n
    S_mod = (1 - 1 / (2 * n)) * S + S**2 / (2 * n * (p + 2))
    return TestResult(
        name="Rayleigh",
        statistic=float(S),
        p_value=float(stats.chi2.sf(S_mod, df=p)),
        df=p,
        critical=float(stats.chi2.ppf(level, df=p)),
        level=level,
        source="Mardia & Jupp (2000), eq. 10.4.4-10.4.6",
        extra={"R": R, "Rbar": R / n, "modified": float(S_mod)},
    )


def bingham_test(T, n, level=0.95):
    """Bingham test of uniformity of axes on the sphere.

    The statistic is ``S = 15 n / 2 (tr(T^2) - 1/3)`` with the orientation
    tensor ``T`` (scaled to trace 1), asymptotically chi-square with 5 degrees
    of freedom (MJ eq. 10.7.1-10.7.2), which gives the p-value. The modified
    statistic ``S*`` of MJ eq. 10.7.3, a polynomial correction meant for ``S``
    near the critical values, is given in ``extra``. ``S`` equals Bingham's (1974, eq. 5.9)
    ``X_U^2`` and ``n I`` with Lisle's (1985) intensity ``I = 15/2 sum((tau_i -
    1/3)^2)``. Large ``S`` indicates a preferred orientation (cluster or
    girdle); a very small ``S`` (lower tail) indicates data more regular than
    random.

    Args:
        T: 3x3 orientation tensor ``mean(x x')``.
        n: number of axes.
        level: confidence level. Default 0.95.

    Returns:
        TestResult: with ``extra`` key ``modified`` (``S*``).

    References:
        Bingham, C. (1974). An antipodally symmetric distribution on the sphere.
            The Annals of Statistics 2, 1201-1225.
        Lisle, R.J. (1985). The use of the orientation tensor for the
            description and statistical testing of fabrics. Journal of
            Structural Geology 7, 115-117.
    """
    T = np.asarray(T, dtype=float)
    S = 7.5 * n * (np.sum(T**2) - 1 / 3)
    # MJ eq. 10.7.3 for p = 3
    b0, b1, b2 = 31 / 42, -41 / 294, 20 / 5292
    S_mod = S * (1 - (b0 + b1 * S + b2 * S**2) / n)
    return TestResult(
        name="Bingham",
        statistic=float(S),
        p_value=float(stats.chi2.sf(S, df=5)),
        df=5,
        critical=float(stats.chi2.ppf(level, df=5)),
        level=level,
        source="Mardia & Jupp (2000), eq. 10.7.1-10.7.3",
        extra={"modified": float(S_mod)},
    )


# circular uniformity -----------------------------------------------------------


def _uniform_scores(theta):
    """Sorted values ``theta / 2pi`` in [0, 1) of angles `theta` (radians)."""
    return np.sort(np.mod(np.asarray(theta, dtype=float), 2 * np.pi) / (2 * np.pi))


def kuiper_test(theta, level=0.95):
    """Kuiper's test of uniformity on the circle.

    ``V = (D+ + D-)(sqrt(n) + 0.155 + 0.24 / sqrt(n))`` with the Kolmogorov
    statistics ``D+``, ``D-`` of the angles against the uniform distribution
    (F93 sec. 4.3, Table A5, after Stephens, 1970). The modified ``V`` follows
    the asymptotic Kuiper distribution, which gives the p-value; the 5 % point
    is 1.747. The test is consistent against all alternatives, axial ones
    included.

    Args:
        theta: angles in radians.
        level: confidence level. Default 0.95.

    Returns:
        TestResult: with ``extra`` keys ``D+`` and ``D-``.
    """
    u = _uniform_scores(theta)
    n = len(u)
    i = np.arange(1, n + 1)
    d_plus = float(np.max(i / n - u))
    d_minus = float(np.max(u - (i - 1) / n))
    V = (d_plus + d_minus) * (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n))
    return TestResult(
        name="Kuiper",
        statistic=float(V),
        p_value=_kuiper_sf(V),
        critical=_kuiper_isf(1 - level),
        level=level,
        source="Fisher (1993), sec. 4.3, Table A5",
        extra={"D+": d_plus, "D-": d_minus},
    )


def _kuiper_sf(v):
    """Survival function of the asymptotic Kuiper distribution."""
    if v < 0.4:
        return 1.0
    j = np.arange(1, 101)
    p = 2 * np.sum((4 * j**2 * v**2 - 1) * np.exp(-2 * j**2 * v**2))
    return float(np.clip(p, 0, 1))


def _kuiper_isf(alpha):
    from scipy.optimize import brentq

    return float(brentq(lambda v: _kuiper_sf(v) - alpha, 0.5, 5))


def watson_u2_test(theta, level=0.95):
    """Watson's U² test of uniformity on the circle.

    ``U² = sum((u_i - (2i - 1) / 2n)^2) - n (mean(u) - 1/2)^2 + 1 / 12n`` with
    the sorted ``u_i = theta_i / 2pi`` (MJ sec. 6.3.5); the modification ``U* =
    (U² - 0.1/n + 0.1/n^2)(1 + 0.8/n)`` follows the asymptotic distribution
    ``P(U² > u) = 2 sum((-1)^(j-1) exp(-2 j^2 pi^2 u))`` (Stephens, 1970).

    Args:
        theta: angles in radians.
        level: confidence level. Default 0.95.

    Returns:
        TestResult: the statistic is the modified ``U*``, ``extra`` holds
        ``U2``.
    """
    u = _uniform_scores(theta)
    n = len(u)
    i = np.arange(1, n + 1)
    u2 = (
        np.sum((u - (2 * i - 1) / (2 * n)) ** 2)
        - n * (u.mean() - 0.5) ** 2
        + 1 / (12 * n)
    )
    u_mod = (u2 - 0.1 / n + 0.1 / n**2) * (1 + 0.8 / n)
    return TestResult(
        name="Watson U2",
        statistic=float(u_mod),
        p_value=_watson_u2_sf(u_mod),
        critical=_watson_u2_isf(1 - level),
        level=level,
        source="Mardia & Jupp (2000), sec. 6.3.5; Stephens (1970)",
        extra={"U2": float(u2)},
    )


def _watson_u2_sf(u):
    if u <= 0.02:
        return 1.0
    j = np.arange(1, 101)
    p = 2 * np.sum((-1.0) ** (j - 1) * np.exp(-2 * j**2 * np.pi**2 * u))
    return float(np.clip(p, 0, 1))


def _watson_u2_isf(alpha):
    from scipy.optimize import brentq

    return float(brentq(lambda u: _watson_u2_sf(u) - alpha, 0.03, 2))


def hodges_ajne_test(theta, level=0.95):
    """Hodges-Ajne test of uniformity on the circle.

    The statistic ``m`` is the smallest number of angles in a half-circle. Its
    exact distribution is ``P(m <= t) = 2^(1 - n) (n - 2t) C(n, t)`` for ``t <
    n / 3`` (MJ sec. 6.3.4), which gives the p-value (1 above ``n / 3``). Small
    ``m`` indicates a preferred direction.

    Args:
        theta: angles in radians.
        level: confidence level. Default 0.95.

    Returns:
        TestResult: the statistic is ``m``.
    """
    from scipy.special import comb

    t = np.sort(np.mod(np.asarray(theta, dtype=float), 2 * np.pi))
    n = len(t)
    # number of angles in the half-circles starting at each angle
    ends = np.searchsorted(np.concatenate([t, t + 2 * np.pi]), t + np.pi)
    counts = ends - np.arange(n)
    m = int(min(np.min(counts), np.min(n - counts)))
    p = 1.0
    if m < n / 3:
        p = min(1.0, float(2.0 ** (1 - n) * (n - 2 * m) * comb(n, m, exact=True)))
    return TestResult(
        name="Hodges-Ajne",
        statistic=float(m),
        p_value=p,
        level=level,
        source="Mardia & Jupp (2000), sec. 6.3.4",
    )


def rao_spacing_test(theta, level=0.95, n_sim=9999, rng=None):
    """Rao's spacing test of uniformity on the circle.

    ``L = 1/2 sum(|T_i - 360 / n|)`` in degrees, with the spacings ``T_i``
    between consecutive angles (MJ eq. 6.3.45). The p-value is the proportion
    of ``n_sim`` uniform samples with a larger ``L``.

    Args:
        theta: angles in radians.
        level: confidence level. Default 0.95.
        n_sim: number of simulated uniform samples. Default 9999.
        rng: None, an integer seed or a ``numpy.random.Generator``.

    Returns:
        TestResult: the statistic is ``L`` in degrees.
    """
    from apsg.feature._distributions._base import as_rng

    def spacing(t):
        t = np.sort(np.mod(t, 2 * np.pi), axis=-1)
        gaps = np.diff(t, axis=-1, append=t[..., :1] + 2 * np.pi)
        return 0.5 * np.sum(np.abs(np.degrees(gaps) - 360 / t.shape[-1]), axis=-1)

    theta = np.asarray(theta, dtype=float)
    L = float(spacing(theta))
    sims = spacing(as_rng(rng).uniform(0, 2 * np.pi, (n_sim, len(theta))))
    p = (np.sum(sims >= L) + 1) / (n_sim + 1)
    return TestResult(
        name="Rao spacing",
        statistic=L,
        p_value=float(p),
        critical=float(np.quantile(sims, level)),
        level=level,
        source="Mardia & Jupp (2000), eq. 6.3.45",
    )


# goodness of fit and model comparison -------------------------------------------

# upper 10, 5 and 1 % points of M_E and M_U (FLE, with eq. 5.15-5.16)
_FISHER_GOF_POINTS = {
    0.90: (0.990, 1.138),
    0.95: (1.094, 1.207),
    0.99: (1.308, 1.347),
}


def fisher_goodness_of_fit(X, level=0.95):
    """Goodness-of-fit tests of the Fisher distribution.

    The data are rotated to their mean direction. Under the Fisher model the
    colatitudes ``theta'`` give ``1 - cos(theta')`` approximately exponential
    with rate ``k = (n - 1) / sum(1 - cos(theta'))`` and the longitudes are
    uniform. The colatitude test ``M_E = (D_n - 0.2/n)(sqrt(n) + 0.26 +
    0.5/sqrt(n))`` uses the Kolmogorov statistic ``D_n`` against the
    exponential distribution, the longitude test ``M_U = V_n(sqrt(n) - 0.467 +
    1.623/sqrt(n))`` the Kuiper statistic ``V_n`` against the uniform
    distribution (FLE eq. 5.15-5.16). Critical values are known for the
    levels 0.90, 0.95 and 0.99.

    Note:
        The constants are FLE's, which are the original ones of Stephens
        (1974) and Fisher & Best (1984). Mardia & Jupp (2000, eq. 12.3.1,
        12.3.4) misquote them.

    Args:
        X: ``(n, 3)`` array of unit vectors.
        level: 0.90, 0.95 or 0.99. Default 0.95.

    Returns:
        dict: ``colatitude`` and ``longitude`` ``TestResult``; the Fisher
        model is rejected when either test rejects.
    """
    if level not in _FISHER_GOF_POINTS:
        raise ValueError("level must be 0.90, 0.95 or 0.99")
    X = np.asarray(X, dtype=float)
    n = len(X)
    mean = X.sum(axis=0)
    mean = mean / np.linalg.norm(mean)
    ref = np.eye(3)[np.argmin(np.abs(mean))]
    e1 = np.cross(ref, mean)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(mean, e1)
    c = np.clip(X @ mean, -1, 1)
    x = np.sort(1 - c)
    k = (n - 1) / np.sum(x)
    i = np.arange(1, n + 1)
    F = 1 - np.exp(-k * x)
    D = max(np.max(i / n - F), np.max(F - (i - 1) / n))
    m_e = (D - 0.2 / n) * (np.sqrt(n) + 0.26 + 0.5 / np.sqrt(n))
    phi = np.arctan2(X @ e2, X @ e1)
    u = _uniform_scores(phi)
    V = np.max(i / n - u) + np.max(u - (i - 1) / n)
    m_u = V * (np.sqrt(n) - 0.467 + 1.623 / np.sqrt(n))
    crit_e, crit_u = _FISHER_GOF_POINTS[level]
    source = "Fisher, Lewis & Embleton (1987), eq. 5.15-5.16"
    return {
        "colatitude": TestResult(
            "Fisher colatitude M_E",
            float(m_e),
            critical=crit_e,
            level=level,
            source=source,
            extra={"D": float(D), "k": float(k)},
        ),
        "longitude": TestResult(
            "Fisher longitude M_U",
            float(m_u),
            critical=crit_u,
            level=level,
            source=source,
            extra={"V": float(V)},
        ),
    }


def fisher_kent_test(X, level=0.95):
    """Score test of the Fisher against the Kent distribution.

    ``T = n (kappa/2)^2 I_1/2(kappa) / I_5/2(kappa) r2^2`` is asymptotically
    chi-square with 2 degrees of freedom (Kent, 1982, eq. 5.5), with the
    maximum-likelihood ``kappa`` of the Fisher distribution (Kent's Theorem
    5.1) and ``r2`` the difference of the two eigenvalues of the second
    moments of the data in the tangent plane of the mean direction. Large ``T``
    indicates elliptical (Kent) rather than circular (Fisher) contours.

    Note:
        FLE (sec. 5.4.6) computes the statistic with the concentration of the
        fitted Kent distribution, which gives a larger value; this follows
        Kent's theorem.

    Args:
        X: ``(n, 3)`` array of unit vectors.
        level: confidence level. Default 0.95.

    Returns:
        TestResult: with ``extra`` keys ``kappa`` and ``r2``.

    References:
        Kent, J.T. (1982). The Fisher-Bingham distribution on the sphere.
            Journal of the Royal Statistical Society B 44, 71-80.
    """
    from scipy.special import ive

    from apsg.feature._distributions._spherical import _fisher_kappa_mle, _tangent_basis

    X = np.asarray(X, dtype=float)
    n = len(X)
    resultant = X.sum(axis=0)
    Rbar = np.linalg.norm(resultant) / n
    e1, e2 = _tangent_basis(resultant / np.linalg.norm(resultant))
    P = X @ np.column_stack([e1, e2])
    lam = np.linalg.eigvalsh(P.T @ P / n)
    r2 = float(lam[1] - lam[0])
    kappa = _fisher_kappa_mle(Rbar)
    T = n * (kappa / 2) ** 2 * ive(0.5, kappa) / ive(2.5, kappa) * r2**2
    return TestResult(
        name="Fisher vs Kent",
        statistic=float(T),
        p_value=float(stats.chi2.sf(T, 2)),
        df=2,
        critical=float(stats.chi2.ppf(level, 2)),
        level=level,
        source="Kent (1982), eq. 5.5",
        extra={"kappa": float(kappa), "r2": r2},
    )


def watson_symmetry_test(eigenvalues, kappas, n, level=0.95):
    """Tests of rotational symmetry of a Bingham distribution.

    Whether a Bingham distribution is a Watson distribution: ``w_b = n
    sum((k_i - mean(k)) (tau_i - mean(tau)))`` over the two smallest
    eigenvalues tests the bipolar form, ``w_g`` over the two largest the girdle
    form, each asymptotically chi-square with 2 degrees of freedom (MJ eq.
    10.7.8-10.7.18). ``k_i`` are the maximum-likelihood Bingham concentrations
    ``(0, kappa_major, kappa_minor)`` and ``tau_i`` the eigenvalues of the
    orientation tensor, both in descending order. Mardia & Jupp's calcite
    example gives ``w_g = 1.33`` (girdle accepted) and ``w_b = 9.97``.

    Args:
        eigenvalues: descending eigenvalues of the orientation tensor.
        kappas: Bingham concentrations ``(0, kappa_major, kappa_minor)``.
        n: number of axes.
        level: confidence level. Default 0.95.

    Returns:
        dict: ``bipolar`` and ``girdle`` ``TestResult``.
    """
    t = np.asarray(eigenvalues, dtype=float)
    k = np.asarray(kappas, dtype=float)
    out = {}
    for name, idx in (("bipolar", [1, 2]), ("girdle", [0, 1])):
        w = n * np.sum((k[idx] - k[idx].mean()) * (t[idx] - t[idx].mean()))
        out[name] = TestResult(
            name=f"Watson {name} symmetry",
            statistic=float(w),
            p_value=float(stats.chi2.sf(w, 2)),
            df=2,
            critical=float(stats.chi2.ppf(level, 2)),
            level=level,
            source="Mardia & Jupp (2000), eq. 10.7.8-10.7.18",
        )
    return out


def likelihood_ratio_test(null, alternative, data, level=0.95):
    """Likelihood ratio test of nested distributions fitted to the same data.

    ``2 (loglik(alternative) - loglik(null))`` is asymptotically chi-square
    with degrees of freedom equal to the difference of the numbers of
    parameters (Wilks' theorem). The models must be nested (e.g. Fisher in
    Kent, Watson in Bingham) and of the same symmetry; the asymptotics fail
    when the null lies on the boundary of the alternative.

    Args:
        null: fitted ``OrientationDistribution`` with fewer parameters.
        alternative: fitted ``OrientationDistribution`` containing `null`.
        data: the data both were fitted to.
        level: confidence level. Default 0.95.

    Returns:
        TestResult: the statistic is twice the log-likelihood ratio.
    """
    df = alternative.n_params - null.n_params
    if df <= 0:
        raise ValueError("The alternative must have more parameters than the null")
    if null.axial != alternative.axial:
        raise ValueError("Models of axial and vectorial data are not nested")
    lr = 2 * (alternative.loglik(data) - null.loglik(data))
    return TestResult(
        name=f"LR {type(null).__name__} vs {type(alternative).__name__}",
        statistic=float(lr),
        p_value=float(stats.chi2.sf(max(lr, 0.0), df)),
        df=df,
        critical=float(stats.chi2.ppf(level, df)),
        level=level,
        source="Wilks' theorem",
    )


# comparing samples ----------------------------------------------------------------


def _resultants(samples):
    """Sizes, resultant lengths and resultant vectors of unit vector arrays."""
    samples = [np.asarray(x, dtype=float) for x in samples]
    if len(samples) < 2:
        raise ValueError("At least two samples are needed")
    dims = {x.shape[1] for x in samples}
    if len(dims) != 1:
        raise ValueError("The samples must have the same dimension")
    n = np.array([len(x) for x in samples])
    S = np.array([x.sum(axis=0) for x in samples])
    return n, np.linalg.norm(S, axis=1), S, dims.pop()


def common_mean_test(samples, method=None, level=0.95):
    """Test whether samples of directions share the mean direction.

    Methods on the sphere:

    - "watson" (default): ``F = (n - q) (sum(R_i) - R) / ((q - 1) (n -
      sum(R_i)))``, ``F(2(q - 1), 2(n - q))`` for samples of a Fisher
      distribution with a common concentration (Watson, 1956, eq. 2.32 for two
      samples; MJ eq. 10.5.1, 10.6.4), adequate for ``Rbar >= 0.67``.
    - "mcfadden_lowes": the conditional test of McFadden & Lowes (1981, eq.
      25, 41), ``g = (n - q) / (q - 1) (sum(R_i) - R^2 / sum(R_i)) / (2 (n -
      sum(R_i)))``, ``F(2(q - 1), 2(n - q))``.
    - "watson_debiche": for unequal concentrations, ``V = 2 (sum(k_i R_i) -
      |sum(k_i R_i mu_i)|)``, chi-square with ``2(q - 1)`` degrees of freedom,
      with Fisher's precisions ``k_i`` (Watson & Debiche, 1992, eq. 22, 25-26
      with their dropped factor 2 restored; equal to FLE's ``g_r``, eq.
      7.33-7.36, for ``n_i >= 25``).

    Methods on the circle:

    - "watson_williams" (default): the same F statistic with ``F(q - 1, n -
      q)``, multiplied by ``1 + 3 / (8 kappa)`` with the maximum-likelihood
      ``kappa`` of the pooled data (MJ eq. 7.3.4, 7.4.4-7.4.6), adequate for
      ``Rbar >= 0.45``.
    - "uniform_scores": the distribution-free test of Mardia, Watson & Wheeler
      on the uniform scores ``2 pi r_i / n`` of the ranks in the pooled sample,
      ``W = 2 sum(C_i^2 + S_i^2) / n_i`` with ``2(q - 1)`` degrees of freedom,
      ``R* = 2 (n - 1) R_1^2 / (n_1 n_2)`` for two samples (MJ eq.
      8.3.3-8.3.4, 8.4.3). It tests the equality of the distributions.
    - "watson_u2": Watson's two-sample U² (MJ eq. 8.3.7), a distribution-free
      test of equal distributions with the asymptotic law of ``U²``.

    Args:
        samples: sequence of ``(n_i, 2)`` or ``(n_i, 3)`` unit vector arrays.
        method: name of the test. Default None ("watson" on the sphere,
            "watson_williams" on the circle).
        level: confidence level. Default 0.95.

    Returns:
        TestResult: ``reject()`` is True when the mean directions differ.
    """
    n_i, R_i, S_i, p = _resultants(samples)
    n, q = n_i.sum(), len(n_i)
    sumR, R = R_i.sum(), np.linalg.norm(S_i.sum(axis=0))
    method = method or ("watson" if p == 3 else "watson_williams")
    sphere = {"watson", "mcfadden_lowes", "watson_debiche"}
    circle = {"watson_williams", "uniform_scores", "watson_u2"}
    if method not in (sphere if p == 3 else circle):
        expected = ", ".join(sorted(sphere if p == 3 else circle))
        raise ValueError(
            f"Unknown method {method!r} for {p}D data, expected {expected}"
        )
    if method in ("watson", "watson_williams"):
        F = (n - q) * (sumR - R) / ((q - 1) * (n - sumR))
        if p == 3:
            df = (2 * (q - 1), 2 * (n - q))
            source = "Watson (1956), eq. 2.32; Mardia & Jupp (2000), eq. 10.6.4"
        else:
            from apsg.feature._distributions._circular import _vonmises_kappa_mle

            kappa = _vonmises_kappa_mle(sumR / n)
            F *= 1 + 3 / (8 * kappa) if kappa > 0 else np.inf
            df = (q - 1, n - q)
            source = "Mardia & Jupp (2000), eq. 7.4.4-7.4.6"
        return _f_result(f"{method} common mean", F, df, level, source)
    if method == "mcfadden_lowes":
        g = (n - q) / (q - 1) * (sumR - R**2 / sumR) / (2 * (n - sumR))
        source = "McFadden & Lowes (1981), eq. 25, 41"
        return _f_result(
            "McFadden-Lowes common mean", g, (2 * (q - 1), 2 * (n - q)), level, source
        )
    if method == "watson_debiche":
        k = (n_i - 1) / (n_i - R_i)
        V = 2 * (np.sum(k * R_i) - np.linalg.norm((k[:, None] * S_i).sum(axis=0)))
        return _chi2_result(
            "Watson-Debiche common mean",
            V,
            2 * (q - 1),
            level,
            "Watson & Debiche (1992), eq. 22, 25-26",
        )
    theta = [np.arctan2(x[:, 1], x[:, 0]) for x in map(np.asarray, samples)]
    if method == "uniform_scores":
        return _uniform_scores_test(theta, level)
    return _watson_u2_two_sample(theta, level)


def _f_result(name, F, df, level, source):
    return TestResult(
        name=name,
        statistic=float(F),
        p_value=float(stats.f.sf(F, *df)),
        df=tuple(int(d) for d in df),
        critical=float(stats.f.ppf(level, *df)),
        level=level,
        source=source,
    )


def _chi2_result(name, X, df, level, source):
    return TestResult(
        name=name,
        statistic=float(X),
        p_value=float(stats.chi2.sf(X, df)),
        df=int(df),
        critical=float(stats.chi2.ppf(level, df)),
        level=level,
        source=source,
    )


def _uniform_scores_test(theta, level):
    pooled = np.mod(np.concatenate(theta), 2 * np.pi)
    n = len(pooled)
    groups = np.repeat(np.arange(len(theta)), [len(t) for t in theta])
    # ties within a sample get consecutive ranks, ties between samples the
    # average rank
    ranks = stats.rankdata(pooled, method="ordinal").astype(float)
    for value in np.unique(pooled):
        tied = pooled == value
        if len(np.unique(groups[tied])) > 1:
            ranks[tied] = ranks[tied].mean()
    beta = 2 * np.pi * ranks / n
    C = np.bincount(groups, np.cos(beta))
    S = np.bincount(groups, np.sin(beta))
    n_i = np.bincount(groups)
    if len(theta) == 2:
        W = 2 * (n - 1) * (C[0] ** 2 + S[0] ** 2) / (n_i[0] * n_i[1])
    else:
        W = 2 * np.sum((C**2 + S**2) / n_i)
    return _chi2_result(
        "uniform scores",
        W,
        2 * (len(theta) - 1),
        level,
        "Mardia & Jupp (2000), eq. 8.3.3-8.3.4, 8.4.3",
    )


def _watson_u2_two_sample(theta, level):
    if len(theta) != 2:
        raise ValueError("Watson's two-sample U2 compares two samples")
    n1, n2 = len(theta[0]), len(theta[1])
    n = n1 + n2
    pooled = np.mod(np.concatenate(theta), 2 * np.pi)
    first = np.concatenate([np.ones(n1), np.zeros(n2)])[
        np.argsort(pooled, kind="stable")
    ]
    a = np.cumsum(first)
    d = (np.arange(1, n + 1) - a) / n2 - a / n1
    u2 = n1 * n2 / n**2 * np.sum((d - d.mean()) ** 2)
    return TestResult(
        name="Watson two-sample U2",
        statistic=float(u2),
        p_value=_watson_u2_sf(u2),
        critical=_watson_u2_isf(1 - level),
        level=level,
        source="Mardia & Jupp (2000), eq. 8.3.7",
    )


def equal_kappa_test(samples, method=None, level=0.95):
    """Test whether samples of directions share the concentration.

    For concentrated data ``2 kappa (n_i - R_i)`` is approximately chi-square
    with ``nu_i = (p - 1)(n_i - 1)`` degrees of freedom in ``p`` dimensions.

    - "f" (default for two samples): ``F = ((n_1 - R_1) / nu_1) / ((n_2 -
      R_2) / nu_2)``, two-sided, ``F(nu_1, nu_2)`` (MJ eq. 10.5.3; Watson,
      1956).
    - "bartlett" (default for more samples): Bartlett's test of the
      dispersions ``(n_i - R_i) / nu_i``, chi-square with ``q - 1`` degrees of
      freedom (MJ eq. 10.6.21; McFadden & Lowes, 1981, eq. 64-66).

    Both need concentrated data (``Rbar >= 0.67`` on the sphere, ``Rbar >=
    0.7`` on the circle); MJ eq. 10.6.24-10.6.27 and 7.3.24-7.3.25 give
    variance-stabilised tests for dispersed data.

    Args:
        samples: sequence of ``(n_i, 2)`` or ``(n_i, 3)`` unit vector arrays.
        method: "f" or "bartlett". Default None.
        level: confidence level. Default 0.95.

    Returns:
        TestResult: ``reject()`` is True when the concentrations differ.
    """
    n_i, R_i, _, p = _resultants(samples)
    q = len(n_i)
    nu = (p - 1) * (n_i - 1)
    dev = n_i - R_i
    method = method or ("f" if q == 2 else "bartlett")
    if method == "f":
        if q != 2:
            raise ValueError("The F test compares two samples")
        F = (dev[0] / nu[0]) / (dev[1] / nu[1])
        p_value = 2 * min(stats.f.cdf(F, nu[0], nu[1]), stats.f.sf(F, nu[0], nu[1]))
        return TestResult(
            name="equal concentration F",
            statistic=float(F),
            p_value=float(min(p_value, 1.0)),
            df=(int(nu[0]), int(nu[1])),
            level=level,
            source="Mardia & Jupp (2000), eq. 10.5.3",
        )
    if method == "bartlett":
        total = nu.sum()
        B = total * np.log(dev.sum() / total) - np.sum(nu * np.log(dev / nu))
        B /= 1 + (np.sum(1 / nu) - 1 / total) / (3 * (q - 1))
        return _chi2_result(
            "equal concentration Bartlett",
            B,
            q - 1,
            level,
            "Mardia & Jupp (2000), eq. 10.6.21",
        )
    raise ValueError(f"Unknown method {method!r}, expected f or bartlett")


def axial_common_mean_test(samples, level=0.95):
    """Test whether samples of axes share the principal axis.

    ``N = (sum(tau_i) - tau) / g`` with the largest eigenvalues ``tau_i`` of the
    unnormalized orientation tensors of the samples, ``tau`` that of the pooled
    sample and ``g = (tau_bar - G) / (3 tau_bar - 1)`` from the pooled
    normalized eigenvalue ``tau_bar`` and ``G = mean((u.x)^4)``; ``N`` is
    chi-square with ``2(q - 1)`` degrees of freedom for samples of rotationally
    symmetric (e.g. Watson) distributions with a common shape (FLE eq.
    7.72-7.76).

    Note:
        FLE prints ``g`` as a product and with a factor 2, ``2 (tau_bar - G) /
        (3 tau_bar - 1)``, taken from Watson (1983, eq. 4.19), whose factor 2 is
        spurious: with it the test has a size of 0.3 % at the 5 % level, without
        it the nominal size.

    Args:
        samples: sequence of ``(n_i, 3)`` unit vector arrays of axes.
        level: confidence level. Default 0.95.

    Returns:
        TestResult: ``reject()`` is True when the principal axes differ.
    """
    samples = [np.asarray(x, dtype=float) for x in samples]
    if len(samples) < 2:
        raise ValueError("At least two samples are needed")
    taus = [np.linalg.eigvalsh(x.T @ x)[-1] for x in samples]
    X = np.vstack(samples)
    w, V = np.linalg.eigh(X.T @ X)
    tau_bar = w[-1] / len(X)
    G = np.mean((X @ V[:, -1]) ** 4)
    g = (tau_bar - G) / (3 * tau_bar - 1)
    N = (np.sum(taus) - w[-1]) / g
    return _chi2_result(
        "axial common mean",
        N,
        2 * (len(samples) - 1),
        level,
        "Fisher, Lewis & Embleton (1987), eq. 7.72-7.76",
    )


def frame_uniformity_test(frames, level=0.95):
    """Test of uniformity of orthogonal axial frames.

    For ``n`` frames of ``r`` orthonormal axes ``u_j`` in ``p`` dimensions the
    statistic is ``S = n (p - 1)(p + 2) / 2 (sum_j tr(T_j^2) + tr(T_+^2) / (p -
    r))`` with ``T_j = mean(u_j u_j') - I / p`` and ``T_+ = -sum_j T_j`` the
    tensor of the remaining axes; it is asymptotically chi-square with ``r(p -
    1)(p + 2) / 2`` degrees of freedom, ``(p - 1)^2 (p + 2) / 2`` when the
    frames are complete (Arnold & Jupp, 2013, eq. 5, Prop. 1). For a single
    axis it is the Bingham test. A pair (lineation and foliation normal) is a
    complete frame in 3D, with 10 degrees of freedom.

    Args:
        frames: ``(n, r, p)`` array of orthonormal axes.
        level: confidence level. Default 0.95.

    Returns:
        TestResult: ``reject()`` is True when the frames are not uniform; the
        ``extra`` key ``dispersion`` holds ``d1 = sum_j (1 - tr((T_j + I /
        p)^2))`` (Arnold & Jupp, 2013, sec. 2.3).

    References:
        Arnold, R. & Jupp, P.E. (2013). Statistics of orthogonal axial frames.
            Biometrika 100, 571-586.
    """
    U = np.asarray(frames, dtype=float)
    n, r, p = U.shape
    T = np.einsum("nja,njb->jab", U, U) / n - np.eye(p) / p
    total = np.sum(T**2)
    if r < p:
        total += np.sum(T.sum(axis=0) ** 2) / (p - r)
    S = n * (p - 1) * (p + 2) / 2 * total
    df = min(r, p - 1) * (p - 1) * (p + 2) // 2
    d1 = float(sum(1 - np.sum((Tj + np.eye(p) / p) ** 2) for Tj in T))
    return TestResult(
        name="Arnold-Jupp frame uniformity",
        statistic=float(S),
        p_value=float(stats.chi2.sf(S, df)),
        df=int(df),
        critical=float(stats.chi2.ppf(level, df)),
        level=level,
        source="Arnold & Jupp (2013), eq. 5",
        extra={"dispersion": d1},
    )
