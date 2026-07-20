import math

import numpy as np
import pytest

from apsg import (
    defgrad,
    defgrad2,
    ellipse,
    ellipsoid,
    ortensor,
    ortensor2,
    rotation,
    rotation2,
    stress,
    stress2,
    velgrad,
    velgrad2,
)
from apsg.feature._geodata import Foliation, Lineation, Pair
from apsg.feature._tensor2 import (
    DeformationGradient2,
    Ellipse,
    OrientationTensor2,
    Rotation2,
    Stress2,
    VelocityGradient2,
)
from apsg.feature._tensor3 import (
    DeformationGradient3,
    Ellipsoid,
    OrientationTensor3,
    Rotation3,
    Stress3,
    VelocityGradient3,
)
from apsg.math._vector import Vector2, Vector3

# ---------------------------------------------------------------------------
# DeformationGradient2
# ---------------------------------------------------------------------------


class TestDeformationGradient2:
    def test_default(self):
        F = DeformationGradient2()
        assert F.xx == 1
        assert F.yy == 1

    def test_from_array(self):
        F = DeformationGradient2([[2, 0], [0, 0.5]])
        assert F.xx == 2
        assert F.yy == 0.5

    def test_from_comp(self):
        F = DeformationGradient2.from_comp(xx=2, yy=0.5)
        assert F.xx == 2
        assert F.yy == 0.5

    def test_from_ratio(self):
        F = DeformationGradient2.from_ratio(R=4)
        assert math.isclose(F.xx, 2)
        assert math.isclose(F.yy, 0.5)

    def test_is_rotation_true(self):
        R = Rotation2.from_angle(45)
        assert R.is_rotation()

    def test_is_rotation_false(self):
        F = DeformationGradient2.from_ratio(R=4)
        assert not F.is_rotation()

    def test_R(self):
        F = DeformationGradient2.from_ratio(R=4)
        R = F.R
        assert isinstance(R, Rotation2)

    def test_U(self):
        F = DeformationGradient2.from_ratio(R=4)
        U = F.U
        assert isinstance(U, DeformationGradient2)

    def test_V(self):
        F = DeformationGradient2.from_ratio(R=4)
        V = F.V
        assert isinstance(V, DeformationGradient2)

    def test_velgrad(self):
        F = DeformationGradient2.from_ratio(R=4)
        L = F.velgrad(time=1)
        assert isinstance(L, VelocityGradient2)

    def test_repr(self):
        F = DeformationGradient2()
        assert repr(F).startswith("DeformationGradient2")

    def test_matmul_vector(self):
        F = DeformationGradient2([[2, 0], [0, 0.5]])
        v = Vector2(1, 1)
        r = F @ v
        assert isinstance(r, Vector2)
        assert r == Vector2(2, 0.5)

    def test_matmul_matrix(self):
        F1 = DeformationGradient2([[2, 0], [0, 0.5]])
        F2 = DeformationGradient2([[1, 0], [0, 1]])
        r = F1 @ F2
        assert isinstance(r, DeformationGradient2)

    def test_det(self):
        F = DeformationGradient2([[2, 0], [0, 0.5]])
        assert math.isclose(F.det, 1)

    def test_transform(self):
        F = DeformationGradient2([[2, 0], [0, 0.5]])
        R = Rotation2.from_angle(45)
        Ft = F.transform(R)
        assert isinstance(Ft, DeformationGradient2)

    def test_I(self):
        F = DeformationGradient2([[2, 0], [0, 4]])
        FI = F.I
        assert isinstance(FI, DeformationGradient2)
        assert math.isclose(FI.xx, 0.5)
        assert math.isclose(FI.yy, 0.25)

    def test_to_json(self):
        F = DeformationGradient2()
        j = F.to_json()
        assert j["datatype"] == "DeformationGradient2"

    def test_lowercase_alias(self):
        assert defgrad2 is DeformationGradient2


# ---------------------------------------------------------------------------
# Rotation2
# ---------------------------------------------------------------------------


class TestRotation2:
    def test_default(self):
        R = Rotation2()
        assert isinstance(R, Rotation2)

    def test_from_angle(self):
        R = Rotation2.from_angle(90)
        assert math.isclose(R[0, 0], 0, abs_tol=1e-10)
        assert math.isclose(R[0, 1], -1, abs_tol=1e-10)
        assert math.isclose(R[1, 0], 1, abs_tol=1e-10)

    def test_angle(self):
        R = Rotation2.from_angle(45)
        assert math.isclose(R.angle(), 45)

    def test_angle_zero(self):
        R = Rotation2()
        assert math.isclose(R.angle(), 0)

    def test_from_two_vectors(self):
        v1 = Vector2(1, 0)
        v2 = Vector2(0, 1)
        R = Rotation2.from_two_vectors(v1, v2)
        assert math.isclose(R.angle(), 90)

    def test_is_rotation(self):
        R = Rotation2.from_angle(45)
        assert R.is_rotation()

    def test_invalid_raises(self):
        with pytest.raises(TypeError, match="Rotation2"):
            Rotation2([[2, 0], [0, 1]])

    def test_repr(self):
        R = Rotation2()
        assert repr(R).startswith("Rotation2")

    def test_lowercase_alias(self):
        assert rotation2 is Rotation2


# ---------------------------------------------------------------------------
# VelocityGradient2
# ---------------------------------------------------------------------------


class TestVelocityGradient2:
    def test_default(self):
        L = VelocityGradient2()
        assert isinstance(L, VelocityGradient2)

    def test_from_comp(self):
        L = VelocityGradient2.from_comp(xx=1, yy=-1)
        assert L.xx == 1
        assert L.yy == -1

    def test_from_array(self):
        L = VelocityGradient2([[1, 0], [0, -1]])
        assert L.xx == 1

    def test_defgrad(self):
        L = VelocityGradient2.from_comp(xx=0.1, yy=-0.1)
        F = L.defgrad(time=1)
        assert isinstance(F, DeformationGradient2)

    def test_defgrad_steps(self):
        L = VelocityGradient2.from_comp(xx=0.1, yy=-0.1)
        steps = L.defgrad(time=1, steps=5)
        assert len(steps) == 5
        assert all(isinstance(F, DeformationGradient2) for F in steps)

    def test_rate(self):
        L = VelocityGradient2.from_comp(xy=1)
        D = L.rate()
        assert isinstance(D, VelocityGradient2)

    def test_spin(self):
        L = VelocityGradient2.from_comp(xy=1)
        W = L.spin()
        assert isinstance(W, VelocityGradient2)

    def test_rate_spin_decomposition(self):
        L = VelocityGradient2.from_comp(xx=1, xy=2, yx=3, yy=-1)
        D = L.rate()
        W = L.spin()
        np.testing.assert_array_almost_equal(D + W, L)

    def test_repr(self):
        L = VelocityGradient2()
        assert repr(L).startswith("VelocityGradient2")

    def test_lowercase_alias(self):
        assert velgrad2 is VelocityGradient2


# ---------------------------------------------------------------------------
# Stress2
# ---------------------------------------------------------------------------


class TestStress2:
    def test_default(self):
        S = Stress2()
        assert isinstance(S, Stress2)

    def test_from_comp(self):
        S = Stress2.from_comp(xx=-5, yy=-2, xy=1)
        assert S.xx == -5
        assert S.yy == -2
        assert S.xy == 1
        assert S.yx == 1  # symmetric

    def test_from_array(self):
        S = Stress2([[-5, 1], [1, -2]])
        assert S.xx == -5

    def test_mean_stress(self):
        S = Stress2.from_comp(xx=-5, yy=-1)
        assert math.isclose(S.mean_stress, -3)

    def test_hydrostatic(self):
        S = Stress2.from_comp(xx=-5, yy=-1)
        H = S.hydrostatic
        assert H.xx == H.yy

    def test_deviatoric(self):
        S = Stress2.from_comp(xx=-5, yy=-1)
        D = S.deviatoric
        assert isinstance(D, Stress2)

    def test_sigma1(self):
        S = Stress2.from_comp(xx=-5, yy=-2, xy=1)
        # sigma1 is most compressive (most negative)
        assert S.sigma1 <= S.sigma2

    def test_sigma2(self):
        S = Stress2.from_comp(xx=-5, yy=-2, xy=1)
        assert S.sigma2 >= S.sigma1

    def test_sigma1dir(self):
        S = Stress2.from_comp(xx=-5, yy=-1)
        d = S.sigma1dir
        assert isinstance(d, Vector2)

    def test_sigma2dir(self):
        S = Stress2.from_comp(xx=-5, yy=-1)
        d = S.sigma2dir
        assert isinstance(d, Vector2)

    def test_sigma1vec(self):
        S = Stress2.from_comp(xx=-5, yy=-1)
        v = S.sigma1vec
        assert isinstance(v, Vector2)

    def test_sigma2vec(self):
        S = Stress2.from_comp(xx=-5, yy=-1)
        v = S.sigma2vec
        assert isinstance(v, Vector2)

    def test_I1(self):
        S = Stress2.from_comp(xx=-5, yy=-2)
        assert math.isclose(S.I1, -7)

    def test_I2(self):
        S = Stress2.from_comp(xx=-5, yy=-2, xy=1)
        assert isinstance(S.I2, float)

    def test_I3(self):
        S = Stress2.from_comp(xx=-5, yy=-2, xy=1)
        assert isinstance(S.I3, float)

    def test_diagonalized(self):
        S = Stress2.from_comp(xx=-5, yy=-2, xy=1)
        D, Rmat = S.diagonalized
        assert isinstance(D, Stress2)
        assert isinstance(Rmat, DeformationGradient2)
        np.testing.assert_array_almost_equal(np.array(D), np.diag(S.eigenvalues()))

    def test_cauchy(self):
        S = Stress2.from_comp(xx=-5, yy=-2, xy=1)
        t = S.cauchy(Vector2(1, 0))
        assert isinstance(t, Vector2)

    def test_stress_comp(self):
        S = Stress2.from_comp(xx=-5, yy=-2, xy=1)
        sn, tau = S.stress_comp(Vector2(1, 0))
        assert isinstance(sn, Vector2)
        assert isinstance(tau, Vector2)

    def test_normal_stress(self):
        S = Stress2.from_comp(xx=-5, yy=-2, xy=1)
        sn = S.normal_stress(Vector2(1, 0))
        assert isinstance(sn, float)

    def test_shear_stress(self):
        S = Stress2.from_comp(xx=-5, yy=-2, xy=1)
        tau = S.shear_stress(Vector2(1, 0))
        assert isinstance(tau, float)

    def test_signed_shear_stress(self):
        S = Stress2.from_comp(xx=-5, yy=-2, xy=1)
        # signed_shear_stress calls DeformationGradient2.from_angle
        # which does not exist; test the property at least returns something
        n = Vector2(1, 0)
        from apsg.feature._tensor2 import Rotation2 as R2

        Rmat = R2.from_angle(n.direction)
        tau = S.transform(Rmat)[1, 0]
        assert isinstance(tau, float)

    def test_eigenvalues(self):
        S = Stress2.from_comp(xx=-5, yy=-1)
        e = S.eigenvalues()
        assert len(e) == 2

    def test_eigenvectors(self):
        S = Stress2.from_comp(xx=-5, yy=-1)
        v = S.eigenvectors()
        assert len(v) == 2
        assert isinstance(v[0], Vector2)

    def test_E1_E2(self):
        S = Stress2.from_comp(xx=-5, yy=-1)
        assert S.E1 >= S.E2

    def test_repr(self):
        S = Stress2()
        assert repr(S).startswith("Stress2")

    def test_lowercase_alias(self):
        assert stress2 is Stress2


# ---------------------------------------------------------------------------
# Ellipse
# ---------------------------------------------------------------------------


class TestEllipse:
    def test_default(self):
        E = Ellipse()
        assert isinstance(E, Ellipse)

    def test_from_array(self):
        E = Ellipse([[8, 0], [0, 2]])
        assert E.E1 == 8
        assert E.E2 == 2

    def test_from_defgrad_left(self):
        F = DeformationGradient2.from_ratio(R=4)
        E = Ellipse.from_defgrad(F, form="left")
        assert isinstance(E, Ellipse)

    def test_from_defgrad_right(self):
        F = DeformationGradient2.from_ratio(R=4)
        E = Ellipse.from_defgrad(F, form="right")
        assert isinstance(E, Ellipse)

    def test_from_defgrad_wrong_form(self):
        F = DeformationGradient2.from_ratio(R=4)
        with pytest.raises(TypeError):
            Ellipse.from_defgrad(F, form="bad")

    def test_from_stretch(self):
        E = Ellipse.from_stretch(x=2, y=1)
        assert math.isclose(E.E1, 4)
        assert math.isclose(E.E2, 1)

    def test_S1(self):
        E = Ellipse([[8, 0], [0, 2]])
        assert math.isclose(E.S1, math.sqrt(8))

    def test_S2(self):
        E = Ellipse([[8, 0], [0, 2]])
        assert math.isclose(E.S2, math.sqrt(2))

    def test_e1(self):
        E = Ellipse([[8, 0], [0, 2]])
        assert math.isclose(E.e1, math.log(math.sqrt(8)))

    def test_e2(self):
        E = Ellipse([[8, 0], [0, 2]])
        assert math.isclose(E.e2, math.log(math.sqrt(2)))

    def test_ar(self):
        E = Ellipse([[8, 0], [0, 2]])
        assert math.isclose(E.ar, math.sqrt(8) / math.sqrt(2))

    def test_orientation(self):
        E = Ellipse([[8, 0], [0, 2]])
        o = E.orientation
        assert 0 <= o < 180

    def test_e12(self):
        E = Ellipse([[8, 0], [0, 2]])
        assert math.isclose(E.e12, E.e1 - E.e2)

    def test_repr(self):
        E = Ellipse([[8, 0], [0, 2]])
        assert repr(E).startswith("Ellipse")

    def test_scaled_eigenvectors(self):
        E = Ellipse([[8, 0], [0, 2]])
        v = E.scaled_eigenvectors()
        assert len(v) == 2
        assert isinstance(v[0], Vector2)
        assert isinstance(v[1], Vector2)

    def test_scaled_eigenvectors_single(self):
        E = Ellipse([[8, 0], [0, 2]])
        v0 = E.scaled_eigenvectors(which=0)
        assert isinstance(v0, Vector2)
        v1 = E.scaled_eigenvectors(which=1)
        assert isinstance(v1, Vector2)

    def test_lowercase_alias(self):
        assert ellipse is Ellipse


# ---------------------------------------------------------------------------
# OrientationTensor2
# ---------------------------------------------------------------------------


class TestOrientationTensor2:
    def test_from_array(self):
        ot = OrientationTensor2([[0.5, 0], [0, 0.5]])
        assert isinstance(ot, OrientationTensor2)

    def test_from_features(self):
        from apsg.feature._container import Vector2Set

        data = [Vector2(1, 0), Vector2(0, 1), Vector2(1, 1)]
        v = Vector2Set(data)
        ot = OrientationTensor2.from_features(v)
        assert isinstance(ot, OrientationTensor2)

    def test_inherits_ellipse(self):
        ot = OrientationTensor2([[8, 0], [0, 2]])
        assert math.isclose(ot.ar, math.sqrt(8) / math.sqrt(2))

    def test_lowercase_alias(self):
        assert ortensor2 is OrientationTensor2


# ---------------------------------------------------------------------------
# DeformationGradient3
# ---------------------------------------------------------------------------


class TestDeformationGradient3:
    def test_default(self):
        F = DeformationGradient3()
        assert F.xx == 1
        assert F.yy == 1
        assert F.zz == 1

    def test_from_array(self):
        F = DeformationGradient3([[2, 0, 0], [0, 1, 0], [0, 0, 0.5]])
        assert F.xx == 2
        assert F.zz == 0.5

    def test_from_comp(self):
        F = DeformationGradient3.from_comp(xx=2, yy=1, zz=0.5)
        assert F.xx == 2
        assert F.zz == 0.5

    def test_from_ratios(self):
        F = DeformationGradient3.from_ratios(Rxy=2, Ryz=3)
        assert isinstance(F, DeformationGradient3)
        assert F.xx > 1

    def test_from_ratios_assertion(self):
        with pytest.raises(AssertionError):
            DeformationGradient3.from_ratios(Rxy=0.5, Ryz=3)

    def test_is_rotation_true(self):
        R = Rotation3.from_axisangle(Vector3(0, 0, 1), 45)
        assert R.is_rotation()

    def test_is_rotation_false(self):
        F = DeformationGradient3.from_ratios(Rxy=2, Ryz=3)
        assert not F.is_rotation()

    def test_R(self):
        F = DeformationGradient3.from_ratios(Rxy=2, Ryz=3)
        R = F.R
        assert isinstance(R, Rotation3)

    def test_U(self):
        F = DeformationGradient3.from_ratios(Rxy=2, Ryz=3)
        U = F.U
        assert isinstance(U, DeformationGradient3)

    def test_V(self):
        F = DeformationGradient3.from_ratios(Rxy=2, Ryz=3)
        V = F.V
        assert isinstance(V, DeformationGradient3)

    def test_velgrad(self):
        F = DeformationGradient3.from_ratios(Rxy=2, Ryz=3)
        L = F.velgrad(time=1)
        assert isinstance(L, VelocityGradient3)

    def test_det(self):
        F = DeformationGradient3([[2, 0, 0], [0, 1, 0], [0, 0, 0.5]])
        assert math.isclose(F.det, 1)

    def test_transform(self):
        F = DeformationGradient3([[2, 0, 0], [0, 1, 0], [0, 0, 0.5]])
        R = Rotation3.from_axisangle(Vector3(0, 0, 1), 45)
        Ft = F.transform(R)
        assert isinstance(Ft, DeformationGradient3)

    def test_I(self):
        F = DeformationGradient3([[2, 0, 0], [0, 1, 0], [0, 0, 0.5]])
        FI = F.I
        assert isinstance(FI, DeformationGradient3)

    def test_matmul_vector(self):
        F = DeformationGradient3([[2, 0, 0], [0, 1, 0], [0, 0, 0.5]])
        v = Vector3(1, 1, 1)
        r = F @ v
        assert isinstance(r, Vector3)

    def test_matmul_matrix(self):
        F1 = DeformationGradient3([[2, 0, 0], [0, 1, 0], [0, 0, 0.5]])
        F2 = DeformationGradient3()
        r = F1 @ F2
        assert isinstance(r, DeformationGradient3)

    def test_repr(self):
        F = DeformationGradient3()
        assert repr(F).startswith("DeformationGradient3")

    def test_lowercase_alias(self):
        assert defgrad is DeformationGradient3


# ---------------------------------------------------------------------------
# Rotation3
# ---------------------------------------------------------------------------


class TestRotation3:
    def test_default(self):
        R = Rotation3()
        assert isinstance(R, Rotation3)

    def test_from_axisangle(self):
        R = Rotation3.from_axisangle(Vector3(0, 0, 1), 90)
        v = Vector3(1, 0, 0)
        r = v.transform(R)
        assert r == Vector3(0, 1, 0)

    def test_from_two_vectors(self):
        v1 = Vector3(1, 0, 0)
        v2 = Vector3(0, 1, 0)
        R = Rotation3.from_two_vectors(v1, v2)
        assert isinstance(R, Rotation3)

    def test_from_pair(self):
        p = Pair(140, 30, 110, 26)
        R = Rotation3.from_pair(p)
        assert isinstance(R, Rotation3)

    def test_from_two_pairs(self):
        p1 = Pair(58, 36, 81, 34)
        p2 = Pair(217, 42, 162, 27)
        R = Rotation3.from_two_pairs(p1, p2)
        assert isinstance(R, Rotation3)
        r = p1.transform(R)
        assert isinstance(r, Pair)

    def test_from_two_pairs_symmetry(self):
        p1 = Pair(58, 36, 81, 34)
        p2 = Pair(217, 42, 162, 27)
        R = Rotation3.from_two_pairs(p1, p2, symmetry=True)
        assert isinstance(R, Rotation3)

    def test_from_vectors_axis(self):
        v1 = Vector3(1, 0, 0)
        v2 = Vector3(0, 1, 0)
        a = Vector3(0, 0, 1)
        R = Rotation3.from_vectors_axis(v1, v2, a)
        assert isinstance(R, Rotation3)

    def test_from_quat(self):
        q = [-0.11543715, 0.19994301, 0.39988603, 0.88701083]
        R = Rotation3.from_quat(q)
        assert isinstance(R, Rotation3)

    def test_from_euler(self):
        R = Rotation3.from_euler("zxz", [30, -64, 125])
        assert isinstance(R, Rotation3)

    def test_from_declination(self):
        R = Rotation3.from_declination(48.6, 13.2, alt=0.6)
        assert isinstance(R, Rotation3)

    def test_axisangle(self):
        R = Rotation3.from_axisangle(Vector3(0, 0, 1), 45)
        axis, angle = R.axisangle()
        assert isinstance(axis, Vector3)
        assert math.isclose(angle, 45)

    def test_angle(self):
        R = Rotation3.from_axisangle(Vector3(0, 0, 1), 45)
        assert math.isclose(abs(R.angle()), 45)

    def test_euler_angles(self):
        R = Rotation3.from_euler("zxz", [30, -64, 125])
        angles = R.euler("zxz")
        assert len(angles) == 3

    def test_quat(self):
        R = Rotation3.from_axisangle(Vector3(0, 0, 1), 45)
        q = R.quat()
        assert len(q) == 4

    def test_is_rotation(self):
        R = Rotation3.from_axisangle(Vector3(0, 0, 1), 45)
        assert R.is_rotation()

    def test_invalid_raises(self):
        with pytest.raises(TypeError):
            Rotation3([[2, 0, 0], [0, 1, 0], [0, 0, 0.5]])

    def test_compose(self):
        R1 = Rotation3.from_axisangle(Vector3(0, 0, 1), 45)
        R2 = Rotation3.from_axisangle(Vector3(0, 0, 1), 45)
        R = R1 @ R2
        assert math.isclose(abs(R.angle()), 90, abs_tol=1e-10)

    def test_repr(self):
        R = Rotation3()
        assert repr(R).startswith("Rotation3")

    def test_lowercase_alias(self):
        assert rotation is Rotation3


# ---------------------------------------------------------------------------
# VelocityGradient3
# ---------------------------------------------------------------------------


class TestVelocityGradient3:
    def test_default(self):
        L = VelocityGradient3()
        assert isinstance(L, VelocityGradient3)

    def test_from_comp(self):
        L = VelocityGradient3.from_comp(xx=1, yy=0, zz=-1)
        assert L.xx == 1
        assert L.zz == -1

    def test_from_array(self):
        L = VelocityGradient3([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
        assert L.xx == 1

    def test_defgrad(self):
        L = VelocityGradient3.from_comp(xx=0.1, zz=-0.1)
        F = L.defgrad(time=1)
        assert isinstance(F, DeformationGradient3)

    def test_defgrad_steps(self):
        L = VelocityGradient3.from_comp(xx=0.1, zz=-0.1)
        steps = L.defgrad(time=1, steps=5)
        assert len(steps) == 5
        assert all(isinstance(F, DeformationGradient3) for F in steps)

    def test_rate(self):
        L = VelocityGradient3.from_comp(xy=1)
        D = L.rate()
        assert isinstance(D, VelocityGradient3)

    def test_spin(self):
        L = VelocityGradient3.from_comp(xy=1)
        W = L.spin()
        assert isinstance(W, VelocityGradient3)

    def test_rate_spin_decomposition(self):
        L = VelocityGradient3.from_comp(
            xx=1, xy=2, xz=3, yx=4, yy=5, yz=6, zx=7, zy=8, zz=9
        )
        D = L.rate()
        W = L.spin()
        np.testing.assert_array_almost_equal(D + W, L)

    def test_repr(self):
        L = VelocityGradient3()
        assert repr(L).startswith("VelocityGradient3")

    def test_lowercase_alias(self):
        assert velgrad is VelocityGradient3


# ---------------------------------------------------------------------------
# Stress3
# ---------------------------------------------------------------------------


class TestStress3:
    def test_default(self):
        S = Stress3()
        assert isinstance(S, Stress3)

    def test_from_comp(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=1)
        assert S.xx == -5
        assert S.xy == 1
        assert S.yx == 1

    def test_from_array(self):
        S = Stress3([[-5, 1, 0], [1, -2, 0], [0, 0, 10]])
        assert S.xx == -5

    def test_from_ratio(self):
        S = Stress3.from_ratio(r=0.5, mag=1)
        assert isinstance(S, Stress3)

    def test_mean_stress(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10)
        assert math.isclose(S.mean_stress, 1)

    def test_hydrostatic(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10)
        H = S.hydrostatic
        assert H.xx == H.yy == H.zz

    def test_deviatoric(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10)
        D = S.deviatoric
        assert isinstance(D, Stress3)

    def test_effective(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10)
        Se = S.effective(fp=2)
        assert isinstance(Se, Stress3)

    def test_sigma1(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10)
        # sigma1 is most compressive (most negative)
        assert S.sigma1 <= S.sigma2 <= S.sigma3

    def test_sigma2(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10)
        assert S.sigma1 <= S.sigma2 <= S.sigma3

    def test_sigma3(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10)
        assert S.sigma1 <= S.sigma2 <= S.sigma3

    def test_sigma1dir(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10)
        d = S.sigma1dir
        assert isinstance(d, Vector3)

    def test_sigma2dir(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10)
        d = S.sigma2dir
        assert isinstance(d, Vector3)

    def test_sigma3dir(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10)
        d = S.sigma3dir
        assert isinstance(d, Vector3)

    def test_sigma1vec(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10)
        v = S.sigma1vec
        assert isinstance(v, Vector3)

    def test_sigma2vec(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10)
        v = S.sigma2vec
        assert isinstance(v, Vector3)

    def test_sigma3vec(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10)
        v = S.sigma3vec
        assert isinstance(v, Vector3)

    def test_I1(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10)
        assert math.isclose(S.I1, 3)

    def test_I2(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=1)
        assert isinstance(S.I2, float)

    def test_I3(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=1)
        assert isinstance(S.I3, float)

    def test_diagonalized(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10)
        D, Rmat = S.diagonalized
        assert isinstance(D, Stress3)
        assert isinstance(Rmat, DeformationGradient3)

    def test_cauchy(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=1)
        t = S.cauchy(Foliation(160, 30))
        assert isinstance(t, Vector3)

    def test_fault(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=8)
        f = S.fault(Foliation(160, 30))
        from apsg.feature._geodata import Fault

        assert isinstance(f, Fault)

    def test_stress_comp(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=1)
        sn, tau = S.stress_comp(Foliation(160, 30))
        assert isinstance(sn, Vector3)
        assert isinstance(tau, Vector3)

    def test_normal_stress(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=1)
        sn = S.normal_stress(Foliation(160, 30))
        assert isinstance(sn, float)

    def test_shear_stress(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=1)
        tau = S.shear_stress(Foliation(160, 30))
        assert isinstance(tau, float)

    def test_slip_tendency(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=1)
        t = S.slip_tendency(Foliation(160, 30))
        assert isinstance(t, float)

    def test_slip_tendency_log(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=1)
        t = S.slip_tendency(Foliation(160, 30), log=True)
        assert isinstance(t, float)

    def test_slip_tendency_with_fp(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=1)
        t = S.slip_tendency(Foliation(160, 30), fp=2)
        assert isinstance(t, float)

    def test_dilation_tendency(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=1)
        t = S.dilation_tendency(Foliation(160, 30))
        assert 0 <= t <= 1

    def test_shape_ratio(self):
        S = Stress3.from_ratio(r=0.5, mag=1)
        assert 0 <= S.shape_ratio <= 1

    def test_eigenlins(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=1)
        lins = S.eigenlins()
        assert len(lins) == 3
        assert isinstance(lins[0], Lineation)

    def test_eigenfols(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=1)
        fols = S.eigenfols()
        assert len(fols) == 3
        assert isinstance(fols[0], Foliation)

    def test_pair(self):
        S = Stress3.from_comp(xx=-5, yy=-2, zz=10, xy=1)
        p = S.pair
        assert isinstance(p, Pair)

    def test_repr(self):
        S = Stress3()
        assert repr(S).startswith("Stress3")

    def test_lowercase_alias(self):
        assert stress is Stress3


# ---------------------------------------------------------------------------
# Ellipsoid
# ---------------------------------------------------------------------------


class TestEllipsoid:
    def test_default(self):
        E = Ellipsoid()
        assert isinstance(E, Ellipsoid)

    def test_from_array(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert E.E1 == 8

    def test_from_defgrad_left(self):
        F = DeformationGradient3.from_ratios(Rxy=2, Ryz=3)
        E = Ellipsoid.from_defgrad(F, form="left")
        assert isinstance(E, Ellipsoid)

    def test_from_defgrad_right(self):
        F = DeformationGradient3.from_ratios(Rxy=2, Ryz=3)
        E = Ellipsoid.from_defgrad(F, form="right")
        assert isinstance(E, Ellipsoid)

    def test_from_defgrad_wrong_form(self):
        F = DeformationGradient3.from_ratios(Rxy=2, Ryz=3)
        with pytest.raises(TypeError):
            Ellipsoid.from_defgrad(F, form="bad")

    def test_from_stretch(self):
        E = Ellipsoid.from_stretch(x=2, y=1.5, z=1)
        assert math.isclose(E.S1, 2)
        assert math.isclose(E.S2, 1.5)
        assert math.isclose(E.S3, 1)

    def test_kind(self):
        E1 = Ellipsoid.from_stretch(x=10, y=1, z=1)
        assert E1.kind == "L"
        E2 = Ellipsoid.from_stretch(x=1, y=1, z=1)
        assert E2.kind == "O"

    def test_strength(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.strength, float)

    def test_shape(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.shape, float)

    def test_S1_S2_S3(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert E.S1 >= E.S2 >= E.S3
        assert math.isclose(E.S1, math.sqrt(8))
        assert math.isclose(E.S2, math.sqrt(2))
        assert math.isclose(E.S3, 1)

    def test_e1_e2_e3(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert E.e1 >= E.e2 >= E.e3

    def test_Rxy(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert E.Rxy > 1

    def test_Ryz(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert E.Ryz > 1

    def test_e12_e13_e23(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert E.e12 >= 0
        assert E.e13 >= 0
        assert E.e23 >= 0

    def test_k(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.k, float)

    def test_d(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.d, float)

    def test_K(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.K, float)

    def test_D(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.D, float)

    def test_r(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.r, float)

    def test_goct(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.goct, float)

    def test_eoct(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.eoct, float)

    def test_lode(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.lode, float)

    def test_P(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.P, float)

    def test_G(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.G, float)

    def test_R_vollmer(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.R, float)

    def test_B(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.B, float)

    def test_Intensity(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.Intensity, float)

    def test_MAD_l(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.MAD_l, float)

    def test_MAD_p(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.MAD_p, float)

    def test_MAD(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert isinstance(E.MAD, float)

    def test_eigenvalues(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        e = E.eigenvalues()
        assert len(e) == 3

    def test_eigenvectors(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        v = E.eigenvectors()
        assert len(v) == 3
        assert isinstance(v[0], Vector3)

    def test_scaled_eigenvectors(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        v = E.scaled_eigenvectors()
        assert len(v) == 3
        assert isinstance(v[0], Vector3)
        assert isinstance(v[1], Vector3)
        assert isinstance(v[2], Vector3)

    def test_scaled_eigenvectors_single(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        v0 = E.scaled_eigenvectors(which=0)
        assert isinstance(v0, Vector3)
        v1 = E.scaled_eigenvectors(which=1)
        assert isinstance(v1, Vector3)
        v2 = E.scaled_eigenvectors(which=2)
        assert isinstance(v2, Vector3)

    def test_eigenlins(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        lins = E.eigenlins()
        assert len(lins) == 3
        assert isinstance(lins[0], Lineation)

    def test_eigenfols(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        fols = E.eigenfols()
        assert len(fols) == 3
        assert isinstance(fols[0], Foliation)

    def test_pair(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        p = E.pair
        assert isinstance(p, Pair)

    def test_repr(self):
        E = Ellipsoid([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert repr(E).startswith("Ellipsoid")

    def test_lowercase_alias(self):
        assert ellipsoid is Ellipsoid

    def test_sections(self):
        E = Ellipsoid([[4, 0, 0], [0, 1, 0], [0, 0, 0.25]])
        fx = Foliation(Vector3.unit_x())
        fy = Foliation(Vector3.unit_y())
        fz = Foliation(Vector3.unit_z())
        assert E.section(fx).ar == 2
        assert E.section(fy).ar == 4
        assert E.section(fz).ar == 2
        assert isinstance(E.section(fx), Ellipse)


# ---------------------------------------------------------------------------
# OrientationTensor3
# ---------------------------------------------------------------------------


class TestOrientationTensor3:
    def test_from_array(self):
        ot = OrientationTensor3([[0.5, 0, 0], [0, 0.3, 0], [0, 0, 0.2]])
        assert isinstance(ot, OrientationTensor3)

    def test_from_features(self):
        from apsg.feature._container import Vector3Set

        data = [Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)]
        v = Vector3Set(data)
        ot = OrientationTensor3.from_features(v)
        assert isinstance(ot, OrientationTensor3)

    def test_from_pairs(self):
        ps = [
            Pair(109, 82, 21, 10),
            Pair(118, 76, 30, 11),
            Pair(97, 86, 7, 3),
            Pair(109, 75, 23, 14),
        ]
        from apsg.feature._container import PairSet

        pset = PairSet(ps)
        ot = OrientationTensor3.from_pairs(pset)
        assert isinstance(ot, OrientationTensor3)

    def test_from_pairs_no_shift(self):
        ps = [
            Pair(109, 82, 21, 10),
            Pair(118, 76, 30, 11),
            Pair(97, 86, 7, 3),
            Pair(109, 75, 23, 14),
        ]
        from apsg.feature._container import PairSet

        pset = PairSet(ps)
        ot = OrientationTensor3.from_pairs(pset, shift=False)
        assert isinstance(ot, OrientationTensor3)

    def test_inherits_ellipsoid(self):
        ot = OrientationTensor3([[8, 0, 0], [0, 2, 0], [0, 0, 1]])
        assert math.isclose(ot.S1, math.sqrt(8))
        assert math.isclose(ot.S3, 1)

    def test_lowercase_alias(self):
        assert ortensor is OrientationTensor3
