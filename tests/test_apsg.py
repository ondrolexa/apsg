#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_apsg
----------------------------------

Tests for `apsg` module.
"""
from apsg import *
import unittest


class TestApsg(unittest.TestCase):

    def setUp(self):
        pass

    def test_lin2vec2lin(self):
        self.assertTrue(Vec3(Lin(110, 37)).aslin == Lin(110, 37))

    def test_fol2vec2fol(self):
        self.assertTrue(Vec3(Fol(213, 52)).asfol == Fol(213, 52))

    def test_rotation_rdegree(self):
        g = Group.randn_lin()
        self.assertTrue(np.allclose(g.rotate(Lin(45, 45), 90).rdegree, g.rdegree))

    def test_rotation_angle_lin(self):
        l1, l2 = Group.randn_lin(2)
        D = DefGrad.from_axis(Lin(45, 45), 60)
        self.assertTrue(np.allclose(l1.angle(l2), l1.transform(D).angle(l2.transform(D))))

    def test_rotation_angle_fol(self):
        f1, f2 = Group.randn_fol(2)
        D = DefGrad.from_axis(Lin(45, 45), 60)
        self.assertTrue(np.allclose(f1.angle(f2), f1.transform(D).angle(f2.transform(D))))

    def test_resultant_rdegree(self):
        g = Group.from_array([45, 135, 225, 315], [45, 45, 45, 45], Lin)
        c1 = g.R.uv == Lin(0, 90)
        c2 = np.allclose(abs(g.R), np.sqrt(8))
        c3 = np.allclose((g.rdegree/100 + 1)**2, 2)
        self.assertTrue(c1 and c2 and c3)

    def test_cross_product(self):
        l1 = Lin(110, 22)
        l2 = Lin(163, 47)
        p = l1**l2
        self.assertTrue(np.allclose(p.angle(l1), p.angle(l2), 90))

    def test_axial_addition(self):
        l1, l2 = Group.randn_lin(2)
        self.assertTrue(l1.transform(l1.H(l2)) == l2)

    def test_vec_H(self):
        m = Lin(135, 10) + Lin(315, 10)
        self.assertTrue(m.uv == Lin(135, 0))

    def test_ortensor_orthogonal(self):
        f = Group.randn_fol(1)[0]
        self.assertTrue(np.allclose(*Ortensor(Group([f.V, f.rake(-45), f.rake(45)])).eigenvals))

    def test_group_heterogenous_error(self):
        with self.assertRaises(Exception) as exc:
            g = Group([Fol(10, 10), Lin(20, 20)])
        self.assertEqual("All data in group must be of same type.", str(exc.exception))

    def test_group_heterogenous_error(self):
        with self.assertRaises(Exception) as exc:
            g = Group([1, 2, 3])
        self.assertEqual("Data must be Fol, Lin or Vec3 type.", str(exc.exception))

    def test_pair_misfit(self):
        n, l = Group.randn_lin(2)
        f = n.asfol
        p = Pair.from_pair(f, f - l.proj(f))
        self.assertTrue(np.allclose(p.misfit, 0))

    def test_pair_rotate(self):
        n, l = Group.randn_lin(2)
        f = n.asfol
        p = Pair.from_pair(f, f - l.proj(f))
        pr = p.rotate(Lin(45, 45), 120)
        self.assertTrue(np.allclose(p.fvec.angle(p.lvec), pr.fvec.angle(pr.lvec), 90))

    def test_lin_vector_dd(self):
        l = Lin(120, 30)
        self.assertTrue(Lin(*l.V.dd) == l)

    def test_fol_vector_dd(self):
        f = Fol(120, 30)
        self.assertTrue(Lin(*f.V.dd).asfol == f)

    def test_fault_rotation_sense(self):
        f = Fault(90, 30, 110, 28, -1)
        self.assertTrue(repr(f.rotate(Lin(220, 10), 60)) == 'F:343/37-301/29 +')

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()

