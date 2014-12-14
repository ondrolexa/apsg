#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_apsg
----------------------------------

Tests for `apsg` module.
"""

import unittest

import numpy as np
from apsg import Vec3, Fol, Lin, Group


class TestApsg(unittest.TestCase):

    def setUp(self):
        pass

    def test_lin2vec2lin(self):
        self.assertTrue(Vec3(Lin(110, 37)).aslin == Lin(110, 37))

    def test_fol2vec2fol(self):
        self.assertTrue(Vec3(Fol(213, 52)).asfol == Fol(213, 52))

    def test_rotation_invariant(self):
        g = Group.randn_lin()
        self.assertTrue(np.allclose(g.rotate(Lin(45, 45), 90).rdegree, g.rdegree))

    def test_resultant_rdegree(self):
        g = Group.from_array([45, 135, 225, 315], [45, 45, 45, 45])
        c1 = g.resultant.uv == Lin(0, 90)
        c2 = np.allclose(abs(g.resultant), np.sqrt(8))
        c3 = np.allclose((g.rdegree/100 + 1)**2, 2)
        self.assertTrue(c1 and c2 and c3)

    def test_cross_product(self):
        l1 = Lin(110, 22)
        l2 = Lin(163, 47)
        p = l1**l2
        self.assertTrue(np.allclose(p.angle(l1), p.angle(l2), 90))

    def test_axial_addition(self):
        m = Lin(135, 10) + Lin(315, 10)
        self.assertTrue(m.uv == Lin(135, 0))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
