# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import mplstereonet

from .core import Vec3, Fol, Lin, Group, Ortensor

class StereoNet(object):
    """API to mplstereonet"""
    def __init__(self, *args, **kwargs):
        _, self._ax = mplstereonet.subplots(*args, **kwargs)
        self._grid_state = False
        self._cax = None
        self._lgd = None

    def draw(self):
        h,l = self._ax.get_legend_handles_labels()
        if h:
            self._lgd = self._ax.legend(h, l, bbox_to_anchor=(1.12, 1), loc=2, borderaxespad=0., numpoints=1, scatterpoints=1)
            plt.subplots_adjust(right=0.75)
        else:
            plt.subplots_adjust(right=0.9)
        plt.draw()

    def cla(self):
        self._ax.cla()
        self._ax.grid(self._grid_state)
        self._cax = None
        self._lgd = None
        self.draw()

    def grid(self, state=True):
        self._ax.grid(state)
        self._grid_state = state
        self.draw()

    def plane(self, obj, *args, **kwargs):
        assert obj.type is Fol, 'Only Fol instance could be plotted as plane.'
        strike, dip = obj.rhr
        self._ax.plane(strike, dip, *args, **kwargs)
        self.draw()

    def pole(self, obj, *args, **kwargs):
        assert obj.type is Fol, 'Only Fol instance could be plotted as pole.'
        strike, dip = obj.rhr
        self._ax.pole(strike, dip, *args, **kwargs)
        self.draw()

    def rake(self, obj, rake_angle, *args, **kwargs):
        assert obj.type is Fol, 'Only Fol instance could be used with rake.'
        strike, dip = obj.rhr
        self._ax.rake(strike, dip, rake_angle, *args, **kwargs)
        self.draw()

    def line(self, obj, *args, **kwargs):
        assert obj.type is Lin, 'Only Lin instance could be plotted as line.'
        bearing, plunge = obj.dd
        self._ax.line(plunge, bearing, *args, **kwargs)
        self.draw()

    def cone(self, obj, angle, segments=100, bidirectional=True, **kwargs):
        assert obj.type is Lin, 'Only Lin instance could be used as cone axis.'
        bearing, plunge = obj.dd
        self._ax.cone(plunge, bearing, angle, segments=segments,
                      bidirectional=bidirectional, **kwargs)
        self.draw()

    def density_contour(self, group, *args, **kwargs):
        assert type(group) is Group, 'Only group of data could be used for contouring.'
        if group.type is Lin:
            bearings, plunges = group.dd
            kwargs['measurement'] = 'lines'
            self._cax = self._ax.density_contour(plunges, bearings, *args, **kwargs)
            plt.draw()
        elif group.type is Fol:
            strikes, dips = group.rhr
            kwargs['measurement'] = 'poles'
            self._cax = self._ax.density_contour(strikes, dips, *args, **kwargs)
            plt.draw()
        else:
            raise 'Only Fol or Lin group is allowed.'

    def density_contourf(self, group, *args, **kwargs):
        assert type(group) is Group, 'Only group of data could be used for contouring.'
        if group.type is Lin:
            bearings, plunges = group.dd
            kwargs['measurement'] = 'lines'
            self._cax = self._ax.density_contourf(plunges, bearings, *args, **kwargs)
            plt.draw()
        elif group.type is Fol:
            strikes, dips = group.rhr
            kwargs['measurement'] = 'poles'
            self._cax = self._ax.density_contourf(strikes, dips, *args, **kwargs)
            plt.draw()
        else:
            raise 'Only Fol or Lin group is allowed.'

    def colorbar(self):
        if self._cax is not None:
            cbaxes = self._ax.figure.add_axes([0.015, 0.2, 0.02, 0.6]) 
            cb = plt.colorbar(self._cax, cax = cbaxes) 

    def savefig(self, filename='stereonet.pdf'):
        if self._lgd is None:
            self._ax.figure.savefig(filename)
        else:
            self._ax.figure.savefig(filename, bbox_extra_artists=(self._lgd,), bbox_inches='tight')

    def show(self):
        plt.show()


def rose(a, bins=13, **kwargs):
    """Plot rose diagram"""
    if isinstance(a, Group):
        a, _ = a.dd
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    arad = a * np.pi / 180
    erad = np.linspace(0, 360, bins) * np.pi / 180
    plt.hist(arad, bins=erad, **kwargs)

