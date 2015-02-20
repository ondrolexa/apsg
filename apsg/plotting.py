# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import mplstereonet
except ImportError:
    pass

from .core import Fol, Lin, Fault, Group
from .helpers import cosd, sind, l2v, p2v, getldd, getfdd, l2xy, v2l, rodrigues


class StereoNetJK(object):
    """API to Joe Kington mplstereonet"""
    def __init__(self, *args, **kwargs):
        _, self._ax = mplstereonet.subplots(*args, **kwargs)
        self._grid_state = False
        self._cax = None
        self._lgd = None

    def draw(self):
        h, l = self._ax.get_legend_handles_labels()
        if h:
            self._lgd = self._ax.legend(h, l, bbox_to_anchor=(1.12, 1),
                                        loc=2, borderaxespad=0.,
                                        numpoints=1, scatterpoints=1)
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

    def arrow(self, obj, sense, *args, **kwargs):
        assert obj.type is Lin, 'Only Lin instance could be plotted as quiver.'
        bearing, plunge = obj.dd
        xx, yy = mplstereonet.line(plunge, bearing)
        xx1, yy1 = mplstereonet.line(plunge - 5, bearing)
        for x, y, x1, y1 in zip(xx, yy, xx1, yy1):
            self._ax.arrow(x, y, sense*(x1-x), sense*(y1-y))
        self.draw()

    def cone(self, obj, angle, segments=100, bidirectional=True, **kwargs):
        assert obj.type is Lin, 'Only Lin instance could be used as cone axis.'
        bearing, plunge = obj.dd
        self._ax.cone(plunge, bearing, angle, segments=segments,
                      bidirectional=bidirectional, **kwargs)
        self.draw()

    def density_contour(self, group, *args, **kwargs):
        assert type(group) is Group, 'Only group could be used for contouring.'
        if group.type is Lin:
            bearings, plunges = group.dd
            kwargs['measurement'] = 'lines'
            self._cax = self._ax.density_contour(plunges, bearings,
                                                 *args, **kwargs)
            plt.draw()
        elif group.type is Fol:
            strikes, dips = group.rhr
            kwargs['measurement'] = 'poles'
            self._cax = self._ax.density_contour(strikes, dips,
                                                 *args, **kwargs)
            plt.draw()
        else:
            raise 'Only Fol or Lin group is allowed.'

    def density_contourf(self, group, *args, **kwargs):
        assert type(group) is Group, 'Only group could be used for contouring.'
        if group.type is Lin:
            bearings, plunges = group.dd
            kwargs['measurement'] = 'lines'
            self._cax = self._ax.density_contourf(plunges, bearings,
                                                  *args, **kwargs)
            plt.draw()
        elif group.type is Fol:
            strikes, dips = group.rhr
            kwargs['measurement'] = 'poles'
            self._cax = self._ax.density_contourf(strikes, dips,
                                                  *args, **kwargs)
            plt.draw()
        else:
            raise 'Only Fol or Lin group is allowed.'

    def colorbar(self):
        if self._cax is not None:
            cbaxes = self._ax.figure.add_axes([0.015, 0.2, 0.02, 0.6])
            plt.colorbar(self._cax, cax=cbaxes)

    def savefig(self, filename='stereonet.pdf'):
        if self._lgd is None:
            self._ax.figure.savefig(filename)
        else:
            self._ax.figure.savefig(filename, bbox_extra_artists=(self._lgd,),
                                    bbox_inches='tight')

    def show(self):
        plt.show()


class StereoNet(object):
    """StereoNet class for Schmidt net lower hemisphere plotting"""
    def __init__(self, *args, **kwargs):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.format_coord = self.format_coord
        self.grid = kwargs.get('grid', True)
        self.grid_style = kwargs.get('grid_style', 'k:')
        self._lgd = None
        self.cla()
        # optionally immidiately plot passed objects
        if args:
            for n, arg in enumerate(args):
                if type(arg) is Group:
                    typ = arg.type
                    cnt = '({:d})'.format(len(arg))
                else:
                    typ = type(arg)
                    cnt = ':{:.0f}/{:.0f}'.format(*arg.dd)
                if typ is Lin:
                    self.line(arg, label='{:2d}-L'.format(n + 1) + cnt)
                if typ is Fol:
                    self.plane(arg, label='{:2d}-S'.format(n + 1) + cnt)
            self.show()

    def draw(self):
        h, l = self.ax.get_legend_handles_labels()
        if h:
            self._lgd = self.ax.legend(h, l, bbox_to_anchor=(1.12, 1),
                                       prop={'size': 11}, loc=2,
                                       borderaxespad=0, scatterpoints=1,
                                       numpoints=1)
            plt.subplots_adjust(right=0.75)
        else:
            plt.subplots_adjust(right=0.9)
        plt.draw()

    def cla(self):
        """Clear projection"""
        # now ok
        self.ax.cla()
        self.ax.set_aspect('equal')
        self.ax.set_autoscale_on(False)
        self.ax.axis([-1.05, 1.05, -1.05, 1.05])
        self.ax.set_axis_off()

        # Projection circle
        self.ax.text(0, 1.02, 'N', ha='center', va='baseline', fontsize=16)
        self.ax.add_artist(plt.Circle((0, 0), 1,
                           color='w', zorder=0))
        self.ax.add_artist(plt.Circle((0, 0), 1,
                           color='None', ec='k', zorder=3))

        if self.grid:
            # Main cross
            self.ax.plot([-1, 1, np.nan, 0, 0],
                         [0, 0, np.nan, -1, 1],
                         self.grid_style, zorder=3)
            # Latitudes
            lat = lambda a, phi: self._cone(l2v(a, 0), l2v(a, phi),
                                            limit=89.9999, res=91)
            lat_n = np.array([lat(0, phi) for phi in range(10, 90, 10)])
            self.ax.plot(lat_n[:, 0, :].T, lat_n[:, 1, :].T,
                         self.grid_style, zorder=3)
            lat_s = np.array([lat(180, phi) for phi in range(10, 90, 10)])
            self.ax.plot(lat_s[:, 0, :].T, lat_s[:, 1, :].T,
                         self.grid_style, zorder=3)
            # Longitudes
            lon = lambda a, theta: self._cone(p2v(a, theta), l2v(a, theta),
                                              limit=80, res=91)
            lon_e = np.array([lon(90, theta) for theta in range(10, 90, 10)])
            self.ax.plot(lon_e[:, 0, :].T, lon_e[:, 1, :].T,
                         self.grid_style, zorder=3)
            lon_w = np.array([lon(270, theta) for theta in range(10, 90, 10)])
            self.ax.plot(lon_w[:, 0, :].T, lon_w[:, 1, :].T,
                         self.grid_style, zorder=3)

        # ticks
        a = np.arange(0, 360, 30)
        tt = np.array([0.98, 1])
        x = np.outer(tt, sind(a))
        y = np.outer(tt, cosd(a))
        self.ax.plot(x, y, 'k', zorder=4)
        # Middle cross
        self.ax.plot([-0.02, 0.02, np.nan, 0, 0],
                     [0, 0, np.nan, -0.02, 0.02],
                     'k', zorder=4)
        self.draw()

    def getlin(self):
        """get Lin by mouse click"""
        x, y = plt.ginput(1)[0]
        return Lin(*getldd(x, y))

    def getfol(self):
        """get Fol by mouse click"""
        x, y = plt.ginput(1)[0]
        return Fol(*getfdd(x, y))

    def getlins(self):
        """get Group of Lin by mouse"""
        pts = plt.ginput(0, mouse_add=1, mouse_pop=2, mouse_stop=3)
        return Group([Lin(*getldd(x, y)) for x, y in pts])

    def getfols(self):
        """get Group of Fol by mouse"""
        pts = plt.ginput(0, mouse_add=1, mouse_pop=2, mouse_stop=3)
        return Group([Fol(*getldd(x, y)) for x, y in pts])

    def _cone(self, axis, vector, limit=180, res=361, split=False):
        a = np.linspace(-limit, limit, res)
        x, y = l2xy(*v2l(rodrigues(axis, vector, a)))
        if split:
            dist = np.hypot(np.diff(x), np.diff(y))
            ix = np.nonzero(dist > 1)[0]
            x = np.insert(x, ix + 1, np.nan)
            y = np.insert(y, ix + 1, np.nan)
        return x, y

    def _arrow(self, pos_lin,  dir_lin=None, sense=1):
        x, y = l2xy(*pos_lin.dd)
        if dir_lin is None:
            dx, dy = -x, -y
        else:
            ax, ay = l2xy(*dir_lin.dd)
            dx, dy = -ax, -ay
        mag = np.hypot(dx, dy)
        u, v = sense * dx / mag, sense * dy / mag
        self.ax.quiver(x, y, u, v, width=1, headwidth=4, units='dots')

    def plane(self, obj, *args, **kwargs):
        assert obj.type is Fol, 'Only Fol instance could be plotted as plane.'
        if isinstance(obj, Group):
            x = []
            y = []
            for azi, inc in obj.dd.T:
                xx, yy = self._cone(p2v(azi, inc), l2v(azi, inc),
                                    limit=89.9999, res=cosd(inc)*179+2)
                x = np.hstack((x, xx, np.nan))
                y = np.hstack((y, yy, np.nan))
            x = x[:-1]
            y = y[:-1]
        else:
            azi, inc = obj.dd
            x, y = self._cone(p2v(azi, inc), l2v(azi, inc),
                              limit=89.9999, res=cosd(inc)*179+2)
        h = self.ax.plot(x, y, *args, **kwargs)
        self.draw()
        return h

    def line(self, obj, *args, **kwargs):
        assert obj.type is Lin, 'Only Lin instance could be plotted as line.'
        # ensure point plot
        if 'ls' not in kwargs and 'linestyle' not in kwargs:
            kwargs['linestyle'] = 'none'
        if not args:
            if 'marker' not in kwargs:
                kwargs['marker'] = 'o'
        x, y = l2xy(*obj.dd)
        h = self.ax.plot(x, y, *args, **kwargs)
        self.draw()
        return h

    def pole(self, obj, *args, **kwargs):
        assert obj.type is Fol, 'Only Fol instance could be plotted as poles.'
        # ensure point plot
        if 'ls' not in kwargs and 'linestyle' not in kwargs:
            kwargs['linestyle'] = 'none'
        if not args:
            if 'marker' not in kwargs:
                kwargs['marker'] = 's'
        x, y = l2xy(*obj.aslin.dd)
        h = self.ax.plot(x, y, *args, **kwargs)
        self.draw()
        return h

    def cone(self, obj, alpha, *args, **kwargs):
        assert obj.type is Lin, 'Only Lin instance could be used as cone axis.'
        if isinstance(obj, Group):
            x = []
            y = []
            for azi, inc in obj.dd.T:
                xx, yy = self._cone(l2v(azi, inc), l2v(azi, inc-alpha),
                                    limit=180, res=sind(alpha)*358+3,
                                    split=True)
                x = np.hstack((x, xx, np.nan))
                y = np.hstack((y, yy, np.nan))
            x = x[:-1]
            y = y[:-1]
        else:
            azi, inc = obj.dd
            x, y = self._cone(l2v(azi, inc), l2v(azi, inc-alpha),
                              limit=180, res=sind(alpha)*358+3, split=True)
        h = self.ax.plot(x, y, *args, **kwargs)
        self.draw()
        return h

    def fault(self, obj, *arg, **kwargs):
        """Plot a fault-and-striae plot"""
        assert obj.type is Fault, 'Only Fault instance could be used.'
        self.plane(obj.fol, *arg, **kwargs)
        self._arrow(obj.lin, sense=obj.sense)
        self.draw()

    def hoeppner(self, obj, *arg, **kwargs):
        """Plot a tangent lineation plot"""
        assert obj.type is Fault, 'Only Fault instance could be used.'
        self._arrow(obj.fvec.aslin, dir_lin=obj.lin, sense=-obj.sense)
        self.draw()

    def contourf(self, obj, *args, **kwargs):
        if 'cmap' not in kwargs and 'colors' not in kwargs:
                kwargs['cmap'] = 'Greys'
        if 'zorder' not in kwargs:
                kwargs['zorder'] = 1
        d = Density(obj, **kwargs)
        cs = self.ax.tricontourf(d.triang, d.density, *args, **kwargs)
        if kwargs.get('legend', False):
            self._add_colorbar(cs)
        self.draw()

    def contour(self, obj, *args, **kwargs):
        if 'cmap' not in kwargs and 'colors' not in kwargs:
                kwargs['cmap'] = 'Greys'
        if 'zorder' not in kwargs:
                kwargs['zorder'] = 1
        d = Density(obj, **kwargs)
        cs = self.ax.tricontour(d.triang, d.density, *args, **kwargs)
        if kwargs.get('legend', False):
            self._add_colorbar(cs)
        self.draw()

    def _add_colorbar(self, cs):
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("left", size="5%", pad=0.5)
        plt.colorbar(cs, cax=cax)
        # modify tick labels
        #cb = plt.colorbar(cs, cax=cax)
        #lbl = [item.get_text()+'S' for item in cb.ax.get_yticklabels()]
        #lbl[lbl.index(next(l for l in lbl if l.startswith('0')))] = 'E'
        #cb.set_ticklabels(lbl)

    def show(self):
        plt.show()

    def savefig(self, filename='apsg_stereonet.pdf'):
        if self._lgd is None:
            self.ax.figure.savefig(filename)
        else:
            self.ax.figure.savefig(filename, bbox_extra_artists=(self._lgd,),
                                   bbox_inches='tight')
        plt.savefig(filename)

    def format_coord(self, x, y):
        if np.hypot(x, y) > 1:
            return ''
        else:
            vals = getfdd(x, y) + getldd(x, y)
            return 'S:{:0>3.0f}/{:0>2.0f} L:{:0>3.0f}/{:0>2.0f}'.format(*vals)


class Density(object):
    """trida Density"""
    def __init__(self, d, **kwargs):
        self.dcdata = np.asarray(d)
        self.calculate(**kwargs)

    def calculate(self, **kwargs):
        import matplotlib.tri as tri
        # parse options
        sigma = kwargs.get('sigma', 3)
        ctn_points = kwargs.get('cnt_points', 180)
        method = kwargs.get('method', 'exponential_kamb')

        func = {'linear_kamb': _linear_inverse_kamb,
                'square_kamb': _square_inverse_kamb,
                'schmidt': _schmidt_count,
                'kamb': _kamb_count,
                'exponential_kamb': _exponential_kamb,
                }[method]

        self.xg = self.yg = 0
        for rho in np.linspace(0, 1, np.round(ctn_points/2/np.pi)):
            theta = np.linspace(0, 360, np.round(ctn_points*rho + 1))[:-1]
            self.xg = np.hstack((self.xg, rho*sind(theta)))
            self.yg = np.hstack((self.yg, rho*cosd(theta)))
        self.dcgrid = l2v(*getldd(self.xg, self.yg)).T
        n = self.dcgrid.shape[0]
        self.density = np.zeros(n, dtype=np.float)
        # weights are given by euclidean norms of data
        weights = np.linalg.norm(self.dcdata, axis=1)
        weights /= weights.mean()
        for i in range(n):
            dist = np.abs(np.dot(self.dcgrid[i], self.dcdata.T))
            count, scale = func(dist, sigma)
            count *= weights
            self.density[i] = (count.sum() - 0.5) / scale
        self.density[self.density < 0] = 0
        self.triang = tri.Triangulation(self.xg, self.yg)

    def plot(self, N=6, cm=plt.cm.jet):
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.tricontourf(self.triang, self.density, N, cm=cm)
        plt.colorbar()
        plt.tricontour(self.triang, self.density, N, colors='k')
        plt.show()

    def plotcountgrid(self):
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.triplot(self.triang, 'bo-')
        plt.show()

#----------------------------------------------------------------
# Following counting routines are from Joe Kington's mplstereonet
# https://github.com/joferkington/mplstereonet


def _kamb_radius(n, sigma):
    """Radius of kernel for Kamb-style smoothing."""
    a = sigma**2 / (float(n) + sigma**2)
    return (1 - a)


def _kamb_units(n, radius):
    """Normalization function for Kamb-style counting."""
    return np.sqrt(n * radius * (1 - radius))


# All of the following kernel functions return an _unsummed_ distribution and
# a normalization factor
def _exponential_kamb(cos_dist, sigma=3):
    """Kernel function from Vollmer for exponential smoothing."""
    n = float(cos_dist.size)
    f = 2 * (1.0 + n / sigma**2)
    count = np.exp(f * (cos_dist - 1))
    units = np.sqrt(n * (f/2.0 - 1) / f**2)
    return count, units


def _linear_inverse_kamb(cos_dist, sigma=3):
    """Kernel function from Vollmer for linear smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, sigma)
    f = 2 / (1 - radius)
    #cos_dist = cos_dist[cos_dist >= radius]
    count = (f * (cos_dist - radius))
    count[cos_dist < radius] = 0
    return count, _kamb_units(n, radius)


def _square_inverse_kamb(cos_dist, sigma=3):
    """Kernel function from Vollemer for inverse square smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, sigma)
    f = 3 / (1 - radius)**2
    #cos_dist = cos_dist[cos_dist >= radius]
    count = (f * (cos_dist - radius)**2)
    count[cos_dist < radius] = 0
    return count, _kamb_units(n, radius)


def _kamb_count(cos_dist, sigma=3):
    """Original Kamb kernel function (raw count within radius)."""
    n = float(cos_dist.size)
    dist = _kamb_radius(n, sigma)
    count = (cos_dist >= dist)
    return count, _kamb_units(n, dist)


def _schmidt_count(cos_dist, sigma=None):
    """Schmidt (a.k.a. 1%) counting kernel function."""
    radius = 0.01
    count = ((1 - cos_dist) <= radius)
    # To offset the count.sum() - 0.5 required for the kamb methods...
    count = 0.5 / count.size + count
    return count, (cos_dist.size * radius)
#------------------------------------------------------------------


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
