# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import matplotlib.cbook as mcb

try:
    import mplstereonet
except ImportError:
    pass

from .core import Vec3, Fol, Lin, Fault, Group, FaultSet, Ortensor
from .helpers import cosd, sind, l2v, p2v, getldd, getfdd, l2xy, v2l, rodrigues

__all__ = ['StereoNet', 'Density', 'rose']

# ignore matplotlib warnings
warnings.filterwarnings('ignore', category=mcb.mplDeprecation)


class StereoNet(object):
    """StereoNet class for Schmidt net plotting"""
    def __init__(self, *args, **kwargs):
        self.ticks = kwargs.get('ticks', True)
        self.grid = kwargs.get('grid', True)
        self.ncols = kwargs.get('ncols', 1)
        self.grid_style = kwargs.get('grid_style', 'k:')
        self.fol_plot = kwargs.get('fol_plot', 'plane')
        self._lgd = None
        self.active = 0
        self.fig, self.ax = plt.subplots(ncols=self.ncols)
        self.fig.canvas.set_window_title('StereoNet - schmidt projection')
        self.fig.set_size_inches(8 * self.ncols, 6)
        self._title = self.fig.suptitle(kwargs.get('title', ''))
        self.cla()
        # optionally immidiately plot passed objects
        if args:
            for arg in args:
                if type(arg) in [Group, FaultSet]:
                    typ = arg.type
                else:
                    typ = type(arg)
                if typ is Lin:
                    self.line(arg, label=repr(arg))
                elif typ is Fol:
                    getattr(self, self.fol_plot)(arg, label=repr(arg))
                elif typ is Vec3:
                    self.vector(arg.aslin, label=repr(arg))
                elif typ is Fault:
                    self.fault(arg, label=repr(arg))
                else:
                    raise TypeError('%s argument is not supported!' % typ)
            self.show()

    def close(self):
        plt.close(self.fig)

    @property
    def closed(self):
        return not plt.fignum_exists(self.fig.number)

    def draw(self):
        if self.closed:
            print('The StereoNet figure have been closed. '
                  'Use new() method or create new one.')
        else:
            for ax in self.fig.axes:
                h, l = ax.get_legend_handles_labels()
                if h:
                    self._lgd = ax.legend(h, l, bbox_to_anchor=(1.12, 1),
                                          prop={'size': 11}, loc=2,
                                          borderaxespad=0, scatterpoints=1,
                                          numpoints=1)
                    plt.subplots_adjust(right=0.75)
                else:
                    plt.subplots_adjust(right=0.9)
            plt.draw()
            # plt.pause(0.001)

    def new(self):
        """Re-initialize StereoNet figure"""
        if self.closed:
            self.__init__()

    def cla(self):
        """Clear projection"""
        # now ok
        for ax in self.fig.axes:
            ax.cla()
            ax.format_coord = self.format_coord
            ax.set_aspect('equal')
            ax.set_autoscale_on(False)
            ax.axis([-1.05, 1.05, -1.05, 1.05])
            ax.set_axis_off()

            # Projection circle
            ax.text(0, 1.02, 'N', ha='center', va='baseline', fontsize=16)
            ax.add_artist(plt.Circle((0, 0), 1,
                                     color='w', zorder=0))
            ax.add_artist(plt.Circle((0, 0), 1,
                                     color='None', ec='k', zorder=3))

            if self.grid:
                # Main cross
                ax.plot([-1, 1, np.nan, 0, 0],
                        [0, 0, np.nan, -1, 1],
                        self.grid_style, zorder=3)
                # Latitudes
                lat = lambda a, phi: self._cone(l2v(a, 0), l2v(a, phi),
                                                limit=89.9999, res=91)
                lat_n = np.array([lat(0, phi) for phi in range(10, 90, 10)])
                ax.plot(lat_n[:, 0, :].T, lat_n[:, 1, :].T,
                        self.grid_style, zorder=3)
                lat_s = np.array([lat(180, phi) for phi in range(10, 90, 10)])
                ax.plot(lat_s[:, 0, :].T, lat_s[:, 1, :].T,
                        self.grid_style, zorder=3)
                # Longitudes
                lon = lambda a, theta: self._cone(p2v(a, theta), l2v(a, theta),
                                                  limit=80, res=91)
                le = np.array([lon(90, theta) for theta in range(10, 90, 10)])
                ax.plot(le[:, 0, :].T, le[:, 1, :].T,
                        self.grid_style, zorder=3)
                lw = np.array([lon(270, theta) for theta in range(10, 90, 10)])
                ax.plot(lw[:, 0, :].T, lw[:, 1, :].T,
                        self.grid_style, zorder=3)

            # ticks
            if self.ticks:
                a = np.arange(0, 360, 30)
                tt = np.array([0.98, 1])
                x = np.outer(tt, sind(a))
                y = np.outer(tt, cosd(a))
                ax.plot(x, y, 'k', zorder=4)
            # Middle cross
            ax.plot([-0.02, 0.02, np.nan, 0, 0],
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
        return Group([Fol(*getfdd(x, y)) for x, y in pts])

    def _cone(self, axis, vector, limit=180, res=361, split=False):
        a = np.linspace(-limit, limit, res)
        x, y = l2xy(*v2l(rodrigues(axis, vector, a)))
        if split:
            dist = np.hypot(np.diff(x), np.diff(y))
            ix = np.nonzero(dist > 1)[0]
            x = np.insert(x, ix + 1, np.nan)
            y = np.insert(y, ix + 1, np.nan)
        return x, y

    def _arrow(self, pos_lin, dir_lin=None, sense=1):
        x, y = l2xy(*pos_lin.dd)
        if dir_lin is None:
            dx, dy = -x, -y
        else:
            ax, ay = l2xy(*dir_lin.dd)
            dx, dy = -ax, -ay
        mag = np.hypot(dx, dy)
        u, v = sense * dx / mag, sense * dy / mag
        self.fig.axes[self.active].quiver(x, y, u, v,
                                          width=1, headwidth=4,
                                          units='dots')

    def plane(self, obj, *args, **kwargs):
        assert obj.type is Fol, 'Only Fol instance could be plotted as plane.'
        if isinstance(obj, Group):
            x = []
            y = []
            for azi, inc in obj.dd.T:
                xx, yy = self._cone(p2v(azi, inc), l2v(azi, inc),
                                    limit=89.9999, res=cosd(inc) * 179 + 2)
                x = np.hstack((x, xx, np.nan))
                y = np.hstack((y, yy, np.nan))
            x = x[:-1]
            y = y[:-1]
        else:
            azi, inc = obj.dd
            x, y = self._cone(p2v(azi, inc), l2v(azi, inc),
                              limit=89.9999, res=cosd(inc) * 179 + 2)
        self.fig.axes[self.active].plot(x, y, *args, **kwargs)
        self.draw()
        # return h

    def line(self, obj, *args, **kwargs):
        assert obj.type is Lin, 'Only Lin instance could be plotted as line.'
        # ensure point plot
        if 'ls' not in kwargs and 'linestyle' not in kwargs:
            kwargs['linestyle'] = 'none'
        if not args:
            if 'marker' not in kwargs:
                kwargs['marker'] = 'o'
        x, y = l2xy(*obj.dd)
        self.fig.axes[self.active].plot(x, y, *args, **kwargs)
        self.draw()

    def vector(self, obj, *args, **kwargs):
        """ This mimics plotting on upper and lower hemisphere"""
        assert issubclass(obj.type, Vec3), \
            'Only Vec3-like instance could be plotted as line.'
        # ensure point plot
        if 'ls' not in kwargs and 'linestyle' not in kwargs:
            kwargs['linestyle'] = 'none'
        if not args:
            if 'marker' not in kwargs:
                kwargs['marker'] = 'o'
        if isinstance(obj, Group):
            uh = np.atleast_2d(np.asarray(obj))[:, 2] < 0
            if np.any(~uh):
                x, y = l2xy(*obj[~uh].asvec3.aslin.dd)
                h = self.fig.axes[self.active].plot(x, y, *args, **kwargs)
                kwargs.pop('label', None)
                cc = h[0].get_color()
            else:
                cc = None
            if np.any(uh):
                kwargs['fillstyle'] = 'none'
                x, y = l2xy(*obj[uh].asvec3.aslin.dd)
                h = self.fig.axes[self.active].plot(-x, -y, *args, **kwargs)
                if cc is not None:
                    h[0].set_color(cc)
        else:
            x, y = l2xy(*obj.asvec3.aslin.dd)
            if obj[2] < 0:
                kwargs['fillstyle'] = 'none'
                self.fig.axes[self.active].plot(-x, -y, *args, **kwargs)
            else:
                self.fig.axes[self.active].plot(x, y, *args, **kwargs)
        self.draw()

    def pole(self, obj, *args, **kwargs):
        assert obj.type is Fol, 'Only Fol instance could be plotted as poles.'
        # ensure point plot
        if 'ls' not in kwargs and 'linestyle' not in kwargs:
            kwargs['linestyle'] = 'none'
        if not args:
            if 'marker' not in kwargs:
                kwargs['marker'] = 's'
        x, y = l2xy(*obj.aslin.dd)
        self.fig.axes[self.active].plot(x, y, *args, **kwargs)
        self.draw()

    def cone(self, obj, alpha, *args, **kwargs):
        assert obj.type is Lin, 'Only Lin instance could be used as cone axis.'
        if isinstance(obj, Group):
            x = []
            y = []
            for azi, inc in obj.dd.T:
                xx, yy = self._cone(l2v(azi, inc), l2v(azi, inc - alpha),
                                    limit=180, res=sind(alpha) * 358 + 3,
                                    split=True)
                x = np.hstack((x, xx, np.nan))
                y = np.hstack((y, yy, np.nan))
            x = x[:-1]
            y = y[:-1]
        else:
            azi, inc = obj.dd
            x, y = self._cone(l2v(azi, inc), l2v(azi, inc - alpha),
                              limit=180, res=sind(alpha) * 358 + 3, split=True)
        self.fig.axes[self.active].plot(x, y, *args, **kwargs)
        self.draw()

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
        if isinstance(obj, Density):
            d = obj
        else:
            d = Density(obj, **kwargs)
        cs = self.fig.axes[self.active].tricontourf(d.triang, d.density,
                                                    *args, **kwargs)
        if kwargs.get('legend', False):
            self._add_colorbar(cs)
        self.draw()

    def contour(self, obj, *args, **kwargs):
        if 'cmap' not in kwargs and 'colors' not in kwargs:
                kwargs['cmap'] = 'Greys'
        if 'zorder' not in kwargs:
                kwargs['zorder'] = 1
        if isinstance(obj, Density):
            d = obj
        else:
            d = Density(obj, **kwargs)
        cs = self.fig.axes[self.active].tricontour(d.triang, d.density,
                                                   *args, **kwargs)
        if kwargs.get('legend', False):
            self._add_colorbar(cs)
        self.draw()

    def _add_colorbar(self, cs):
        divider = make_axes_locatable(self.fig.axes[self.active])
        cax = divider.append_axes("left", size="5%", pad=0.5)
        plt.colorbar(cs, cax=cax)
        # modify tick labels
        # cb = plt.colorbar(cs, cax=cax)
        # lbl = [item.get_text()+'S' for item in cb.ax.get_yticklabels()]
        # lbl[lbl.index(next(l for l in lbl if l.startswith('0')))] = 'E'
        # cb.set_ticklabels(lbl)

    def show(self):
        plt.show()

    def savefig(self, filename='apsg_stereonet.pdf', **kwargs):
        if self._lgd is None:
            self.fig.savefig(filename, **kwargs)
        else:
            self.fig.savefig(filename,
                             bbox_extra_artists=(self._lgd, self._title),
                             bbox_inches='tight', **kwargs)

    def format_coord(self, x, y):
        if np.hypot(x, y) > 1:
            return ''
        else:
            vals = getfdd(x, y) + getldd(x, y)
            return 'S:{:0>3.0f}/{:0>2.0f} L:{:0>3.0f}/{:0>2.0f}'.format(*vals)


class FabricPlot(object):
    """FabricPlot class for triangular fabric plot (Vollmer, 1989)"""
    def __init__(self, *args, **kwargs):
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('Vollmer fabric plot')
        self.ticks = kwargs.get('ticks', True)
        self.grid = kwargs.get('grid', True)
        self.grid_style = kwargs.get('grid_style', 'k:')
        self._lgd = None
        self.A = np.asarray(kwargs.get('A', [0, 3**0.5 / 2]))
        self.B = np.asarray(kwargs.get('B', [1, 3**0.5 / 2]))
        self.C = np.asarray(kwargs.get('C', [0.5, 0]))
        self.Ti = np.linalg.inv(np.array([self.A - self.C, self.B - self.C]).T)
        self.cla()
        # optionally immidiately plot passed objects
        if args:
            for arg in args:
                self.plot(arg)
            self.show()

    def close(self):
        plt.close(self.fig)

    @property
    def closed(self):
        return not plt.fignum_exists(self.fig.number)

    def draw(self):
        if self.closed:
            print('The FabricPlot figure have been closed. '
                  'Use new() method or create new one.')
        else:
            h, l = self.ax.get_legend_handles_labels()
            if h:
                self._lgd = self.ax.legend(h, l, prop={'size': 11}, loc=4,
                                           borderaxespad=0, scatterpoints=1,
                                           numpoints=1)
            plt.draw()
            # plt.pause(0.001)

    def new(self):
        """Re-initialize StereoNet figure"""
        if self.closed:
            self.__init__()

    def cla(self):
        """Clear projection"""
        # now ok
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.format_coord = self.format_coord
        self.ax.set_aspect('equal')
        self.ax.set_autoscale_on(False)
        triangle = np.c_[self.A, self.B, self.C, self.A]
        n = 10
        tick_size = 0.2
        margin = 0.05
        self.ax.set_axis_off()
        plt.axis([self.A[0] - margin, self.B[0] + margin,
                  self.C[1] - margin, self.A[1] + margin])

        # Projection triangle
        bg = plt.Polygon([self.A, self.B, self.C], color='w', edgecolor=None)
        self.ax.add_patch(bg)
        self.ax.plot(triangle[0], triangle[1], 'k', lw=2)
        self.ax.text(self.A[0] - 0.02, self.A[1], 'P',
                     ha='right', va='bottom', fontsize=14)
        self.ax.text(self.B[0] + 0.02, self.B[1], 'G',
                     ha='left', va='bottom', fontsize=14)
        self.ax.text(self.C[0], self.C[1] - 0.02, 'R',
                     ha='center', va='top', fontsize=14)

        if self.grid:
            for l in np.arange(0.1, 1, 0.1):
                self.triplot([l, l], [0, 1 - l], [1 - l, 0], 'k:')
                self.triplot([0, 1 - l], [l, l], [1 - l, 0], 'k:')
                self.triplot([0, 1 - l], [1 - l, 0], [l, l], 'k:')

        # ticks
        if self.ticks:
            r = np.linspace(0, 1, n + 1)
            tick = tick_size * (self.B - self.C) / n
            x = self.A[0] * (1 - r) + self.B[0] * r
            x = np.vstack((x, x + tick[0]))
            y = self.A[1] * (1 - r) + self.B[1] * r
            y = np.vstack((y, y + tick[1]))
            self.ax.plot(x, y, 'k', lw=1)
            tick = tick_size * (self.C - self.A) / n
            x = self.B[0] * (1 - r) + self.C[0] * r
            x = np.vstack((x, x + tick[0]))
            y = self.B[1] * (1 - r) + self.C[1] * r
            y = np.vstack((y, y + tick[1]))
            self.ax.plot(x, y, 'k', lw=1)
            tick = tick_size * (self.A - self.B) / n
            x = self.A[0] * (1 - r) + self.C[0] * r
            x = np.vstack((x, x + tick[0]))
            y = self.A[1] * (1 - r) + self.C[1] * r
            y = np.vstack((y, y + tick[1]))
            self.ax.plot(x, y, 'k', lw=1)
        self.ax.set_title('Fabric plot')
        self.draw()

    def triplot(self, a, b, c, *args, **kwargs):
        a = np.asarray(a)
        b = np.asarray(b)
        c = np.asarray(c)
        x = (a * self.A[0] + b * self.B[0] + c * self.C[0]) / (a + b + c)
        y = (a * self.A[1] + b * self.B[1] + c * self.C[1]) / (a + b + c)
        self.ax.plot(x, y, *args, **kwargs)
        self.draw()

    def plot(self, obj, *args, **kwargs):
        if type(obj) is Group:
            obj = obj.ortensor
        if type(obj) is not Ortensor:
            raise TypeError('%s argument is not supported!' % type(obj))
        # ensure point plot
        if 'ls' not in kwargs and 'linestyle' not in kwargs:
            kwargs['linestyle'] = 'none'
        if not args:
            if 'marker' not in kwargs:
                kwargs['marker'] = 'o'
        if 'label' not in kwargs:
            kwargs['label'] = obj.name
        self.triplot(obj.P, obj.G, obj.R, *args, **kwargs)
        self.draw()

    def show(self):
        plt.show()

    def savefig(self, filename='apsg_fabricplot.pdf', **kwargs):
        if self._lgd is None:
            self.ax.figure.savefig(filename, **kwargs)
        else:
            self.ax.figure.savefig(filename, bbox_extra_artists=(self._lgd,),
                                   bbox_inches='tight', **kwargs)

    def format_coord(self, x, y):
        a, b = self.Ti.dot(np.r_[x, y] - self.C)
        c = 1 - a - b
        if a < 0 or b < 0 or c < 0:
            return ''
        else:
            return 'P:{:0.2f} G:{:0.2f} R:{:0.2f}'.format(a, b, c)


class Density(object):
    """Class to store regular grid of values to be contoured on ``StereoNet``.

    ``Density`` object could be calculated from ``Group`` object or by user-
    defined function, which accept unit vector as argument.

    Args:
      g: ``Group`` object of data to be used for desity calculation
         If ommited, zero density grid is returned.

    Kwargs:
      cnt_points: Value specify density of uniform grid [180]
      sigma: sigma for kernels. Default 1
      method: 'exp_kamb', 'linear_kamb', 'square_kamb', 'schmidt', 'kamb'
              Default 'exp_kamb'
      Trim: Set negative density values to zero. Default False

    """
    def __init__(self, d=None, **kwargs):
        self.initgrid(**kwargs)
        if d:
            assert isinstance(d, Group), 'Density need Group as argument'
            self.calculate_density(np.asarray(d), **kwargs)

    def initgrid(self, **kwargs):
        import matplotlib.tri as tri
        # parse options
        ctn_points = kwargs.get('cnt_points', 180)
        # calc grid
        self.xg = 0
        self.yg = 0
        for rho in np.linspace(0, 1, np.round(ctn_points / 2 / np.pi)):
            theta = np.linspace(0, 360, np.round(ctn_points * rho + 1))[:-1]
            self.xg = np.hstack((self.xg, rho * sind(theta)))
            self.yg = np.hstack((self.yg, rho * cosd(theta)))
        self.dcgrid = l2v(*getldd(self.xg, self.yg)).T
        self.n = self.dcgrid.shape[0]
        self.density = np.zeros(self.n, dtype=np.float)
        self.triang = tri.Triangulation(self.xg, self.yg)

    def calculate_density(self, dcdata, **kwargs):
        """Calculate density of elements from ``Group`` object

        """
        # parse options
        sigma = kwargs.get('sigma', 1)
        method = kwargs.get('method', 'exp_kamb')
        trim = kwargs.get('trim', False)

        func = {'linear_kamb': _linear_inverse_kamb,
                'square_kamb': _square_inverse_kamb,
                'schmidt': _schmidt_count,
                'kamb': _kamb_count,
                'exp_kamb': _exponential_kamb,
                }[method]

        # weights are given by euclidean norms of data
        weights = np.linalg.norm(dcdata, axis=1)
        weights /= weights.mean()
        for i in range(self.n):
            dist = np.abs(np.dot(self.dcgrid[i], dcdata.T))
            count, scale = func(dist, sigma)
            count *= weights
            self.density[i] = (count.sum() - 0.5) / scale
        if trim:
            self.density[self.density < 0] = 0

    def apply_func(self, func, **kwargs):
        """Calculate density using function passed as argument.
        Function must accept unit vector as argument and retrun scalar value.

        """
        for i in range(self.n):
            self.density[i] = func(self.dcgrid[i])

    def plot(self, N=6, cm=plt.cm.jet):
        """ Show contoured density."""
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.tricontourf(self.triang, self.density, N, cm=cm)
        plt.colorbar()
        plt.tricontour(self.triang, self.density, N, colors='k')
        plt.show()

    def plotcountgrid(self):
        """ Show counting grid."""
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.triplot(self.triang, 'bo-')
        plt.show()

# ----------------------------------------------------------------
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
    units = np.sqrt(n * (f / 2.0 - 1) / f**2)
    return count, units


def _linear_inverse_kamb(cos_dist, sigma=3):
    """Kernel function from Vollmer for linear smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, sigma)
    f = 2 / (1 - radius)
    # cos_dist = cos_dist[cos_dist >= radius]
    count = (f * (cos_dist - radius))
    count[cos_dist < radius] = 0
    return count, _kamb_units(n, radius)


def _square_inverse_kamb(cos_dist, sigma=3):
    """Kernel function from Vollemer for inverse square smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, sigma)
    f = 3 / (1 - radius)**2
    # cos_dist = cos_dist[cos_dist >= radius]
    count = (f * (cos_dist - radius)**2)
    count[cos_dist < radius] = 0
    return count, _kamb_units(n, radius)


def _kamb_count(cos_dist, sigma=3):
    """Original Kamb kernel function (raw count within radius)."""
    n = float(cos_dist.size)
    dist = _kamb_radius(n, sigma)
    # count = (cos_dist >= dist)
    count = np.array(cos_dist >= dist, dtype=float)
    return count, _kamb_units(n, dist)


def _schmidt_count(cos_dist, sigma=None):
    """Schmidt (a.k.a. 1%) counting kernel function."""
    radius = 0.01
    count = ((1 - cos_dist) <= radius)
    # To offset the count.sum() - 0.5 required for the kamb methods...
    count = 0.5 / count.size + count
    return count, cos_dist.size * radius
# ------------------------------------------------------------------


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
            self._ax.arrow(x, y, sense * (x1 - x), sense * (y1 - y))
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

    def savefig(self, filename='stereonet.pdf', **kwargs):
        if self._lgd is None:
            self._ax.figure.savefig(filename, **kwargs)
        else:
            self._ax.figure.savefig(filename, bbox_extra_artists=(self._lgd,),
                                    bbox_inches='tight', **kwargs)

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
