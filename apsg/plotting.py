# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import mplstereonet

from .core import Vec3, Fol, Lin, Group, Ortensor
from .helpers import *

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


class SchmidtNet(object):
    """SchmidtNet lower hemisphere class"""
    def __init__(self, **kwargs):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.format_coord = format_coord
        self.grid = kwargs.get('grid', True)
        self.grid_style = kwargs.get('grid_style', 'k:')
        self.cm = kwargs.get('cmap', plt.cm.Greys)
        self.cla()
    
    def cla(self):
        """Clear projection"""
        # now ok
        self.ax.cla()
        self.ax.set_aspect('equal')
        self.ax.set_autoscale_on(False)
        self.ax.axis([-1.05,1.05,-1.05,1.05])
        self.ax.set_axis_off()
        
        # Projection circle
        self.ax.text(0, 1.02, 'N', ha='center', va='baseline', fontsize=16)
        self.ax.add_artist(plt.Circle((0, 0), 1, color='w', ec='k', zorder=0))
        
        if self.grid:
            # Main cross
            self.ax.plot([-1,1,np.nan,0,0], [0,0,np.    nan,-1,1], self.grid_style, zorder=2)
            # Latitudes
            lat_n = np.array([self._cone(l2v(0,0), l2v(0,a), limit=89.9999, res=91) for a in range(10,90,10)])
            self.ax.plot(lat_n[:,0,:].T, lat_n[:,1,:].T, self.grid_style, zorder=2)
            lat_s = np.array([self._cone(l2v(180,0), l2v(180,a), limit=89.9999, res=91) for a in range(10,90,10)])
            self.ax.plot(lat_s[:,0,:].T, lat_s[:,1,:].T, self.grid_style, zorder=2)
            # Longitudes
            lon_e = np.array([self._cone(p2v(90,a), l2v(90,a), limit=80, res=91) for a in range(10,90,10)])
            self.ax.plot(lon_e[:,0,:].T, lon_e[:,1,:].T, self.grid_style, zorder=2)
            lon_w = np.array([self._cone(p2v(270,a), l2v(270,a), limit=80, res=91) for a in range(10,90,10)])
            self.ax.plot(lon_w[:,0,:].T, lon_w[:,1,:].T, self.grid_style, zorder=2)

        # ticks
        a = np.arange(0,360,30)
        tt = np.array([0.98, 1])
        x = np.outer(tt, sind(a))
        y = np.outer(tt, cosd(a))
        self.ax.plot(x, y, 'k', zorder=3)
        # Middle cross
        self.ax.plot([-0.02,0.02,np.nan,0,0], [0,0,np.    nan,-0.02,0.02], 'k', zorder=3)

    def getlin(self):
        """get Lin by mouse click"""
        x,y = plt.ginput(1)[0]
        return Lin(*getldd(x,y))
        
    def getfol(self):
        """get Fol by mouse click"""
        x,y = plt.ginput(1)[0]
        return Fol(*getfdd(x,y))
    
    def getlins(self):
        """vrati Lin Dataset pomoci kliknuti mysi"""
        pts = plt.ginput(0, mouse_add=1, mouse_pop=2, mouse_stop=3)
        return Group([Lin(*getldd(x,y)) for x,y in pts])

    def getfols(self):
        """vrati Fol Dataset pomoci kliknuti mysi"""
        pts = plt.ginput(0, mouse_add=1, mouse_pop=2, mouse_stop=3)
        return Group([Fol(*getldd(x,y)) for x,y in pts])
    
    def set_density(self, density):
        """Nastavi density grid"""
        if type(density)==Density or density==None:
            self.density = density

    def _cone(self, axis, vector, limit=180, res=361):
        a = np.linspace(-limit, limit, res)
        return l2xy(*v2l(rodrigues(axis, vector, a)))

    def plane(self, obj, *args, **kwargs):
        assert obj.type is Fol, 'Only Fol instance could be plotted as plane.'
        if isinstance(obj, Group):
            x = []
            y = []
            for azi,inc in obj.dd.T:
                xx, yy = self._cone(p2v(azi, inc), l2v(azi, inc), limit=89.9999, res=cosd(inc)*179+2)
                x = np.hstack((x, xx, np.nan))
                y = np.hstack((y, yy, np.nan))
            x = x[:-1]
            y = y[:-1]
        else:
            azi, inc = obj.dd
            x, y = self._cone(p2v(azi, inc), l2v(azi, inc), limit=89.9999, res=cosd(inc)*179+2)
        h = self.ax.plot(x, y, *args, **kwargs)
        plt.draw()
        return h

    def line(self, obj, *args, **kwargs):
        assert obj.type is Lin, 'Only Lin instance could be plotted as line.'
        # ensure point plot
        if not args:
            if 'ls' not in kwargs and 'linestyle' not in kwargs:
                kwargs['linestyle'] = 'none'
            if 'marker' not in kwargs:
                kwargs['marker'] = 'o'
        x, y = l2xy(*obj.dd)
        h = self.ax.plot(x, y, *args, **kwargs)
        plt.draw()
        return h


    def cone(self, obj, alpha):
        assert obj.type is Lin, 'Only Lin instance could be used as cone axis.'
        if isinstance(obj, Group):
            x = []
            y = []
            for azi,inc in obj.dd.T:
                xx, yy = self._cone(l2v(azi, inc), l2v(azi, inc-alpha), limit=180, res=sind(alpha)*358+3)    
                x = np.hstack((x, xx, np.nan))
                y = np.hstack((y, yy, np.nan))
            x = x[:-1]
            y = y[:-1]
        else:
            azi, inc = obj.dd
            x, y = self._cone(l2v(azi, inc), l2v(azi, inc-alpha), limit=180, res=sind(alpha)*358+3)    
        h = self.ax.plot(x, y, *args, **kwargs)
        plt.draw()
        return h


    def show(self):
        plt.show()

    def old_show(self):
        """Draw figure"""
        plt.ion()
        # test if closed
        if not plt._pylab_helpers.Gcf.figs.values():
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.format_coord = format_coord
        
        #density grid
        if self.density:
            cs = self.ax.tricontourf(self.density.triang, self.density.density, self.nc, cmap=self.cm, zorder=1)
            self.ax.tricontour(self.density.triang, self.density.density, self.nc, colors='k', zorder=1)
        
        #grid
        if self.grid:
            grds = list(range(10,100,10)) + list(range(-80,0,10))
            a = Lin(0,0)
            for dip in grds:
                l = Lin(0,dip)
                gc = map(l.rotate, 91*[a], np.linspace(-89.99, 89.99, 91))
                x, y = np.array([r.toxy() for r in gc]).T
                self.ax.plot(x,y,'k:')
            for dip in grds:
                a = Fol(90,dip)
                l = Lin(90,dip)
                gc = map(l.rotate, 81*[a], np.linspace(-80, 80, 81))
                x, y = np.array([r.toxy() for r in gc]).T
                self.ax.plot(x,y,'k:')
        
        # init labels
        handles = []
        labels = []
        
        # plot data
        for arg in self.data:
            
            #fol great circle
            dd = arg.getfols()
            if dd:
                for d in dd:
                    l = Lin(*d.getdd())
                    gc = map(l.rotate, 91*[d], np.linspace(-89.99,89.99,91))
                    x, y = np.array([r.toxy() for r in gc]).T
                    h = self.ax.plot(x, y, color=arg.color, zorder=2, **arg.lines)
                handles.append(h[0])
                labels.append('S ' + arg.name)
            
            #lin point
            dd = arg.getlins()
            if dd:
                for d in dd:
                    x, y = d.toxy()
                    h = self.ax.scatter(x, y, color=arg.color, zorder=4, **arg.points)
                handles.append(h)
                labels.append('L ' + arg.name)
            
            #pole point
            dd = arg.getpoles()
            if dd:
                for d in dd:
                    x, y = d.toxy()
                    h = self.ax.scatter(x, y, color=arg.color, zorder=3, **arg.poles)
                handles.append(h)
                labels.append('P ' + arg.name)
        
        # legend
        if handles:
            self.ax.legend(handles, labels, bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0., numpoints=1, scatterpoints=1)
        
        #density grid contours
        if self.density:
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("left", size="5%", pad=0.5)
            cb = plt.colorbar(cs, cax=cax)
            # modify tick labels
            lbl = [item.get_text()+'S' for item in cb.ax.get_yticklabels()]
            lbl[lbl.index(next(l for l in lbl if l.startswith('0')))] = 'E'
            cb.set_ticklabels(lbl)
        
        #finish
        plt.subplots_adjust(left=0.02,bottom=0.05,right=0.78,top=0.95)
        self.fig.canvas.draw()
        plt.show()
        plt.ioff()

    def savefig(self, filename='schmidtnet.pdf'):
        plt.savefig(filename)


class Density(object):
    """trida Density"""
    def __init__(self, d, k=100, npoints=180):
        self.dcdata = np.asarray(d)
        self.calculate(k, npoints)

    def calculate(self,k, npoints=180):
        import matplotlib.tri as tri
        self.xg = 0
        self.yg = 0
        for rho in np.linspace(0,1,np.round(npoints/2/np.pi)):
            theta = np.linspace(0,360,np.round(npoints*rho + 1))[:-1]
            self.xg = np.hstack((self.xg,rho*sind(theta)))
            self.yg = np.hstack((self.yg,rho*cosd(theta)))
        self.dcgrid = np.asarray(getldc(*getldd(self.xg,self.yg)))
        n = len(self.dcdata)
        E = n/k  # some points on periphery are equivalent
        s = np.sqrt((n*(0.5-1/k)/k)) 
        w = np.zeros(len(self.xg))
        for i in range(n):
            w += np.exp(k*(np.abs(np.dot(self.dcdata[i],self.dcgrid))-1))
        self.density = (w-E)/s
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


def format_coord(x, y):
    if np.hypot(x,y) > 1:
        return ''
    else:
        return 'S:{:0>3.0f}/{:0>2.0f}'.format(*getfdd(x,y)) + ' L:{:0>3.0f}/{:0>2.0f}'.format(*getldd(x,y))


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

