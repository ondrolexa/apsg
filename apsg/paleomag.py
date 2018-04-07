# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import numpy as np
from datetime import datetime
from .core import Vec3, Fol, Lin, Group
from .helpers import sind, cosd, eformat

__all__ = ['Core']


class Core(object):
    """``Core`` store palemomagnetic analysis data

    Keyword Args:
      info:
      name:
      filename:
      alpha:
      beta:
      strike:
      dip:
      volume:
      date:
      steps:
      a95:
      comments:
      vectors:

    Returns:
      ``Core`` object instance

    """
    def __init__(self, **kwargs):
        self.info = kwargs.get('info', 'Default')
        self.name = kwargs.get('name', 'Default')
        self.filename = kwargs.get('filename', None)
        self.alpha = kwargs.get('alpha', 0)
        self.beta = kwargs.get('beta', 0)
        self.strike = kwargs.get('strike', 90)
        self.dip = kwargs.get('dip', 0)
        self.volume = kwargs.get('volume', 'Default')
        self.date = kwargs.get('date', datetime.now())
        self.steps = kwargs.get('steps', [])
        self.a95 = kwargs.get('a95', [])
        self.comments = kwargs.get('comments', [])
        self._vectors = kwargs.get('vectors', [])

    def __repr__(self):
        return 'Core:' + str(self.name)

    @classmethod
    def from_pmd(cls, filename):
        """Return ``Core`` instance generated from PMD file.

        Args:
          filename: PMD file

        Example:
          >>> d = Core.from_pmd('K509A2-1.PMD')

        """
        with open(filename, encoding='latin1') as f:
            d = f.read().splitlines()
        data = {}
        fields = {'Xc': slice(5, 14), 'Yc': slice(15, 24), 'Zc': slice(25, 34),
                  'a95': slice(69, 73)}
        data['info'] = d[0].strip()
        vline = d[1].strip()
        data['filename'] = filename
        data['name'] = vline[:10].strip()
        data['alpha'] = float(vline[10:20].strip().split('=')[1])
        data['beta'] = float(vline[20:30].strip().split('=')[1])
        data['strike'] = float(vline[30:40].strip().split('=')[1])
        data['dip'] = float(vline[40:50].strip().split('=')[1])
        data['volume'] = float(vline[50:63].strip().split('=')[1].strip('m3'))
        data['date'] = datetime.strptime(vline[63:].strip(), '%m-%d-%Y %H:%M')
        data['steps'] = [ln[:4].strip() for ln in d[3:-1]]
        data['comments'] = [ln[73:].strip() for ln in d[3:-1]]
        data['a95'] = [float(ln[fields['a95']].strip()) for ln in d[3:-1]]
        data['vectors'] = []
        for ln in d[3:-1]:
            x = float(ln[fields['Xc']].strip())
            y = float(ln[fields['Yc']].strip())
            z = float(ln[fields['Zc']].strip())
            data['vectors'].append(Vec3((x, y, z)))
        return cls(**data)

    def write_pmd(self, filename=None):
        """Save ``Core`` instance to PMD file.

        Args:
          filename: PMD file

        Example:
          >>> d.write_pmd(filename='K509A2-1.PMD')

        """
        if filename is None:
            filename = self.filename
        ff = os.path.splitext(os.path.basename(filename))[0][:8]
        dt = self.date.strftime("%m-%d-%Y %H:%M")
        infoln = '{:<8}  a={:5.1f}   b={:5.1f}   s={:5.1f}   d={:5.1f}   v={}m3  {}'
        ln0 = infoln.format(ff, self.alpha, self.beta, *self.bedding.rhr, eformat(self.volume, 2), dt)
        headln = 'STEP  Xc [Am2]  Yc [Am2]  Zc [Am2]  MAG[A/m]   Dg    Ig    Ds    Is  a95 '
        with open(filename, 'w') as pmdfile:
            print(self.info, file=pmdfile, end='\r\n')
            print(ln0, file=pmdfile, end='\r\n')
            print(headln, file=pmdfile, end='\r\n')
            for ln in self.datatable:
                print(ln, file=pmdfile, end='\r\n')
            pmdfile.write(chr(26))

    @property
    def datatable(self):
        tb = []
        for step, V, MAG, geo, strata, a95, comments in zip(self.steps, self.V, self.MAG, self.geo, self.strata, self.a95, self.comments):
            ln = '{:<4} {: 9.2E} {: 9.2E} {: 9.2E} {: 9.2E} {:5.1f} {:5.1f} {:5.1f} {:5.1f} {:4.1f} {}'.format(step, *V, MAG, *geo.dd, *strata.dd, a95, comments)
            tb.append(ln)
        return tb

    def show(self):
        ff = os.path.splitext(os.path.basename(self.filename))[0][:8]
        dt = self.date.strftime("%m-%d-%Y %H:%M")
        print('{:<8}  α={:5.1f}   ß={:5.1f}   s={:5.1f}   d={:5.1f}   v=m3  {}'.format(ff, self.alpha, self.beta, *self.bedding.rhr, eformat(self.volume, 2), dt))
        print('STEP  Xc [Am²]  Yc [Am²]  Zc [Am²]  MAG[A/m]   Dg    Ig    Ds    Is  a95 ')
        print('\n'.join(self.datatable))

    @property
    def MAG(self):
        return [abs(v) / self.volume for v in self._vectors]

    @property
    def V(self):
        "Returns `Group` of vectors in sample (or core) coordinates system"
        return Group(self._vectors, name=self.name)

    @property
    def geo(self):
        "Returns `Group` of vectors in in-situ coordinates system"
        return self.V.rotate(Lin(0, 90), self.alpha).rotate(Lin(self.alpha + 90, 0), self.beta)

    @property
    def strata(self):
        "Returns `Group` of vectors in tilt‐corrected coordinates system"
        return self.geo.rotate(Lin(self.strike, 0), -self.dip)

    @property
    def bedding(self):
        return Fol(self.strike + 90, self.dip)
