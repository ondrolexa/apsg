# -*- coding: utf-8 -*-


from __future__ import division, print_function
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from .core import Vec3, Fol, Lin, Group
from .plotting import StereoNet
from .helpers import sind, cosd, eformat


__all__ = ["Core"]


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
        self.info = kwargs.get("info", "Default")
        self.name = kwargs.get("name", "Default")
        self.filename = kwargs.get("filename", None)
        self.alpha = kwargs.get("alpha", 0)
        self.beta = kwargs.get("beta", 0)
        self.strike = kwargs.get("strike", 90)
        self.dip = kwargs.get("dip", 0)
        self.volume = kwargs.get("volume", "Default")
        self.date = kwargs.get("date", datetime.now())
        self.steps = kwargs.get("steps", [])
        self.a95 = kwargs.get("a95", [])
        self.comments = kwargs.get("comments", [])
        self._vectors = kwargs.get("vectors", [])

    def __repr__(self):
        return "Core:" + str(self.name)

    def __getitem__(self, key):
        """Group fancy indexing"""
        if isinstance(key, int):
            key = self.steps[key]
        if isinstance(key, str):
            if key in self.steps:
                ix = self.steps.index(key)
                return dict(
                    step=key,
                    V=self._vectors[ix],
                    MAG=self.MAG[ix],
                    geo=self.geo[ix],
                    strata=self.strata[ix],
                    a95=self.a95[ix],
                    comment=self.comments[ix],
                )
            else:
                raise (Exception("Key {} not found.".format(key)))
        else:
            raise (Exception("Key of {} not supported.".format(type(key))))

    @classmethod
    def from_pmd(cls, filename):
        """Return ``Core`` instance generated from PMD file.

        Args:
          filename: PMD file

        Example:
          >>> d = Core.from_pmd('K509A2-1.PMD')

        """
        with open(filename, encoding="latin1") as f:
            d = f.read().splitlines()
        data = {}
        fields = {
            "Xc": slice(5, 14),
            "Yc": slice(15, 24),
            "Zc": slice(25, 34),
            "a95": slice(69, 73),
        }
        data["info"] = d[0].strip()
        vline = d[1].strip()
        data["filename"] = filename
        data["name"] = vline[:10].strip()
        data["alpha"] = float(vline[10:20].strip().split("=")[1])
        data["beta"] = float(vline[20:30].strip().split("=")[1])
        data["strike"] = float(vline[30:40].strip().split("=")[1])
        data["dip"] = float(vline[40:50].strip().split("=")[1])
        data["volume"] = float(vline[50:63].strip().split("=")[1].strip("m3"))
        data["date"] = datetime.strptime(vline[63:].strip(), "%m-%d-%Y %H:%M")
        data["steps"] = [ln[:4].strip() for ln in d[3:-1]]
        data["comments"] = [ln[73:].strip() for ln in d[3:-1]]
        data["a95"] = [float(ln[fields["a95"]].strip()) for ln in d[3:-1]]
        data["vectors"] = []
        for ln in d[3:-1]:
            x = float(ln[fields["Xc"]].strip())
            y = float(ln[fields["Yc"]].strip())
            z = float(ln[fields["Zc"]].strip())
            data["vectors"].append(Vec3((x, y, z)))
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
        infoln = "{:<8}  a={:5.1f}   b={:5.1f}   s={:5.1f}   d={:5.1f}   v={}m3  {}"
        ln0 = infoln.format(
            ff, self.alpha, self.beta, *self.bedding.rhr, eformat(self.volume, 2), dt
        )
        headln = (
            "STEP  Xc [Am2]  Yc [Am2]  Zc [Am2]  MAG[A/m]   Dg    Ig    Ds    Is  a95 "
        )
        with open(filename, "w") as pmdfile:
            print(self.info, file=pmdfile, end="\r\n")
            print(ln0, file=pmdfile, end="\r\n")
            print(headln, file=pmdfile, end="\r\n")
            for ln in self.datatable:
                print(ln, file=pmdfile, end="\r\n")
            pmdfile.write(chr(26))

    @property
    def datatable(self):
        tb = []
        for step, V, MAG, geo, strata, a95, comments in zip(
            self.steps,
            self._vectors,
            self.MAG,
            self.geo,
            self.strata,
            self.a95,
            self.comments,
        ):
            ln = "{:<4} {: 9.2E} {: 9.2E} {: 9.2E} {: 9.2E} {:5.1f} {:5.1f} {:5.1f} {:5.1f} {:4.1f} {}".format(
                step, *V, MAG, *geo.dd, *strata.dd, a95, comments
            )
            tb.append(ln)
        return tb

    def show(self):
        ff = os.path.splitext(os.path.basename(self.filename))[0][:8]
        dt = self.date.strftime("%m-%d-%Y %H:%M")
        print(
            "{:<8}  α={:5.1f}   ß={:5.1f}   s={:5.1f}   d={:5.1f}   v=m3  {}".format(
                ff,
                self.alpha,
                self.beta,
                *self.bedding.rhr,
                eformat(self.volume, 2),
                dt
            )
        )
        print(
            "STEP  Xc [Am²]  Yc [Am²]  Zc [Am²]  MAG[A/m]   Dg    Ig    Ds    Is  a95 "
        )
        print("\n".join(self.datatable))

    @property
    def MAG(self):
        return np.array([abs(v) / self.volume for v in self._vectors])

    @property
    def nsteps(self):
        pp = [re.findall("\d+", s) for s in self.steps]
        return np.array([int(s[0]) if s else 0 for s in pp])

    @property
    def V(self):
        "Returns `Group` of vectors in sample (or core) coordinates system"
        return Group([v / self.volume for v in self._vectors], name=self.name)

    @property
    def geo(self):
        "Returns `Group` of vectors in in-situ coordinates system"
        return self.V.rotate(Lin(0, 90), self.alpha).rotate(
            Lin(self.alpha + 90, 0), self.beta
        )

    @property
    def strata(self):
        "Returns `Group` of vectors in tilt‐corrected coordinates system"
        return self.geo.rotate(Lin(self.strike, 0), -self.dip)

    @property
    def bedding(self):
        return Fol(self.strike + 90, self.dip)

    def zijderveld_plot(self):
        N, E, Z = np.array(self.geo).T
        N0, E0, Z0 = self.geo[0]
        fig, ax = plt.subplots(facecolor="white")
        ax.plot(E0, N0, "b+", markersize=14)
        ax.plot(E, N, "bo-", label="Horizontal")
        ax.plot(E0, -Z0, "g+", markersize=14)
        ax.plot(E, -Z, "go-", label="Vertical")
        mx = np.max(np.abs(ax.axis()))
        ax.axis([-mx, mx, -mx, mx])
        ax.set_aspect(1)
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.spines["top"].set_position("zero")
        t = ax.xaxis.get_ticklocs()
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        # t = ax.xaxis.get_ticklocs()
        # ax.xaxis.set_ticks(t[t != 0])
        # t = ax.yaxis.get_ticklocs()
        # ax.yaxis.set_ticks(t[t != 0])
        ax.set_title(self.name, loc="left")
        plt.legend(title="Unit={:g}A/m".format(t[1] - t[0]))
        plt.tight_layout()
        plt.show()

    def demag_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.nsteps[0], self.MAG[0] / self.MAG.max(), "k+", markersize=14)
        ax.plot(self.nsteps, self.MAG / self.MAG.max(), "ko-")
        ax.set_ylabel("M/Mmax")
        ax.set_title("{} (Mmax = {:g})".format(self.name, self.MAG.max()))
        ax.set_ylim(0, 1.02)
        ax.yaxis.grid()
        plt.show()

    def stereo_plot(self, **kwargs):
        title = kwargs.pop("title", self.name)
        s = StereoNet(title=title, **kwargs)
        for f1, f2 in zip(self.geo[:-1], self.geo[1:]):
            s.arc(f1, f2, "k:")
        s.vector(self.geo[0], "k+", markersize=14)
        s.vector(self.geo, "ko")
        s.show()
