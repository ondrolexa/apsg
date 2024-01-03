# -*- coding: utf-8 -*-

import os
import re
from copy import deepcopy
from datetime import datetime

import numpy as np

from apsg.helpers._helper import eformat
from apsg.math._vector import Vector3
from apsg.feature._geodata import Lineation, Foliation, Pair
from apsg.feature._container import Vector3Set
from apsg.feature._tensor3 import DeformationGradient3

__all__ = ("Core",)


class Core(object):
    """
    ``Core`` class to store palemomagnetic analysis data

    Keyword Args:
      info:
      specimen:
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
        self.site = kwargs.get("site", "Default")
        self.specimen = kwargs.get("specimen", "Default")
        self.filename = kwargs.get("filename", None)
        self.latitude = kwargs.get("latitude", None)
        self.longitude = kwargs.get("longitude", None)
        self.height = kwargs.get("height", None)
        self.rock = kwargs.get("rock", None)
        self.age = kwargs.get("age", None)
        self.formation = kwargs.get("formation", None)
        self.sref = kwargs.get("sref", Pair(180, 0, 180, 0))
        self.gref = kwargs.get("gref", Pair(180, 0, 180, 0))
        self.bedding = kwargs.get("bedding", Foliation(0, 0))
        self.foldaxis = kwargs.get("foldaxis", Lineation(0, 0))
        self.volume = kwargs.get("volume", 1.0)
        self.date = kwargs.get("date", datetime.now())
        self.steps = kwargs.get("steps", [])
        self.a95 = kwargs.get("a95", [])
        self.comments = kwargs.get("comments", [])
        self._vectors = kwargs.get("vectors", [])
        self.module_units = kwargs.get("module_units", "A/m")
        self.susceptibility_units = kwargs.get("susceptibility_units", "e-06 SI")
        self.demag_units = kwargs.get("demag_units", "°C")

    def __repr__(self):
        return f"Core {self.site} {self.specimen}"

    def __getitem__(self, key):
        if np.issubdtype(type(key), np.integer):
            key = self.steps[key]
        if isinstance(key, slice):
            res = deepcopy(self)
            if key.start:
                start_ok = key.start
            else:
                start_ok = self.nsteps[0]
            if key.stop:
                stop_ok = key.stop
            else:
                stop_ok = self.nsteps[-1]
            ix = (self.nsteps >= start_ok) & (self.nsteps <= stop_ok)
            res.steps = [val for (val, ok) in zip(self.steps, ix) if ok]
            res.a95 = [val for (val, ok) in zip(self.a95, ix) if ok]
            res.comments = [val for (val, ok) in zip(self.comments, ix) if ok]
            res._vectors = [val for (val, ok) in zip(self._vectors, ix) if ok]
            res.name = self.specimen + "({}-{})".format(start_ok, stop_ok)
            return res
        if isinstance(key, str):
            if key in self.steps:
                ix = self.steps.index(key)
                return dict(
                    step=key,
                    MAG=self.MAG[ix],
                    V=self._vectors[ix],
                    geo=self.geo[ix],
                    tilt=self.tilt[ix],
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
        data["specimen"] = vline[:10].strip()
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
            data["vectors"].append(Vector3((x, y, z)))
        return cls(**data)

    def write_pmd(self, filename=None):
        """Save ``Core`` instance to PMD file.

        Args:
          filename: PMD file

        """
        if filename is None:
            filename = self.filename
        ff = os.path.splitext(os.path.basename(filename))[0][:8]
        dt = self.date.strftime("%m-%d-%Y %H:%M")
        infoln = "{:<8}  a={:5.1f}   b={:5.1f}   s={:5.1f}   d={:5.1f}   v={}m3  {}"
        ln0 = infoln.format(
            ff,
            self.gref.lin.geo[0],
            self.gref.lin.geo[1],
            self.bedding.geo[0],
            self.bedding.geo[1],
            eformat(self.volume, 2),
            dt,
        )
        headln = (
            "STEP  Xc [Am2]  Yc [Am2]  Zc [Am2]  MAG[A/m]   Dg    Ig    Ds    Is   a95 "
        )
        with open(filename, "w") as pmdfile:
            print("/".join([self.site, self.name]), file=pmdfile, end="\r\n")
            print(ln0, file=pmdfile, end="\r\n")
            print(headln, file=pmdfile, end="\r\n")
            for ln in self.datatable:
                print(ln, file=pmdfile, end="\r\n")
            pmdfile.write(chr(26))

    @classmethod
    def from_rs3(cls, filename, exclude=["C", "G"]):
        """Return ``Core`` instance generated from PMD file.

        Args:
          filename: Remasoft rs3 file

        Kwargs:
          exclude: Labels to be excluded. Default ['C', 'G']

        """
        with open(filename, encoding="windows-1250") as f:
            d = f.read().splitlines()

        from io import StringIO
        import pandas as pd

        headspec = [
            [0, 9],
            [10, 19],
            [20, 29],
            [30, 40],
            [41, 50],
            [51, 65],
            [66, 70],
            [71, 73],
            [74, 79],
            [80, 85],
            [86, 91],
            [92, 97],
            [98, 103],
            [104, 109],
            [110, 112],
            [113, 115],
            [116, 118],
            [119, 121],
            [122, 126],
        ]
        bodyspec = [
            [0, 2],
            [3, 13],
            [14, 27],
            [28, 33],
            [34, 39],
            [40, 45],
            [46, 51],
            [52, 57],
            [58, 63],
            [64, 69],
            [70, 75],
            [76, 81],
            [82, 95],
            [96, 105],
            [106, 115],
            [116, 126],
        ]

        head = pd.read_fwf(StringIO("\n".join(d[:2])), colspecs=headspec)
        body = pd.read_fwf(StringIO("\n".join(d[2:])), colspecs=bodyspec)

        data = {}
        data["site"] = head["Site"][0] if not pd.isna(head["Site"][0]) else ""
        data["filename"] = filename
        data["specimen"] = head["Name"][0] if not pd.isna(head["Name"][0]) else ""
        data["longitude"] = (
            float(head["Longitude"][0]) if not pd.isna(head["Longitude"][0]) else None
        )
        data["latitude"] = (
            float(head["Latitude"][0]) if not pd.isna(head["Latitude"][0]) else None
        )
        data["height"] = (
            float(head["Height"][0]) if not pd.isna(head["Height"][0]) else None
        )
        data["rock"] = head["Rock"][0] if not pd.isna(head["Rock"][0]) else ""
        data["age"] = head["Age"][0] if not pd.isna(head["Age"][0]) else ""
        data["formation"] = head["Fm"][0] if not pd.isna(head["Fm"][0]) else ""
        data["sref"] = Pair(180, 0, 180, 0)
        data["gref"] = Pair(
            float(head["SDec"][0]),
            float(head["SInc"][0]),
            float(head["SDec"][0]),
            float(head["SInc"][0]),
        )
        data["bedding"] = (
            Foliation(float(head["BDec"][0]), float(head["BInc"][0]))
            if not pd.isna(head["BDec"][0]) and not pd.isna(head["BInc"][0])
            else None
        )
        data["foldaxis"] = (
            Lineation(float(head["FDec"][0]), float(head["FInc"][0]))
            if not pd.isna(head["FDec"][0]) and not pd.isna(head["FInc"][0])
            else None
        )
        data["date"] = datetime.now()
        ix = body.iloc[:, 0].apply(lambda x: x not in exclude)
        data["steps"] = body[ix].iloc[:, 1].astype(int).to_list()
        data["comments"] = body[ix]["Note"].to_list()
        data["a95"] = body[ix]["Prec"].to_list()
        data["vectors"] = []
        for n, r in body[ix].iterrows():
            data["vectors"].append(r.iloc[2] * Vector3(r["Dsp"], r["Isp"]))
        return cls(**data)

    def write_rs3(self, filename=None):
        """Save ``Core`` instance to RS3 file.

        Args:
          filename: RS3 file

        """
        if filename is None:
            filename = self.filename

        head = "Name      Site      Latitude  Longitude  Height    Rock           Age  Fm SDec  SInc  BDec  BInc  FDec  FInc  P1 P2 P3 P4 Note"
        step_lbl = f"Step[{self.demag_units}]"
        module_lbl = f"M[{self.module_units}]"
        susceptibility_lbl = f"K[{self.susceptibility_units}]"
        subhead = f"ID {step_lbl:<10} {module_lbl:>12}   Dsp   Isp   Dge   Ige   Dtc   Itc   Dfc   Ifc   Prec {susceptibility_lbl:>13} Limit1    Limit2    Note      "
        latitude = self.latitude if self.latitude is not None else ""
        longitude = self.longitude if self.longitude is not None else ""
        height = self.height if self.height is not None else ""
        sdec, sinc = (round(self.gref.fol.geo[0]), round(self.gref.fol.geo[1]))
        bdec, binc = (
            (round(self.bedding.geo[0]), round(self.bedding.geo[1]))
            if self.bedding is not None
            else ("", "")
        )
        fdec, finc = (
            (round(self.foldaxis.geo[0]), round(self.foldaxis.geo[1]))
            if self.foldaxis is not None
            else ("", "")
        )
        hline = f"{self.specimen:9} {self.site:9} {latitude:<9} {longitude:<10} {height:<9} {self.rock:14} {self.age:<7} {sdec:<5} {sinc:<5} {bdec:<5} {binc:<5} {fdec:<5} {finc:<5} 12 0  6  0      "
        prefix = "T" if self.demag_units == "°C" else "M"
        with open(filename, "w", encoding="windows-1250") as res3file:
            print(head, file=res3file, end="\r\n")
            print(hline, file=res3file, end="\r\n")
            print(subhead, file=res3file, end="\r\n")

            ids = ["N"] + (len(self.steps) - 1) * [prefix]
            for id, step, MAG, V, geo, tilt, a95, comment in zip(
                ids,
                self.steps,
                self.MAG,
                self.V,
                self.geo,
                self.tilt,
                self.a95,
                self.comments,
            ):
                ln = f"{id:2} {step:<10} {MAG:>13g} {V.geo[0]:>5.1f} {V.geo[1]:> 5.1f} {geo.geo[0]:>5.1f} {geo.geo[1]:> 5.1f} {tilt.geo[0]:>5.1f} {tilt.geo[1]:> 5.1f}             {a95:>5.1f}                                   {comment:10}"
                print(ln, file=res3file, end="\r\n")

    @property
    def datatable(self):
        """Return data list of strings"""
        tb = []
        for step, MAG, V, geo, tilt, a95, comment in zip(
            self.steps,
            self.MAG,
            self.V,
            self.geo,
            self.tilt,
            self.a95,
            self.comments,
        ):
            ln = "{:<4} {: 9.2E} {: 9.2E} {: 9.2E} {: 9.2E} {:5.1f} {:5.1f} {:5.1f} {:5.1f} {:4.1f} {}".format(
                step,
                V.x,
                V.y,
                V.z,
                MAG,
                geo.geo[0],
                geo.geo[1],
                tilt.geo[0],
                tilt.geo[1],
                a95,
                comment,
            )
            tb.append(ln)
        return tb

    def show(self):
        """Show data"""
        print(
            "site:{} specimen:{} file:{}\nbedding:{} volume:{}m3  {}".format(
                self.site,
                self.specimen,
                os.path.basename(self.filename),
                self.bedding,
                eformat(self.volume, 2),
                self.date.strftime("%m-%d-%Y %H:%M"),
            )
        )
        print(
            "STEP  Xc [Am2]  Yc [Am2]  Zc [Am2]  MAG[A/m]  Dge   Ige   Dtc   Itc   a95 "
        )
        print("\n".join(self.datatable))

    @property
    def MAG(self):
        """Returns numpy array of MAG values"""
        return np.array([abs(v) / self.volume for v in self._vectors])

    @property
    def nsteps(self):
        """Returns steps as numpy array of numbers"""
        pp = [re.findall(r"\d+", str(s)) for s in self.steps]
        return np.array([int(s[0]) if s else 0 for s in pp])

    @property
    def V(self):
        """Returns ``Vector3Set`` of vectors in sample (or core) coordinates system"""
        return Vector3Set([v / self.volume for v in self._vectors], name=self.specimen)

    @property
    def geo(self):
        """Returns ``Vector3Set`` of vectors in in-situ coordinates system"""
        H = DeformationGradient3.from_two_pairs(self.sref, self.gref)
        return self.V.transform(H)

    @property
    def tilt(self):
        """Returns ``Vector3Set`` of vectors in tilt‐corrected coordinates system"""
        return self.geo.rotate(
            Lineation(self.bedding.geo[0] - 90, 0), -self.bedding.geo[1]
        )

    def pca(self, kind="geo", origin=False):
        """
        PCA analysis to calculate principal component and MAD

        Keyword Args:
            kind (str): "V", "geo" or "tilt". Default "geo"
            origin (bool): Whether to include origin. Default False
        """
        data = getattr(self, kind)
        if not origin:
            r = data.R(mean=True) / len(data)
            data = Vector3Set([v - r for v in data])
        ot = data.ortensor()
        if ot.shape > 1:
            pca = ot.V1
            if pca.angle(r) > 90:
                pca = -pca
        else:
            pca = ot.V3
        mad = ot.MAD()
        return pca, mad
