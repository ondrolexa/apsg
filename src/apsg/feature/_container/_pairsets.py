"""Sets of pairs (foliation + lineation) and faults."""

import csv
from itertools import combinations
from os.path import basename

import numpy as np

from apsg.config import apsg_conf
from apsg.feature._container._base import (
    FeatureSet,
    _finite_float,
    _per_item,
    _read_csv_columns,
)
from apsg.feature._container._tensorsets import Stress3Set
from apsg.feature._container._vectorsets import (
    FoliationSet,
    LineationSet,
    Vector3Set,
)
from apsg.feature._distributions._frames import (
    lisle_ellipses,
    lisle_moments,
    pair_arrays,
)
from apsg.feature._geodata import (
    Fault,
    Pair,
)
from apsg.feature._tensor3 import (
    OrientationTensor3,
    Rotation3,
    Stress3,
)
from apsg.helpers._helper import check_count
from apsg.math._vector import Vector3


class PairSet(FeatureSet):
    """
    Class to store set of ``Pair`` features.
    """

    __feature_class__ = Pair

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, Pair) for obj in data):
            raise TypeError("Data must be instances of Pair")

    def __repr__(self):
        return f"P({len(self)}) {self.name}"

    @property
    def fol(self):
        """Return Foliations of pairs as FoliationSet."""

        return FoliationSet([e.fol for e in self], name=self.name)

    @property
    def fvec(self):
        """Return planar normal vectors of pairs as Vector3Set."""

        return Vector3Set([e.fvec for e in self], name=self.name)

    @property
    def lin(self):
        """Return Lineation of pairs as LineationSet."""

        return LineationSet([e.lin for e in self], name=self.name)

    @property
    def lvec(self):
        """Return lineation vectors of pairs as Vector3Set"""

        return Vector3Set([e.lvec for e in self], name=self.name)

    misfit = _per_item("misfit", "Return array of misfits.")
    rake = _per_item(
        "rake", "Return array of rakes of linear features on planar features."
    )

    @property
    def rax(self):
        """Return vectors perpendicular to both planar and linear parts."""
        return Vector3Set([e.rax for e in self], name=self.name)

    def angle(self, other=None):
        """Return angles of all data in ``PairSet`` object."""
        res = []
        if other is None:
            res = [
                abs(Rotation3.from_two_pairs(e, f, symmetry=True).axisangle()[1])
                for e, f in combinations(self.data, 2)
            ]
        elif isinstance(other, PairSet):
            self._check_same_length(other, "angle")
            res = [
                abs(Rotation3.from_two_pairs(e, f, symmetry=True).axisangle()[1])
                for e, f in zip(self, other)
            ]
        elif isinstance(other, Pair):
            res = [
                abs(Rotation3.from_two_pairs(e, other, symmetry=True).axisangle()[1])
                for e in self
            ]
        else:
            raise TypeError("Wrong argument type!")
        return np.asarray(res)

    def ortensor(self):
        """Return Lisle (1989) orientation tensor ``OrientationTensor3`` of orientations."""
        self._require(1, "ortensor")
        return OrientationTensor3.from_pairs(self)

    def lisle_statistics(self, level=0.95):
        """Mean pair and confidence ellipses of its lineation and foliation normal.

        The mean is the pair of Lisle (1989): the lineation and the foliation
        normal are the first and the last eigenvector of the tensor
        ``mean(l l' - n n')``, which treats lineations and normals together and
        gives an orthogonal mean. The confidence ellipses are the large-sample
        (central limit theorem) regions of these eigenvectors. For faults the
        sense is not used.

        Lisle, R. (1989). The Statistical Analysis of Orthogonal Orientation Data.
            The Journal of Geology, 97(3), 360-364.

        Args:
            level: confidence level. Default 0.95.

        Returns:
            dict: with keys ``mean`` (``Pair``), ``eigenvalues`` (of the tensor,
                sum 0, ``(1, 0, -1)`` for identical pairs), ``lin`` and ``fol``
                (confidence ellipses of the mean lineation and normal, with keys
                ``mu``, ``axes`` and ``gamma`` as ``bingham_statistics``), ``n``
                and ``level``. ``Lisle.fit(pairs)`` also fits the
                concentrations of the ``Lisle`` distribution.
        """
        self._require(2, "lisle_statistics")
        X = pair_arrays(self)
        taus, vecs, moments = lisle_moments(X)
        lin, fol = lisle_ellipses(taus, vecs, moments, len(X), level)
        return {
            "mean": Pair(Vector3(vecs[2]), Vector3(vecs[0])),
            "eigenvalues": tuple(float(t) for t in taus),
            "lin": lin.as_dict(),
            "fol": fol.as_dict(),
            "n": len(X),
            "level": level,
        }

    def label(self):
        return str(self)

    @classmethod
    def random(cls, n=25):
        """Create PairSet of random pairs."""

        return PairSet([Pair.random() for i in range(check_count(n))])

    @classmethod
    def from_csv(
        cls,
        filename,
        delimiter=None,
        facol=0,
        ficol=1,
        lacol=2,
        licol=3,
        name=None,
    ):
        """Read ``PairSet`` from csv file.

        Args:
            filename (str): name of CSV file to load.

        Keyword Args:
            delimiter (str): values delimiter. Default None, i.e. detected
                from the file.
            facol, ficol, lacol, licol (int or str): columns of the planar
                azimuth and inclination and the linear azimuth and
                inclination, as indexes (starting from 0) or column headers.
                Default 0, 1, 2, 3.
            name (str): name of the created ``PairSet``. Default is the
                filename without its directory part.
        """

        fazi, finc, lazi, linc = _read_csv_columns(
            filename, (facol, ficol, lacol, licol), delimiter
        )
        return cls.from_array(
            fazi,
            finc,
            lazi,
            linc,
            name=name if name is not None else basename(filename),
        )

    def to_csv(self, filename, delimiter=","):
        """Save ``PairSet`` object to csv file.

        Args:
            filename (str): name of CSV file to save.

        Keyword Args:
            delimiter (str): values delimiter. Default ','.

        Note:
            Written values are rounded according to `ndigits` settings in apsg_conf.

        Returns:
            None
        """

        n = apsg_conf.ndigits

        with open(filename, "w", newline="") as csvfile:
            fieldnames = ["fazi", "finc", "lazi", "linc"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            for dt in self:
                fazi, finc = dt.fol.geo
                lazi, linc = dt.lin.geo
                writer.writerow(
                    {
                        "fazi": round(fazi, n),
                        "finc": round(finc, n),
                        "lazi": round(lazi, n),
                        "linc": round(linc, n),
                    }
                )

    @classmethod
    def from_array(cls, fazis, fincs, lazis, lincs, senses=None, name="Default"):
        """Create ``PairSet`` from arrays of azimuths, inclinations and senses.

        Args:
            fazis: list or array of azimuths.
            fincs: list or array of inclinations.
            lazis: list or array of azimuths.
            lincs: list or array of inclinations.

        Keyword Args:
            name: name of ``PairSet`` object. Default is 'Default'.
        Returns:
            PairSet: The created feature set.
        """

        return cls(
            [
                cls.__feature_class__(fazi, finc, lazi, linc)
                for fazi, finc, lazi, linc in zip(fazis, fincs, lazis, lincs)
            ],
            name=name,
        )


class FaultSet(PairSet):
    """
    Class to store set of ``Fault`` features.
    """

    __feature_class__ = Fault

    def __init__(self, data, name="Default"):
        super().__init__(data, name=name)
        if not all(isinstance(obj, Fault) for obj in data):
            raise TypeError("Data must be instances of Fault")

    def __repr__(self):
        return f"F({len(self)}) {self.name}"

    sense = _per_item("sense", "Return array of sense values.")
    sense_str = _per_item("sense_str", "Return array of sense characters.")

    def p_vector(self, ptangle=90):
        """Return p-axes of FaultSet as Vector3Set."""

        return Vector3Set([e.p_vector(ptangle) for e in self], name=self.name)

    def t_vector(self, ptangle=90):
        """Return t-axes of FaultSet as Vector3Set."""

        return Vector3Set([e.t_vector(ptangle) for e in self], name=self.name)

    @property
    def p(self):
        """Return p-axes of FaultSet as LineationSet."""

        return LineationSet([e.p for e in self], name=self.name + "-P")

    @property
    def t(self):
        """Return t-axes of FaultSet as LineationSet."""

        return LineationSet([e.t for e in self], name=self.name + "-T")

    @property
    def m(self):
        """Return m-planes of FaultSet as FoliationSet."""

        return FoliationSet([e.m for e in self], name=self.name + "-M")

    @property
    def d(self):
        """Return dihedra planes of FaultSet as FoliationSet."""

        return FoliationSet([e.d for e in self], name=self.name + "-D")

    def angle(self, other=None):
        """Return angles of all data in ``FaultSet`` object."""
        res = []
        if other is None:
            res = [
                abs(Rotation3.from_two_pairs(e, f, symmetry=False).axisangle()[1])
                for e, f in combinations(self.data, 2)
            ]
        elif isinstance(other, FaultSet):
            self._check_same_length(other, "angle")
            res = [
                abs(Rotation3.from_two_pairs(e, f, symmetry=False).axisangle()[1])
                for e, f in zip(self, other)
            ]
        elif isinstance(other, Fault):
            res = [
                abs(Rotation3.from_two_pairs(e, other, symmetry=False).axisangle()[1])
                for e in self
            ]
        else:
            raise TypeError("Wrong argument type!")
        return np.asarray(res)

    @classmethod
    def random(cls, n=25):
        """Create FaultSet of random faults."""

        return FaultSet([Fault.random() for i in range(check_count(n))])

    @classmethod
    def from_csv(
        cls,
        filename,
        delimiter=None,
        sense_str=False,
        facol=0,
        ficol=1,
        lacol=2,
        licol=3,
        scol=4,
        name=None,
    ):
        """Read ``FaultSet`` from csv file.

        Args:
            filename (str): name of CSV file to load.

        Keyword Args:
            delimiter (str): values delimiter. Default None, i.e. detected
                from the file.
            sense_str (bool): True when the sense column holds N, R, S or D,
                False when it holds integers. Default False.
            facol, ficol, lacol, licol, scol (int or str): columns of the
                planar azimuth and inclination, the linear azimuth and
                inclination and the sense, as indexes (starting from 0) or
                column headers. Default 0, 1, 2, 3, 4.
            name (str): name of the created ``FaultSet``. Default is the
                filename without its directory part.
        """

        fazi, finc, lazi, linc, sense = _read_csv_columns(
            filename,
            (facol, ficol, lacol, licol, scol),
            delimiter,
            convert=(*(_finite_float,) * 4, str if sense_str else int),
        )
        return cls.from_array(
            fazi,
            finc,
            lazi,
            linc,
            sense,
            name=name if name is not None else basename(filename),
        )

    def to_csv(self, filename, delimiter=",", sense_str=False):
        """Save ``FaultSet`` object to csv file.

        Args:
            filename (str): name of CSV file to save.

        Keyword Args:
            delimiter (str): values delimiter. Default ','.
            sense_str (bool): save sense as N, R, S or D. Default False.

        Note:
            Written values are rounded according to `ndigits` settings in apsg_conf.

        Returns:
            None
        """

        n = apsg_conf.ndigits

        with open(filename, "w", newline="") as csvfile:
            fieldnames = ["fazi", "finc", "lazi", "linc", "sense"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            for dt in self:
                fazi, finc = dt.fol.geo
                lazi, linc = dt.lin.geo
                writer.writerow(
                    {
                        "fazi": round(fazi, n),
                        "finc": round(finc, n),
                        "lazi": round(lazi, n),
                        "linc": round(linc, n),
                        "sense": dt.sense_str if sense_str else dt.sense,
                    }
                )

    @classmethod
    def from_array(cls, fazis, fincs, lazis, lincs, senses=None, name="Default"):
        """Create ``FaultSet`` from arrays of azimuths, inclinations and senses.

        Args:
            fazis: list or array of azimuths.
            fincs: list or array of inclinations.
            lazis: list or array of azimuths.
            lincs: list or array of inclinations.
            senses: list or array of senses.

        Keyword Args:
            name: name of ``FaultSet`` object. Default is 'Default'.
        Returns:
            FaultSet: The created feature set.
        """
        if senses is None:
            senses = []
        return cls(
            [
                cls.__feature_class__(fazi, finc, lazi, linc, sense)
                for fazi, finc, lazi, linc, sense in zip(
                    fazis, fincs, lazis, lincs, senses
                )
            ],
            name=name,
        )

    def stress_inversion(self, bootstrap=False, n=100):
        """Stress inversion from fault-slip data.

        The 4-D inversion method developed by Michael (1984) is a classic approach
        in structural geology. It determines the orientation of the principal stress
        axes (σ1, σ2, σ3) and the stress ratio R by assuming that the slip on a fault
        occurs in the direction of the maximum resolved shear stress.

        Assumptions of the method:

        - one homogeneous stress tensor explains all the faults;
        - each slip vector is parallel to the shear traction resolved on its
          fault plane;
        - the shear traction has the same magnitude on every fault, which
          turns the problem into a linear least-squares one (each fault
          gives three equations, one per component of the unit slip vector);
        - the isotropic part of the tensor cannot be resolved, so the result
          is the trace-free (deviatoric) tensor with five unknowns
          (σ33 = -(σ11 + σ22)), and only the principal axes and R are
          meaningful, not the absolute magnitudes.

        Args:
            bootstrap (bool): When True return set of stress tensors.
            n (int): number of bootstrapped samples. Default 100.

        Returns:
            Stress3: Stress inversion from fault-slip data.

        References:
            Michael, A.J. (1984). Determination of stress from slip data:
                faults and folds. Journal of Geophysical Research 89(B13),
                11517-11526.
        """

        def solve_michael_inversion(faults):
            """Linear inversion for the deviatoric stress tensor components."""
            faults = list(faults)
            n = len(faults)
            A = np.empty((3 * n, 5))
            d = np.empty(3 * n)

            for i, f in enumerate(faults):
                nx, ny, nz = f.fvec._coords
                sx, sy, sz = f.lvec._coords
                # Rows are the components of the shear traction
                # tau = sigma n - (n . sigma n) n (Michael 1984, eq. 2) on the
                # plane with normal n, written out for the unknowns
                # s11, s12, s13, s22, s23 with s33 = -(s11 + s22). The
                # coefficients are frame independent (the same as in
                # Michael's eq. 3 and in SATSI).
                A[3 * i : 3 * i + 3] = [
                    [
                        nx * (1 - nx**2 + nz**2),
                        ny * (1 - 2 * nx**2),
                        nz * (1 - 2 * nx**2),
                        -nx * (ny**2 - nz**2),
                        -2 * nx * ny * nz,
                    ],
                    [
                        -ny * (nx**2 - nz**2),
                        nx * (1 - 2 * ny**2),
                        -2 * nx * ny * nz,
                        ny * (1 - ny**2 + nz**2),
                        nz * (1 - 2 * ny**2),
                    ],
                    [
                        -nz * (1 + nx**2 - nz**2),
                        -2 * nx * ny * nz,
                        nx * (1 - 2 * nz**2),
                        -nz * (1 + ny**2 - nz**2),
                        ny * (1 - 2 * nz**2),
                    ],
                ]
                d[3 * i : 3 * i + 3] = [sx, sy, sz]
            x, _, _, _ = np.linalg.lstsq(A, d, rcond=None)
            s11, s12, s13, s22, s23 = x
            s33 = -(s11 + s22)
            solution = np.array([[s11, s12, s13], [s12, s22, s23], [s13, s23, s33]])
            # Michael's n points from the footwall into the hanging wall and his
            # stress is tension positive. Fault.fvec is the opposite normal (as
            # in Stress3.fault, lvec = -tau(fvec)), so the solution is minus the
            # tension-positive tensor, whatever stress_convention is active.
            return Stress3._from_canonical(-solution)

        if not bootstrap:
            return solve_michael_inversion(self)
        else:
            return Stress3Set(
                [solve_michael_inversion(sample) for sample in self.bootstrap(n=n)]
            )

    def angular_misfit(self, sigma):
        """Angular misfit (°) between observed slip and predicted shear-traction direction.

        Args:
            sigma (Stress3): Stress tensor

        Returns:
            np.ndarray: Array of angular misfits in degrees.
        """
        return np.array([f.angular_misfit(sigma) for f in self])
