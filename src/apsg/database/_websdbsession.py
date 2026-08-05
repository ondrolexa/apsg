"""
REST API access to a websdb instance (https://github.com/ondrolexa/websdb).
"""

import warnings

from apsg.database._webtransport import (
    ProjectResolutionError,
    ReadOnlySessionError,
    WebSDBNotFoundError,
    _WebsdbTransport,
)
from apsg.feature._container import FaultSet, FoliationSet, LineationSet, PairSet
from apsg.feature._geodata import Fault, Foliation, Lineation, Pair


class WebSDBSession:
    """
    REST API interface to a websdb project.

    Args:
        project (int | str): project id, or project name (must match exactly one of
            the caller's projects).

    Keyword Args:
        url (str): websdb base URL. Falls back to the ``WEBSDB_URL`` environment
            variable, then ``"http://localhost:8080"``.
        username (str): falls back to ``WEBSDB_USERNAME``.
        password (str): falls back to ``WEBSDB_PASSWORD``.
        token (str): a pre-obtained JWT; skips the login round-trip. Falls back to
            ``WEBSDB_TOKEN``.
        mode (str): ``"r"`` (default, read-only) or ``"rw"``. Write methods raise
            ``ReadOnlySessionError`` immediately (before any HTTP call) when
            ``mode="r"``. The server independently enforces the caller's actual
            project role, so a ``mode="rw"`` session can still be rejected with a
            ``WebSDBPermissionError`` if that role is only ``"viewer"``.

    Note:
        Unlike ``SDBSession`` (which returns SQLAlchemy ORM instances), the listing
        methods here (``sites()``, ``rocks()``, ``units()``, ``structypes()``,
        ``tags()``) return plain JSON dicts - there is no local ORM layer for a REST
        backend.

    Examples:
        >>> db = WebSDBSession('Erzgebirge', mode='rw')
        >>> fols = db.folset(unit='Gneiss', tags=['banding'])

    """

    def __init__(
        self, project, url=None, username=None, password=None, token=None, mode="r"
    ):
        if mode not in ("r", "rw"):
            raise ValueError(f"mode must be 'r' or 'rw', got {mode!r}")
        self.mode = mode
        self._t = _WebsdbTransport(url, username, password, token)
        self._resolve_project(project)

    def _resolve_project(self, project):
        projects = self._t.request("GET", "/api/projects")
        if isinstance(project, int):
            match = next((p for p in projects if p["id"] == project), None)
            if match is None:
                raise WebSDBNotFoundError(
                    404, f"Project id {project} not found among caller's projects"
                )
        else:
            matches = [p for p in projects if p["name"] == project]
            if not matches:
                raise ProjectResolutionError(
                    f"No project named {project!r} found among caller's projects"
                )
            if len(matches) > 1:
                ids = [p["id"] for p in matches]
                raise ProjectResolutionError(
                    f"{len(matches)} projects named {project!r} found (ids: {ids}); "
                    "pass project=<id> to disambiguate"
                )
            match = matches[0]
        self._project = match
        self.project_id = match["id"]

    def refresh(self):
        """Re-validate the bound project (catches a rename or lost membership).

        Every read/write method below already hits the network live on every call and
        never caches anything, so this is otherwise a no-op - it exists only as an
        explicit, documented way to surface staleness early.
        """
        self._resolve_project(self.project_id)

    def __repr__(self):
        return self.info()

    def info(self):
        """Return a human-readable summary of the bound project."""

        n_geodata = len(self._list_all(f"/api/projects/{self.project_id}/geodata"))
        lines = [
            f"websdb project: {self._project['name']} (id={self.project_id})",
            f"URL: {self._t.base_url}",
            f"Mode: {self.mode}",
            f"Number of sites: {len(self.sites())}",
            f"Number of rocks: {len(self.rocks())}",
            f"Number of units: {len(self.units())}",
            f"Number of structypes: {len(self.structypes())}",
            f"Number of measurements: {n_geodata}",
        ]
        return "\n".join(lines)

    def close(self):
        """No persistent connection to close - kept for API symmetry with SDBSession."""

    def _require_write(self):
        if self.mode != "rw":
            raise ReadOnlySessionError(
                f"Session is read-only (mode={self.mode!r}); re-open with mode='rw' "
                "to write. The server may still reject the request with a permission "
                "error if your websdb role for this project is 'viewer'."
            )

    # --- pagination / listing ---

    def _list_all(self, path, params=None):
        params = dict(params or {})
        out = []
        offset = 0
        limit = 1000
        while True:
            page = self._t.request(
                "GET", path, params={**params, "limit": limit, "offset": offset}
            )
            out.extend(page)
            if len(page) < limit:
                return out
            offset += limit

    def sites(self):
        """Return list of all site dicts in the bound project."""

        return self._list_all(f"/api/projects/{self.project_id}/sites")

    def rocks(self, unit=None):
        """Return list of all rock dicts, optionally filtered to a single unit name."""

        params = {}
        if unit is not None:
            params["unit_id"] = self._resolve_unit_id(unit)
        return self._list_all(f"/api/projects/{self.project_id}/rocks", params)

    def units(self):
        """Return list of all unit dicts in the bound project."""

        return self._list_all(f"/api/projects/{self.project_id}/units")

    def structypes(self):
        """Return list of all structype dicts in the bound project."""

        return self._list_all(f"/api/projects/{self.project_id}/structypes")

    def tags(self):
        """Return list of ``{"name", "count"}`` dicts, sorted by count desc then name."""

        return self._t.request("GET", f"/api/projects/{self.project_id}/tags")

    # --- name resolution helpers ---

    def _resolve_unit_id(self, name):
        if name is None:
            return None
        for u in self.units():
            if u["name"] == name:
                return u["id"]
        raise ValueError(
            f"Unit {name!r} not found in project {self._project['name']!r}"
        )

    def _resolve_structype_id(self, name):
        if name is None:
            return None
        for s in self.structypes():
            if s["name"] == name:
                return s["id"]
        raise ValueError(
            f"Structype {name!r} not found in project {self._project['name']!r}"
        )

    def _resolve_site(self, site):
        if isinstance(site, dict):
            return site
        if isinstance(site, int):
            row = next((s for s in self.sites() if s["id"] == site), None)
            if row is None:
                raise ValueError(f"Site id {site} not found")
            return row
        row = next((s for s in self.sites() if s["name"] == site), None)
        if row is None:
            raise ValueError(f"Site {site!r} not found")
        return row

    @staticmethod
    def _rock_id(rock):
        return rock["id"] if isinstance(rock, dict) else rock

    def _rock_by_id(self, rock_id):
        return self._t.request("GET", f"/api/rocks/{rock_id}")

    def _site_by_id(self, site_id):
        return self._t.request("GET", f"/api/sites/{site_id}")

    @staticmethod
    def _id_of(obj, key="id"):
        attrs = getattr(obj, "_attrs", None) or {}
        if key not in attrs:
            raise ValueError(
                f"{type(obj).__name__} has no {key!r} in _attrs - it was not fetched "
                "or created via this WebSDBSession (or was built by hand); pass the "
                "id directly instead."
            )
        return attrs[key]

    def _resolve_geodata_id(self, obj):
        if isinstance(obj, int):
            return obj
        return self._id_of(obj, "id")

    # --- read: filtered geodata join ---

    def _fetch_filtered_geodata(
        self,
        datatype,
        site=None,
        site_contains=None,
        rock=None,
        rock_contains=None,
        unit=None,
        units=None,
        tags=None,
        structype=None,
    ):
        structype_id = self._resolve_structype_id(structype)
        geodata_params = (
            {"structype_id": structype_id} if structype_id is not None else {}
        )
        geodata = self._list_all(
            f"/api/projects/{self.project_id}/geodata", geodata_params
        )

        unit_id = self._resolve_unit_id(unit) if unit is not None else None
        rocks = self.rocks(unit=unit) if unit is not None else self.rocks()

        unit_ids = None
        if units is not None:
            name_to_id = {u["name"]: u["id"] for u in self.units()}
            unit_ids = {name_to_id[u] for u in units if u in name_to_id}

        site_by_id = {s["id"]: s for s in self.sites()}
        rock_by_id = {r["id"]: r for r in rocks}
        tag_set = set(tags) if tags else None

        out = []
        for gd in geodata:
            value = gd.get("value") or {}
            if value.get("datatype") != datatype:
                continue
            rock_row = rock_by_id.get(gd["rock_id"])
            if rock_row is None:
                continue
            site_row = site_by_id.get(rock_row["site_id"])
            if site_row is None:
                continue
            if site is not None and site_row["name"] != site:
                continue
            if (
                site_contains is not None
                and site_contains.casefold() not in site_row["name"].casefold()
            ):
                continue
            if rock is not None and rock_row["name"] != rock:
                continue
            if (
                rock_contains is not None
                and rock_contains.casefold() not in rock_row["name"].casefold()
            ):
                continue
            if unit_id is not None and rock_row.get("unit_id") != unit_id:
                continue
            if unit_ids is not None and rock_row.get("unit_id") not in unit_ids:
                continue
            if tag_set is not None:
                row_tags = set(value.get("kwargs", {}).get("tags", []))
                if not (tag_set & row_tags):
                    continue
            gd = dict(gd, _rock=rock_row, _site=site_row)
            out.append(gd)
        return out

    @staticmethod
    def _make_attrs(gd, unit_names, structype_names):
        rock = gd["_rock"]
        site = gd["_site"]
        unit_id = rock.get("unit_id")
        structype_id = gd.get("structype_id")
        return dict(
            id=gd["id"],
            rock_id=gd["rock_id"],
            site_id=rock["site_id"],
            site=site["name"],
            rock=rock["name"],
            unit=unit_names.get(unit_id) if unit_id is not None else None,
            structype=structype_names.get(structype_id)
            if structype_id is not None
            else None,
            tags=list(gd.get("value", {}).get("kwargs", {}).get("tags", [])),
            description=gd.get("description"),
        )

    def _name_lookups(self):
        return (
            {u["id"]: u["name"] for u in self.units()},
            {s["id"]: s["name"] for s in self.structypes()},
        )

    # --- read: FeatureSet-returning methods ---

    def folset(
        self,
        site=None,
        site_contains=None,
        rock=None,
        rock_contains=None,
        unit=None,
        units=None,
        tags=None,
        structype=None,
        name="Foliation",
    ):
        """Return matching Foliation geodata as a ``FoliationSet``.

        Every returned ``Foliation`` carries ``site``/``rock``/``unit``/``tags`` plus
        ``id``/``rock_id``/``site_id``/``structype``/``description`` in its
        ``_attrs``, so it can be passed straight to ``update_fol()``/``delete_fol()``.
        """

        rows = self._fetch_filtered_geodata(
            "Foliation",
            site=site,
            site_contains=site_contains,
            rock=rock,
            rock_contains=rock_contains,
            unit=unit,
            units=units,
            tags=tags,
            structype=structype,
        )
        unit_names, structype_names = self._name_lookups()
        items = [
            Foliation(
                *gd["value"]["args"],
                **self._make_attrs(gd, unit_names, structype_names),
            )
            for gd in rows
        ]
        return FoliationSet(items, name=name)

    def linset(
        self,
        site=None,
        site_contains=None,
        rock=None,
        rock_contains=None,
        unit=None,
        units=None,
        tags=None,
        structype=None,
        name="Lineation",
    ):
        """Return matching Lineation geodata as a ``LineationSet`` (see ``folset``)."""

        rows = self._fetch_filtered_geodata(
            "Lineation",
            site=site,
            site_contains=site_contains,
            rock=rock,
            rock_contains=rock_contains,
            unit=unit,
            units=units,
            tags=tags,
            structype=structype,
        )
        unit_names, structype_names = self._name_lookups()
        items = [
            Lineation(
                *gd["value"]["args"],
                **self._make_attrs(gd, unit_names, structype_names),
            )
            for gd in rows
        ]
        return LineationSet(items, name=name)

    def faultset(
        self,
        site=None,
        site_contains=None,
        rock=None,
        rock_contains=None,
        unit=None,
        units=None,
        tags=None,
        structype=None,
        name="Fault",
    ):
        """Return matching Fault geodata as a ``FaultSet`` (see ``folset``).

        websdb fault sense (``"N"``/``"R"``/``"S"``/``"D"``/``""``) is lowercased and
        handed to ``Fault``'s ``'n'``/``'r'``/``'s'``/``'d'`` convention. Records with
        an empty (unknown) sense, or whose planar/linear halves misfit by more than
        20 degrees, are skipped with a ``warnings.warn`` diagnostic rather than
        raising, mirroring ``SDBSession.getfaults()``'s existing skip behavior.
        """

        rows = self._fetch_filtered_geodata(
            "Fault",
            site=site,
            site_contains=site_contains,
            rock=rock,
            rock_contains=rock_contains,
            unit=unit,
            units=units,
            tags=tags,
            structype=structype,
        )
        unit_names, structype_names = self._name_lookups()
        faults = []
        for gd in rows:
            fazi, finc, lazi, linc, sense = gd["value"]["args"]
            if not sense:
                warnings.warn(
                    f"Skipping Fault geodata id={gd['id']} on rock "
                    f"{gd['_rock']['name']!r}: empty sense cannot be mapped to apsg's "
                    "Fault.calc_sense (requires one of 'n', 'r', 's', 'd')."
                )
                continue
            attrs = self._make_attrs(gd, unit_names, structype_names)
            misfit = False
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    fault = Fault(fazi, finc, lazi, linc, sense.lower(), **attrs)
                except UserWarning:
                    misfit = True
            if misfit:
                warnings.warn(
                    f"Skipping Fault geodata id={gd['id']} on rock "
                    f"{gd['_rock']['name']!r}: misfit angle between planar and "
                    "linear components exceeds 20 degrees."
                )
                continue
            faults.append(fault)
        return FaultSet(faults, name=name)

    def pairset(
        self,
        site=None,
        site_contains=None,
        rock=None,
        rock_contains=None,
        unit=None,
        units=None,
        ptags=None,
        ltags=None,
        structype=None,
        name="Pair",
    ):
        """Return Foliation+Lineation pairs (linked via websdb's ``pair_id``) as a
        ``PairSet``. Each half is independently filtered by the given criteria
        (``ptags``/``ltags`` apply to the planar/linear half respectively) - a pair
        whose partner half doesn't pass its own filter is silently excluded.
        """

        fol_rows = self._fetch_filtered_geodata(
            "Foliation",
            site=site,
            site_contains=site_contains,
            rock=rock,
            rock_contains=rock_contains,
            unit=unit,
            units=units,
            tags=ptags,
            structype=structype,
        )
        lin_rows = self._fetch_filtered_geodata(
            "Lineation",
            site=site,
            site_contains=site_contains,
            rock=rock,
            rock_contains=rock_contains,
            unit=unit,
            units=units,
            tags=ltags,
            structype=structype,
        )
        lin_by_id = {gd["id"]: gd for gd in lin_rows}
        unit_names, _ = self._name_lookups()

        pairs = []
        for fol_gd in fol_rows:
            partner_id = fol_gd.get("pair_id")
            if partner_id is None:
                continue
            lin_gd = lin_by_id.get(partner_id)
            if lin_gd is None:
                continue
            fazi, finc = fol_gd["value"]["args"]
            lazi, linc = lin_gd["value"]["args"]
            rock_row = fol_gd["_rock"]
            attrs = dict(
                fol_id=fol_gd["id"],
                lin_id=lin_gd["id"],
                rock_id=fol_gd["rock_id"],
                site_id=rock_row["site_id"],
                site=fol_gd["_site"]["name"],
                rock=rock_row["name"],
                unit=unit_names.get(rock_row.get("unit_id")),
            )
            misfit = False
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    pair = Pair(fazi, finc, lazi, linc, **attrs)
                except UserWarning:
                    misfit = True
            if misfit:
                warnings.warn(
                    f"Skipping pair (geodata ids {fol_gd['id']}/{lin_gd['id']}) on "
                    f"rock {rock_row['name']!r}: misfit angle exceeds 20 degrees."
                )
                continue
            pairs.append(pair)
        return PairSet(pairs, name=name)

    # --- write: get-or-create / update-in-place lookups ---

    def site(self, name, **kwargs):
        """Query a site by name, or create/update one when keyword args are given.

        Keyword Args:
            lon (float): longitude (WGS84 decimal degrees).
            lat (float): latitude (WGS84 decimal degrees).
            description (str): site description.
        """

        existing = next((s for s in self.sites() if s["name"] == name), None)
        if kwargs:
            self._require_write()
            if len(name) > 16:
                raise ValueError(
                    f"Site name {name!r} exceeds websdb's 16-character limit"
                )
            if existing is None:
                return self._t.request(
                    "POST",
                    f"/api/projects/{self.project_id}/sites",
                    json_body={
                        "name": name,
                        "lon": kwargs.get("lon"),
                        "lat": kwargs.get("lat"),
                        "description": kwargs.get("description"),
                    },
                )
            body = {
                k: v for k, v in kwargs.items() if k in ("lon", "lat", "description")
            }
            return self._t.request(
                "PUT", f"/api/sites/{existing['id']}", json_body=body
            )
        if existing is None:
            raise ValueError(f"Site {name!r} not found")
        return existing

    def rock(self, name, site, **kwargs):
        """Query a rock by name+site, or create/update one when keyword args are given.

        Args:
            name (str): rock name (not project-unique - scoped to `site`).
            site (dict | str | int): site dict, name, or id.

        Keyword Args:
            unit (str): unit name to assign.
        """

        site_row = self._resolve_site(site)
        existing = next(
            (
                r
                for r in self.rocks()
                if r["name"] == name and r["site_id"] == site_row["id"]
            ),
            None,
        )
        if kwargs:
            self._require_write()
            if len(name) > 255:
                raise ValueError(
                    f"Rock name {name!r} exceeds websdb's 255-character limit"
                )
            unit_id = (
                self._resolve_unit_id(kwargs["unit"]) if "unit" in kwargs else None
            )
            if existing is None:
                return self._t.request(
                    "POST",
                    f"/api/projects/{self.project_id}/rocks",
                    json_body={
                        "site_id": site_row["id"],
                        "name": name,
                        "unit_id": unit_id,
                    },
                )
            body = {"unit_id": unit_id} if "unit" in kwargs else {}
            return self._t.request(
                "PUT", f"/api/rocks/{existing['id']}", json_body=body
            )
        if existing is None:
            raise ValueError(f"Rock {name!r} not found at site {site_row['name']!r}")
        return existing

    def unit(self, name, **kwargs):
        """Query a unit by name, or create/update one when keyword args are given.

        Keyword Args:
            description (str): unit description.
        """

        existing = next((u for u in self.units() if u["name"] == name), None)
        if kwargs:
            self._require_write()
            if len(name) > 255:
                raise ValueError(
                    f"Unit name {name!r} exceeds websdb's 255-character limit"
                )
            if existing is None:
                return self._t.request(
                    "POST",
                    f"/api/projects/{self.project_id}/units",
                    json_body={"name": name, "description": kwargs.get("description")},
                )
            return self._t.request(
                "PUT",
                f"/api/units/{existing['id']}",
                json_body={k: v for k, v in kwargs.items() if k == "description"},
            )
        if existing is None:
            raise ValueError(f"Unit {name!r} not found")
        return existing

    def structype(self, name, **kwargs):
        """Query a structype by name, or create/update one when keyword args are given.

        Keyword Args:
            description (str): structype description.
        """

        existing = next((s for s in self.structypes() if s["name"] == name), None)
        if kwargs:
            self._require_write()
            if len(name) > 12:
                raise ValueError(
                    f"Structype name {name!r} exceeds websdb's 12-character limit"
                )
            if existing is None:
                return self._t.request(
                    "POST",
                    f"/api/projects/{self.project_id}/structypes",
                    json_body={"name": name, "description": kwargs.get("description")},
                )
            return self._t.request(
                "PUT",
                f"/api/structypes/{existing['id']}",
                json_body={k: v for k, v in kwargs.items() if k == "description"},
            )
        if existing is None:
            raise ValueError(f"Structype {name!r} not found")
        return existing

    # --- write: geodata construction from a fresh row ---

    def _foliation_from_geodata(self, gd):
        rock_row = self._rock_by_id(gd["rock_id"])
        site_row = self._site_by_id(rock_row["site_id"])
        gd = dict(gd, _rock=rock_row, _site=site_row)
        unit_names, structype_names = self._name_lookups()
        return Foliation(
            *gd["value"]["args"], **self._make_attrs(gd, unit_names, structype_names)
        )

    def _lineation_from_geodata(self, gd):
        rock_row = self._rock_by_id(gd["rock_id"])
        site_row = self._site_by_id(rock_row["site_id"])
        gd = dict(gd, _rock=rock_row, _site=site_row)
        unit_names, structype_names = self._name_lookups()
        return Lineation(
            *gd["value"]["args"], **self._make_attrs(gd, unit_names, structype_names)
        )

    def _fault_from_geodata(self, gd):
        rock_row = self._rock_by_id(gd["rock_id"])
        site_row = self._site_by_id(rock_row["site_id"])
        gd = dict(gd, _rock=rock_row, _site=site_row)
        unit_names, structype_names = self._name_lookups()
        fazi, finc, lazi, linc, sense = gd["value"]["args"]
        return Fault(
            fazi,
            finc,
            lazi,
            linc,
            sense.lower(),
            **self._make_attrs(gd, unit_names, structype_names),
        )

    # --- write: add ---

    def add_fol(
        self, rock, azimuth, inclination, structype=None, description=None, tags=None
    ):
        """Create a Foliation geodata row on `rock` (id or rock dict)."""

        self._require_write()
        value = {
            "datatype": "Foliation",
            "args": [azimuth, inclination],
            "kwargs": {"tags": tags or []},
        }
        gd = self._t.request(
            "POST",
            f"/api/projects/{self.project_id}/geodata",
            json_body={
                "rock_id": self._rock_id(rock),
                "value": value,
                "description": description,
                "structype_id": self._resolve_structype_id(structype),
            },
        )
        return self._foliation_from_geodata(gd)

    def add_lin(
        self, rock, azimuth, inclination, structype=None, description=None, tags=None
    ):
        """Create a Lineation geodata row on `rock` (id or rock dict)."""

        self._require_write()
        value = {
            "datatype": "Lineation",
            "args": [azimuth, inclination],
            "kwargs": {"tags": tags or []},
        }
        gd = self._t.request(
            "POST",
            f"/api/projects/{self.project_id}/geodata",
            json_body={
                "rock_id": self._rock_id(rock),
                "value": value,
                "description": description,
                "structype_id": self._resolve_structype_id(structype),
            },
        )
        return self._lineation_from_geodata(gd)

    def add_fault(
        self, rock, fault: Fault, structype=None, description=None, tags=None
    ):
        """Create a Fault geodata row on `rock` (id or rock dict) from an apsg
        ``Fault`` instance. Uses ``fault.sense_str`` (already ``'N'``/``'R'``/``'S'``/
        ``'D'``) for the written sense.
        """

        self._require_write()
        fazi, finc = fault.fol.geo
        lazi, linc = fault.lin.geo
        value = {
            "datatype": "Fault",
            "args": [fazi, finc, lazi, linc, fault.sense_str],
            "kwargs": {"tags": tags or []},
        }
        gd = self._t.request(
            "POST",
            f"/api/projects/{self.project_id}/geodata",
            json_body={
                "rock_id": self._rock_id(rock),
                "value": value,
                "description": description,
                "structype_id": self._resolve_structype_id(structype),
            },
        )
        return self._fault_from_geodata(gd)

    def add_pair(
        self,
        rock,
        pair: Pair,
        fol_structype=None,
        lin_structype=None,
        description=None,
        tags=None,
    ):
        """Create a Foliation + Lineation geodata pair on `rock` from an apsg ``Pair``
        instance, and link them via ``POST /api/geodata/pair``.

        Note: unlike ``Foliation``/``Lineation``/``Fault`` (each a single geodata
        row), a ``Pair`` maps to two rows - the returned ``Pair``'s ``_attrs`` carries
        ``fol_id``/``lin_id`` instead of a single ``id``.
        """

        self._require_write()
        fazi, finc = pair.fol.geo
        lazi, linc = pair.lin.geo
        fol = self.add_fol(
            rock,
            fazi,
            finc,
            structype=fol_structype,
            description=description,
            tags=tags,
        )
        lin = self.add_lin(
            rock,
            lazi,
            linc,
            structype=lin_structype,
            description=description,
            tags=tags,
        )
        fol_id = fol._attrs["id"]
        lin_id = lin._attrs["id"]
        self._t.request(
            "POST", "/api/geodata/pair", json_body={"geodata_ids": [fol_id, lin_id]}
        )
        attrs = dict(
            fol_id=fol_id,
            lin_id=lin_id,
            rock_id=fol._attrs["rock_id"],
            site_id=fol._attrs["site_id"],
            site=fol._attrs["site"],
            rock=fol._attrs["rock"],
            unit=fol._attrs["unit"],
        )
        return Pair(fazi, finc, lazi, linc, **attrs)

    # --- write: update / delete geodata ---

    def update_fol(
        self,
        fol,
        azimuth=None,
        inclination=None,
        structype=None,
        description=None,
        tags=None,
        rock=None,
    ):
        """Update a Foliation geodata row. `fol` may be a previously-fetched/created
        ``Foliation`` (reads ``_attrs['id']``) or a raw geodata id.
        """

        self._require_write()
        gd_id = self._resolve_geodata_id(fol)
        body = {}
        if azimuth is not None or inclination is not None or tags is not None:
            current = self._t.request("GET", f"/api/geodata/{gd_id}")
            cur_azi, cur_inc = current["value"]["args"]
            cur_tags = current["value"].get("kwargs", {}).get("tags", [])
            body["value"] = {
                "datatype": "Foliation",
                "args": [
                    azimuth if azimuth is not None else cur_azi,
                    inclination if inclination is not None else cur_inc,
                ],
                "kwargs": {"tags": tags if tags is not None else cur_tags},
            }
        if description is not None:
            body["description"] = description
        if structype is not None:
            body["structype_id"] = self._resolve_structype_id(structype)
        if rock is not None:
            body["rock_id"] = self._rock_id(rock)
        gd = self._t.request("PUT", f"/api/geodata/{gd_id}", json_body=body)
        return self._foliation_from_geodata(gd)

    def delete_fol(self, fol):
        """Delete a Foliation geodata row. `fol` may be a ``Foliation`` or a raw id."""

        self._require_write()
        gd_id = self._resolve_geodata_id(fol)
        self._t.request("DELETE", f"/api/geodata/{gd_id}")

    def update_lin(
        self,
        lin,
        azimuth=None,
        inclination=None,
        structype=None,
        description=None,
        tags=None,
        rock=None,
    ):
        """Update a Lineation geodata row (see ``update_fol``)."""

        self._require_write()
        gd_id = self._resolve_geodata_id(lin)
        body = {}
        if azimuth is not None or inclination is not None or tags is not None:
            current = self._t.request("GET", f"/api/geodata/{gd_id}")
            cur_azi, cur_inc = current["value"]["args"]
            cur_tags = current["value"].get("kwargs", {}).get("tags", [])
            body["value"] = {
                "datatype": "Lineation",
                "args": [
                    azimuth if azimuth is not None else cur_azi,
                    inclination if inclination is not None else cur_inc,
                ],
                "kwargs": {"tags": tags if tags is not None else cur_tags},
            }
        if description is not None:
            body["description"] = description
        if structype is not None:
            body["structype_id"] = self._resolve_structype_id(structype)
        if rock is not None:
            body["rock_id"] = self._rock_id(rock)
        gd = self._t.request("PUT", f"/api/geodata/{gd_id}", json_body=body)
        return self._lineation_from_geodata(gd)

    def delete_lin(self, lin):
        """Delete a Lineation geodata row. `lin` may be a ``Lineation`` or a raw id."""

        self._require_write()
        gd_id = self._resolve_geodata_id(lin)
        self._t.request("DELETE", f"/api/geodata/{gd_id}")

    def update_fault(
        self,
        fault,
        planar_azimuth=None,
        planar_inclination=None,
        linear_azimuth=None,
        linear_inclination=None,
        sense=None,
        structype=None,
        description=None,
        tags=None,
        rock=None,
    ):
        """Update a Fault geodata row (see ``update_fol``). `sense` is
        ``'N'``/``'R'``/``'S'``/``'D'``/``''`` (websdb's convention, not apsg's).
        """

        self._require_write()
        gd_id = self._resolve_geodata_id(fault)
        body = {}
        if any(
            v is not None
            for v in (
                planar_azimuth,
                planar_inclination,
                linear_azimuth,
                linear_inclination,
                sense,
                tags,
            )
        ):
            current = self._t.request("GET", f"/api/geodata/{gd_id}")
            cur_fazi, cur_finc, cur_lazi, cur_linc, cur_sense = current["value"]["args"]
            cur_tags = current["value"].get("kwargs", {}).get("tags", [])
            body["value"] = {
                "datatype": "Fault",
                "args": [
                    planar_azimuth if planar_azimuth is not None else cur_fazi,
                    planar_inclination if planar_inclination is not None else cur_finc,
                    linear_azimuth if linear_azimuth is not None else cur_lazi,
                    linear_inclination if linear_inclination is not None else cur_linc,
                    sense if sense is not None else cur_sense,
                ],
                "kwargs": {"tags": tags if tags is not None else cur_tags},
            }
        if description is not None:
            body["description"] = description
        if structype is not None:
            body["structype_id"] = self._resolve_structype_id(structype)
        if rock is not None:
            body["rock_id"] = self._rock_id(rock)
        gd = self._t.request("PUT", f"/api/geodata/{gd_id}", json_body=body)
        return self._fault_from_geodata(gd)

    def delete_fault(self, fault):
        """Delete a Fault geodata row. `fault` may be a ``Fault`` or a raw id."""

        self._require_write()
        gd_id = self._resolve_geodata_id(fault)
        self._t.request("DELETE", f"/api/geodata/{gd_id}")

    def update_pair(
        self,
        pair,
        fazi=None,
        finc=None,
        lazi=None,
        linc=None,
        description=None,
        tags=None,
    ):
        """Update the planar and/or linear half of a pair. `pair` must carry
        ``_attrs['fol_id']``/``['lin_id']`` (i.e. came from ``pairset()`` or
        ``add_pair()``).
        """

        self._require_write()
        fol_id = self._id_of(pair, "fol_id")
        lin_id = self._id_of(pair, "lin_id")
        if (
            fazi is not None
            or finc is not None
            or description is not None
            or tags is not None
        ):
            self.update_fol(
                fol_id,
                azimuth=fazi,
                inclination=finc,
                description=description,
                tags=tags,
            )
        if (
            lazi is not None
            or linc is not None
            or description is not None
            or tags is not None
        ):
            self.update_lin(
                lin_id,
                azimuth=lazi,
                inclination=linc,
                description=description,
                tags=tags,
            )
        fol_gd = self._t.request("GET", f"/api/geodata/{fol_id}")
        lin_gd = self._t.request("GET", f"/api/geodata/{lin_id}")
        new_fazi, new_finc = fol_gd["value"]["args"]
        new_lazi, new_linc = lin_gd["value"]["args"]
        rock_row = self._rock_by_id(fol_gd["rock_id"])
        site_row = self._site_by_id(rock_row["site_id"])
        unit_names, _ = self._name_lookups()
        return Pair(
            new_fazi,
            new_finc,
            new_lazi,
            new_linc,
            fol_id=fol_id,
            lin_id=lin_id,
            rock_id=fol_gd["rock_id"],
            site_id=rock_row["site_id"],
            site=site_row["name"],
            rock=rock_row["name"],
            unit=unit_names.get(rock_row.get("unit_id")),
        )

    def delete_pair(self, pair):
        """Delete both halves of a pair. `pair` must carry ``_attrs['fol_id']``/
        ``['lin_id']`` (i.e. came from ``pairset()`` or ``add_pair()``).
        """

        self._require_write()
        fol_id = self._id_of(pair, "fol_id")
        lin_id = self._id_of(pair, "lin_id")
        self._t.request("DELETE", f"/api/geodata/{fol_id}")
        self._t.request("DELETE", f"/api/geodata/{lin_id}")

    # --- write: delete site/rock/unit/structype ---

    def delete_site(self, site):
        """Delete a site (cascades to its rocks and their geodata)."""

        self._require_write()
        if isinstance(site, dict):
            site_id = site["id"]
        elif isinstance(site, int):
            site_id = site
        else:
            site_id = self._resolve_site(site)["id"]
        self._t.request("DELETE", f"/api/sites/{site_id}")

    def delete_rock(self, rock):
        """Delete a rock (cascades to its geodata)."""

        self._require_write()
        self._t.request("DELETE", f"/api/rocks/{self._rock_id(rock)}")

    def delete_unit(self, unit):
        """Delete a unit (clears ``unit_id`` on any rock that referenced it)."""

        self._require_write()
        if isinstance(unit, dict):
            unit_id = unit["id"]
        elif isinstance(unit, int):
            unit_id = unit
        else:
            unit_id = self._resolve_unit_id(unit)
        self._t.request("DELETE", f"/api/units/{unit_id}")

    def delete_structype(self, structype):
        """Delete a structype (clears ``structype_id`` on referencing geodata)."""

        self._require_write()
        if isinstance(structype, dict):
            structype_id = structype["id"]
        elif isinstance(structype, int):
            structype_id = structype
        else:
            structype_id = self._resolve_structype_id(structype)
        self._t.request("DELETE", f"/api/structypes/{structype_id}")

    # --- write: tags ---

    def tag_rename(self, old_name, new_name):
        """Rename a tag project-wide. Returns ``{"updated": <row count touched>}``."""

        self._require_write()
        return self._t.request(
            "PUT",
            f"/api/projects/{self.project_id}/tags/rename",
            json_body={"old_name": old_name, "new_name": new_name},
        )

    def tag_delete(self, name):
        """Delete a tag project-wide. Returns ``{"updated": <row count touched>}``."""

        self._require_write()
        return self._t.request(
            "POST",
            f"/api/projects/{self.project_id}/tags/delete",
            json_body={"name": name},
        )
