"""
Tests for apsg.database.WebSDBSession against a fake in-memory websdb backend.

No real HTTP/sockets are used - `_WebsdbTransport._send` (the sole network-touching
seam) is monkeypatched, either with a fake urllib response (transport-level tests, to
exercise the real request-building/pagination/error-mapping code) or with an in-memory
router that mimics the websdb REST API (WebSDBSession-level tests).
"""

import io
import json
import urllib.error
import urllib.parse

import pytest

from apsg.database import (
    ProjectResolutionError,
    ReadOnlySessionError,
    WebSDBAuthError,
    WebSDBConflictError,
    WebSDBNotFoundError,
    WebSDBPermissionError,
    WebSDBSession,
)
from apsg.database._webtransport import _WebsdbTransport
from apsg.feature._geodata import Fault, Foliation, Pair

# --- transport-level tests (real urllib error-mapping / pagination code) ---


class _FakeHTTPResponse:
    def __init__(self, status, body):
        self.status = status
        self._body = json.dumps(body).encode() if body is not None else b""

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_transport_login_sets_token(monkeypatch):
    def fake_urlopen(req, *a, **kw):
        assert req.full_url.endswith("/api/login")
        return _FakeHTTPResponse(
            200, {"access_token": "tok123", "token_type": "bearer"}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    t = _WebsdbTransport(base_url="http://x", username="a", password="b")
    result = t.login()
    assert t.token == "tok123"
    assert result["access_token"] == "tok123"


@pytest.mark.parametrize(
    "status,exc_cls",
    [
        (401, WebSDBAuthError),
        (403, WebSDBPermissionError),
        (404, WebSDBNotFoundError),
        (409, WebSDBConflictError),
        (500, Exception),
    ],
)
def test_transport_maps_error_status(monkeypatch, status, exc_cls):
    def fake_urlopen(req, *a, **kw):
        raise urllib.error.HTTPError(
            req.full_url,
            status,
            "err",
            {},
            io.BytesIO(json.dumps({"detail": "x"}).encode()),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    t = _WebsdbTransport(base_url="http://x", token="tok")
    with pytest.raises(exc_cls):
        t.request("GET", "/api/whatever")


def test_list_all_pagination(monkeypatch):
    pages = {0: list(range(1000)), 1000: list(range(1000, 1037))}

    def fake_send(req):
        parsed = urllib.parse.urlparse(req.full_url)
        qs = urllib.parse.parse_qs(parsed.query)
        offset = int(qs["offset"][0])
        return pages[offset]

    monkeypatch.setattr(_WebsdbTransport, "_send", staticmethod(fake_send))
    session = object.__new__(WebSDBSession)
    session._t = _WebsdbTransport(base_url="http://x", token="tok")
    session.project_id = 1
    result = session._list_all("/api/projects/1/geodata")
    assert result == list(range(1037))


# --- fake in-memory websdb backend for WebSDBSession-level tests ---


class FakeWebsdb:
    def __init__(self):
        self._next_id = 1
        self.projects = {}
        self.sites = {}
        self.rocks = {}
        self.units = {}
        self.structypes = {}
        self.geodata = {}
        self.calls = []  # (method, path, query_params, body)

    def _new_id(self):
        i = self._next_id
        self._next_id += 1
        return i

    def add_project(self, name, description=None):
        pid = self._new_id()
        self.projects[pid] = {"id": pid, "name": name, "description": description}
        return self.projects[pid]

    def add_site(self, project_id, name, lon=None, lat=None, description=None):
        sid = self._new_id()
        self.sites[sid] = {
            "id": sid,
            "project_id": project_id,
            "name": name,
            "lon": lon,
            "lat": lat,
            "description": description,
        }
        return self.sites[sid]

    def add_unit(self, project_id, name, description=None):
        uid = self._new_id()
        self.units[uid] = {
            "id": uid,
            "project_id": project_id,
            "name": name,
            "description": description,
        }
        return self.units[uid]

    def add_structype(self, project_id, name, description=None):
        tid = self._new_id()
        self.structypes[tid] = {
            "id": tid,
            "project_id": project_id,
            "name": name,
            "description": description,
        }
        return self.structypes[tid]

    def add_rock(self, project_id, site_id, name, unit_id=None):
        rid = self._new_id()
        self.rocks[rid] = {
            "id": rid,
            "project_id": project_id,
            "site_id": site_id,
            "name": name,
            "unit_id": unit_id,
        }
        return self.rocks[rid]

    def add_geodata(
        self, project_id, rock_id, value, description=None, structype_id=None
    ):
        gid = self._new_id()
        self.geodata[gid] = {
            "id": gid,
            "project_id": project_id,
            "rock_id": rock_id,
            "value": value,
            "description": description,
            "structype_id": structype_id,
            "pair_id": None,
        }
        return self.geodata[gid]

    @staticmethod
    def _paginate(rows, qs):
        limit = int(qs.get("limit", 200))
        offset = int(qs.get("offset", 0))
        return rows[offset : offset + limit]

    def send(self, req):
        method = req.get_method()
        parsed = urllib.parse.urlparse(req.full_url)
        path = parsed.path
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        body = json.loads(req.data) if req.data else None
        self.calls.append((method, path, qs, body))
        return self._route(method, path, qs, body)

    def _route(self, method, path, qs, body):
        parts = [p for p in path.split("/") if p]
        rest = parts[1:]  # drop leading "api"

        if rest == ["login"]:
            return {"access_token": "faketoken", "token_type": "bearer"}

        if rest == ["projects"] and method == "GET":
            return list(self.projects.values())

        if len(rest) == 2 and rest[0] == "projects" and method == "GET":
            return self.projects[int(rest[1])]

        if len(rest) == 3 and rest[0] == "projects" and rest[2] == "sites":
            pid = int(rest[1])
            if method == "GET":
                return self._paginate(
                    [s for s in self.sites.values() if s["project_id"] == pid], qs
                )
            if method == "POST":
                return self.add_site(
                    pid,
                    body["name"],
                    body.get("lon"),
                    body.get("lat"),
                    body.get("description"),
                )

        if len(rest) == 2 and rest[0] == "sites":
            sid = int(rest[1])
            if method == "GET":
                return self.sites[sid]
            if method == "PUT":
                self.sites[sid].update(body)
                return self.sites[sid]
            if method == "DELETE":
                del self.sites[sid]
                return None

        if len(rest) == 3 and rest[0] == "projects" and rest[2] == "rocks":
            pid = int(rest[1])
            if method == "GET":
                rows = [r for r in self.rocks.values() if r["project_id"] == pid]
                if qs.get("unit_id") is not None:
                    rows = [r for r in rows if str(r.get("unit_id")) == qs["unit_id"]]
                return self._paginate(rows, qs)
            if method == "POST":
                return self.add_rock(
                    pid, body["site_id"], body["name"], body.get("unit_id")
                )

        if len(rest) == 2 and rest[0] == "rocks":
            rid = int(rest[1])
            if method == "GET":
                return self.rocks[rid]
            if method == "PUT":
                self.rocks[rid].update(body)
                return self.rocks[rid]
            if method == "DELETE":
                del self.rocks[rid]
                return None

        if len(rest) == 3 and rest[0] == "projects" and rest[2] == "units":
            pid = int(rest[1])
            if method == "GET":
                return self._paginate(
                    [u for u in self.units.values() if u["project_id"] == pid], qs
                )
            if method == "POST":
                return self.add_unit(pid, body["name"], body.get("description"))

        if len(rest) == 2 and rest[0] == "units":
            uid = int(rest[1])
            if method == "PUT":
                self.units[uid].update(body)
                return self.units[uid]
            if method == "DELETE":
                del self.units[uid]
                for r in self.rocks.values():
                    if r.get("unit_id") == uid:
                        r["unit_id"] = None
                return None

        if len(rest) == 3 and rest[0] == "projects" and rest[2] == "structypes":
            pid = int(rest[1])
            if method == "GET":
                return self._paginate(
                    [s for s in self.structypes.values() if s["project_id"] == pid], qs
                )
            if method == "POST":
                return self.add_structype(pid, body["name"], body.get("description"))

        if len(rest) == 2 and rest[0] == "structypes":
            tid = int(rest[1])
            if method == "PUT":
                self.structypes[tid].update(body)
                return self.structypes[tid]
            if method == "DELETE":
                del self.structypes[tid]
                for g in self.geodata.values():
                    if g.get("structype_id") == tid:
                        g["structype_id"] = None
                return None

        if rest == ["geodata", "pair"] and method == "POST":
            id1, id2 = body["geodata_ids"]
            self.geodata[id1]["pair_id"] = id2
            self.geodata[id2]["pair_id"] = id1
            return [self.geodata[id1], self.geodata[id2]]

        if len(rest) == 3 and rest[0] == "projects" and rest[2] == "geodata":
            pid = int(rest[1])
            if method == "GET":
                rows = [g for g in self.geodata.values() if g["project_id"] == pid]
                if qs.get("structype_id") is not None:
                    rows = [
                        g
                        for g in rows
                        if str(g.get("structype_id")) == qs["structype_id"]
                    ]
                return self._paginate(rows, qs)
            if method == "POST":
                return self.add_geodata(
                    pid,
                    body["rock_id"],
                    body["value"],
                    body.get("description"),
                    body.get("structype_id"),
                )

        if len(rest) == 2 and rest[0] == "geodata":
            gid = int(rest[1])
            if method == "GET":
                return self.geodata[gid]
            if method == "PUT":
                self.geodata[gid].update(body)
                return self.geodata[gid]
            if method == "DELETE":
                partner = self.geodata[gid].get("pair_id")
                del self.geodata[gid]
                if partner is not None and partner in self.geodata:
                    self.geodata[partner]["pair_id"] = None
                return None

        if (
            len(rest) == 3
            and rest[0] == "projects"
            and rest[2] == "tags"
            and method == "GET"
        ):
            pid = int(rest[1])
            counts = {}
            for g in self.geodata.values():
                if g["project_id"] != pid:
                    continue
                for t in g["value"].get("kwargs", {}).get("tags", []):
                    counts[t] = counts.get(t, 0) + 1
            return sorted(
                ({"name": k, "count": v} for k, v in counts.items()),
                key=lambda x: (-x["count"], x["name"]),
            )

        if len(rest) == 4 and rest[0] == "projects" and rest[2:] == ["tags", "rename"]:
            pid = int(rest[1])
            old, new = body["old_name"], body["new_name"]
            n = 0
            for g in self.geodata.values():
                if g["project_id"] != pid:
                    continue
                tags = g["value"].get("kwargs", {}).get("tags", [])
                if old in tags:
                    seen = []
                    for t in (new if t == old else t for t in tags):
                        if t not in seen:
                            seen.append(t)
                    g["value"]["kwargs"]["tags"] = seen
                    n += 1
            return {"updated": n}

        if len(rest) == 4 and rest[0] == "projects" and rest[2:] == ["tags", "delete"]:
            pid = int(rest[1])
            name = body["name"]
            n = 0
            for g in self.geodata.values():
                if g["project_id"] != pid:
                    continue
                tags = g["value"].get("kwargs", {}).get("tags", [])
                if name in tags:
                    g["value"]["kwargs"]["tags"] = [t for t in tags if t != name]
                    n += 1
            return {"updated": n}

        raise AssertionError(f"Unhandled fake route: {method} {path}")


@pytest.fixture
def fake(monkeypatch):
    backend = FakeWebsdb()
    monkeypatch.setattr(_WebsdbTransport, "_send", staticmethod(backend.send))
    return backend


@pytest.fixture
def project(fake):
    return fake.add_project("Erzgebirge")


@pytest.fixture
def populated(fake, project):
    pid = project["id"]
    unit_gneiss = fake.add_unit(pid, "Gneiss")
    unit_schist = fake.add_unit(pid, "Schist")
    st_s1 = fake.add_structype(pid, "S1")
    st_s2 = fake.add_structype(pid, "S2")

    site_a = fake.add_site(pid, "Erzgebirge-N", lon=13.1, lat=50.6)
    site_b = fake.add_site(pid, "Krusne-S", lon=13.5, lat=50.3)

    rock_a1 = fake.add_rock(pid, site_a["id"], "Default", unit_id=unit_gneiss["id"])
    rock_b1 = fake.add_rock(pid, site_b["id"], "Default", unit_id=unit_schist["id"])

    fol1 = fake.add_geodata(
        pid,
        rock_a1["id"],
        {"datatype": "Foliation", "args": [120, 45], "kwargs": {"tags": ["banding"]}},
        structype_id=st_s1["id"],
    )
    fol2 = fake.add_geodata(
        pid,
        rock_b1["id"],
        {
            "datatype": "Foliation",
            "args": [200, 30],
            "kwargs": {"tags": ["crenulation"]},
        },
        structype_id=st_s2["id"],
    )

    return dict(
        project=project,
        unit_gneiss=unit_gneiss,
        unit_schist=unit_schist,
        st_s1=st_s1,
        st_s2=st_s2,
        site_a=site_a,
        site_b=site_b,
        rock_a1=rock_a1,
        rock_b1=rock_b1,
        fol1=fol1,
        fol2=fol2,
    )


# --- project resolution ---


def test_project_resolve_by_id(fake, project):
    session = WebSDBSession(project=project["id"], token="tok")
    assert session.project_id == project["id"]


def test_project_resolve_by_name(fake, project):
    session = WebSDBSession(project=project["name"], token="tok")
    assert session.project_id == project["id"]


def test_project_resolve_by_name_zero_matches(fake):
    with pytest.raises(ProjectResolutionError):
        WebSDBSession(project="Nonexistent", token="tok")


def test_project_resolve_by_name_ambiguous(fake):
    fake.add_project("Dup")
    fake.add_project("Dup")
    with pytest.raises(ProjectResolutionError):
        WebSDBSession(project="Dup", token="tok")


# --- read-only guard ---


def test_read_only_blocks_pure_write_methods(fake, project):
    pid = project["id"]
    site = fake.add_site(pid, "S1")
    rock = fake.add_rock(pid, site["id"], "Default")
    session = WebSDBSession(project=project["name"], token="tok", mode="r")
    before = len(fake.calls)
    with pytest.raises(ReadOnlySessionError):
        session.add_fol(rock["id"], 120, 45)
    assert len(fake.calls) == before  # no HTTP call was made


def test_read_only_blocks_hybrid_get_or_create(fake, project):
    session = WebSDBSession(project=project["name"], token="tok", mode="r")
    with pytest.raises(ReadOnlySessionError):
        session.site("New", lon=1, lat=2)


# --- folset / linset filtering ---


def test_folset_filters_unit_tags_structype(fake, populated):
    session = WebSDBSession(project=populated["project"]["name"], token="tok")

    result = session.folset(unit="Gneiss")
    assert len(result) == 1
    f = result[0]
    assert f._attrs["site"] == "Erzgebirge-N"
    assert f._attrs["rock"] == "Default"
    assert f._attrs["unit"] == "Gneiss"
    assert f._attrs["tags"] == ["banding"]
    assert f._attrs["structype"] == "S1"
    assert f._attrs["id"] == populated["fol1"]["id"]

    assert len(session.folset(tags=["crenulation"])) == 1
    assert len(session.folset(structype="S1")) == 1
    assert len(session.folset()) == 2


def test_folset_site_contains_case_insensitive(fake, populated):
    session = WebSDBSession(project=populated["project"]["name"], token="tok")
    result = session.folset(site_contains="erz")
    assert len(result) == 1
    assert result[0]._attrs["site"] == "Erzgebirge-N"


def test_folset_units_list_filter_no_server_side_param(fake, populated):
    session = WebSDBSession(project=populated["project"]["name"], token="tok")
    before = len(fake.calls)
    result = session.folset(units=["Gneiss", "Schist"])
    assert len(result) == 2
    rock_calls = [c for c in fake.calls[before:] if c[1].endswith("/rocks")]
    assert rock_calls and all("unit_id" not in c[2] for c in rock_calls)


# --- faultset sense mapping ---


def test_faultset_sense_mapping_and_empty_sense_skip(fake, project):
    pid = project["id"]
    site = fake.add_site(pid, "S1")
    rock = fake.add_rock(pid, site["id"], "Default")

    # geometries chosen so each apsg Fault naturally reports a distinct sense_str
    # (N/R = dip-slip, S/D = strike-slip - a function of both rake and rake-axis dip,
    # not directly of the input sense code alone, so a single fixed geometry can't
    # exercise all four).
    sources = [
        Fault(120, 60, 110, 58, "n"),
        Fault(120, 60, 110, 58, "r"),
        Fault(0, 70, 90, 5, "s"),
        Fault(0, 70, 90, 5, "d"),
    ]
    for src in sources:
        fazi, finc = src.fol.geo
        lazi, linc = src.lin.geo
        fake.add_geodata(
            pid,
            rock["id"],
            {
                "datatype": "Fault",
                "args": [fazi, finc, lazi, linc, src.sense_str],
                "kwargs": {"tags": []},
            },
        )
    fake.add_geodata(
        pid,
        rock["id"],
        {"datatype": "Fault", "args": [120, 60, 110, 58, ""], "kwargs": {"tags": []}},
    )

    session = WebSDBSession(project=project["name"], token="tok")
    with pytest.warns(UserWarning, match="empty sense"):
        result = session.faultset()
    assert len(result) == 4
    assert sorted(f.sense_str for f in result) == ["D", "N", "R", "S"]


def test_faultset_misfit_over_20_skipped(fake, project):
    pid = project["id"]
    site = fake.add_site(pid, "S1")
    rock = fake.add_rock(pid, site["id"], "Default")
    fake.add_geodata(
        pid,
        rock["id"],
        {"datatype": "Fault", "args": [0, 0, 0, 80, "N"], "kwargs": {"tags": []}},
    )
    session = WebSDBSession(project=project["name"], token="tok")
    with pytest.warns(UserWarning, match="misfit"):
        result = session.faultset()
    assert len(result) == 0


# --- pairset ---


def test_pairset_joins_and_excludes_filtered_partner(fake, project):
    pid = project["id"]
    site = fake.add_site(pid, "S1")
    rock = fake.add_rock(pid, site["id"], "Default")

    fol1 = fake.add_geodata(
        pid,
        rock["id"],
        {"datatype": "Foliation", "args": [120, 45], "kwargs": {"tags": []}},
    )
    lin1 = fake.add_geodata(
        pid,
        rock["id"],
        {"datatype": "Lineation", "args": [115, 40], "kwargs": {"tags": []}},
    )
    fol1["pair_id"] = lin1["id"]
    lin1["pair_id"] = fol1["id"]

    # unpaired foliation - must be ignored
    fake.add_geodata(
        pid,
        rock["id"],
        {"datatype": "Foliation", "args": [200, 30], "kwargs": {"tags": []}},
    )

    fol2 = fake.add_geodata(
        pid,
        rock["id"],
        {"datatype": "Foliation", "args": [130, 50], "kwargs": {"tags": []}},
    )
    lin2 = fake.add_geodata(
        pid,
        rock["id"],
        {"datatype": "Lineation", "args": [125, 45], "kwargs": {"tags": ["special"]}},
    )
    fol2["pair_id"] = lin2["id"]
    lin2["pair_id"] = fol2["id"]

    session = WebSDBSession(project=project["name"], token="tok")
    assert len(session.pairset()) == 2

    only_special = session.pairset(ltags=["special"])
    assert len(only_special) == 1
    assert only_special[0]._attrs["fol_id"] == fol2["id"]
    assert only_special[0]._attrs["lin_id"] == lin2["id"]


# --- write path ---


def test_add_fol_and_update_and_delete_writeback(fake, project):
    pid = project["id"]
    site = fake.add_site(pid, "S1")
    rock = fake.add_rock(pid, site["id"], "Default")
    session = WebSDBSession(project=project["name"], token="tok", mode="rw")

    fol = session.add_fol(rock["id"], 120, 45, tags=["x"])
    assert fol._attrs["site"] == "S1"
    assert fol._attrs["rock"] == "Default"
    assert fol._attrs["tags"] == ["x"]

    updated = session.update_fol(fol, description="note")
    assert updated._attrs["description"] == "note"

    session.delete_fol(fol)
    assert fol._attrs["id"] not in fake.geodata


def test_update_fol_missing_attrs_raises_clear_error(fake, project):
    session = WebSDBSession(project=project["name"], token="tok", mode="rw")
    bare = Foliation(120, 45)
    with pytest.raises(ValueError, match="not fetched"):
        session.update_fol(bare, description="x")


def test_add_pair_creates_two_rows_and_pairs_in_order(fake, project):
    pid = project["id"]
    site = fake.add_site(pid, "S1")
    rock = fake.add_rock(pid, site["id"], "Default")
    session = WebSDBSession(project=project["name"], token="tok", mode="rw")

    before = len(fake.calls)
    result = session.add_pair(rock["id"], Pair(120, 40, 110, 26))
    calls = [c for c in fake.calls[before:] if c[0] == "POST"]
    assert len(calls) == 3
    assert (
        calls[0][1].endswith("/geodata")
        and calls[0][3]["value"]["datatype"] == "Foliation"
    )
    assert (
        calls[1][1].endswith("/geodata")
        and calls[1][3]["value"]["datatype"] == "Lineation"
    )
    assert calls[2][1] == "/api/geodata/pair"
    assert result._attrs["fol_id"] and result._attrs["lin_id"]


def test_add_fault_uses_sense_str(fake, project):
    pid = project["id"]
    site = fake.add_site(pid, "S1")
    rock = fake.add_rock(pid, site["id"], "Default")
    session = WebSDBSession(project=project["name"], token="tok", mode="rw")

    f = Fault(120, 60, 110, 58, "n")
    result = session.add_fault(rock["id"], f)
    gd = fake.geodata[result._attrs["id"]]
    assert gd["value"]["args"][4] == f.sense_str


def test_site_name_length_validation(fake, project):
    session = WebSDBSession(project=project["name"], token="tok", mode="rw")
    with pytest.raises(ValueError, match="16-character"):
        session.site("this-name-is-way-too-long", lon=1, lat=2)


def test_structype_name_length_validation(fake, project):
    session = WebSDBSession(project=project["name"], token="tok", mode="rw")
    with pytest.raises(ValueError, match="12-character"):
        session.structype("way-too-long-name", description="x")


def test_tag_rename_and_delete(fake, project):
    pid = project["id"]
    site = fake.add_site(pid, "S1")
    rock = fake.add_rock(pid, site["id"], "Default")
    fake.add_geodata(
        pid,
        rock["id"],
        {"datatype": "Foliation", "args": [120, 45], "kwargs": {"tags": ["old"]}},
    )
    session = WebSDBSession(project=project["name"], token="tok", mode="rw")

    result = session.tag_rename("old", "new")
    assert result == {"updated": 1}
    assert session.folset()[0]._attrs["tags"] == ["new"]

    result = session.tag_delete("new")
    assert result == {"updated": 1}
    assert session.folset()[0]._attrs["tags"] == []
