import warnings

import pytest

from apsg.database import SDBSession
from apsg.feature import Foliation, Lineation, Pair


@pytest.fixture
def db(tmp_path):
    db = SDBSession(str(tmp_path / "test.sdb"), create=True)
    S = db.structype("S", planar=1)
    L = db.structype("L", planar=0)
    site = db.site("A", x_coord=0, y_coord=0)
    db.add_pair(site, S, L, Pair(140, 30, 110, 26))
    # lineation close to the pole of the foliation -> too big misfit, skipped
    planar = db.add_structdata(site, S, 140, 30)
    linear = db.add_structdata(site, L, 320, 55)
    db.attach(planar, linear)
    db.commit()
    yield db
    db.close()


class TestGetPairs:
    def test_getpairs_skips_big_misfit(self, db):
        with pytest.warns(UserWarning, match="Too big misfit"):
            assert len(db.getpairs("S", "L")) == 1

    def test_warning_filters_preserved(self, db):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", UserWarning)
            before = list(warnings.filters)
            db.getpairs("S", "L")
            assert warnings.filters == before

    def test_no_getfaults(self, db):
        # SDB stores no sense of movement, faults are stored as pairs
        assert not hasattr(db, "getfaults")


class TestGetPairsReturnsAttachedPairsOnly:
    @pytest.fixture
    def rich_db(self, tmp_path):
        db = SDBSession(str(tmp_path / "rich.sdb"), create=True)
        S = db.structype("S", planar=1)
        L = db.structype("L", planar=0)
        S2 = db.structype("S2", planar=1)
        L2 = db.structype("L2", planar=0)
        a = db.site("A", x_coord=0, y_coord=0)
        b = db.site("B", x_coord=1, y_coord=1)
        tags = [db.tag(name, description=name) for name in ("t1", "t2", "t3")]
        attached = db.add_pair(a, S, L, Pair(140, 30, 110, 26))
        # several tags on both measurements must not duplicate the pair
        attached.planar.tags.extend(tags[:2])
        attached.linear.tags.extend([tags[0], tags[2]])
        db.add_pair(b, S, L, Pair(200, 40, 230, 37))
        db.add_pair(a, S2, L2, Pair(10, 50, 40, 47))  # other structure types
        db.add_structdata(a, S, 100, 20)  # not attached
        db.add_structdata(b, L, 5, 5)  # not attached
        db.commit()
        yield db
        db.close()

    def test_only_attached_pairs_of_the_types(self, rich_db):
        pairs = rich_db.getpairs("S", "L")
        assert sorted(p.fol.geo for p in pairs) == sorted(
            [Pair(140, 30, 110, 26).fol.geo, Pair(200, 40, 230, 37).fol.geo]
        )
        assert len(rich_db.getpairs("S2", "L2")) == 1
        assert len(rich_db.getpairs("S", "L2")) == 0

    def test_filters(self, rich_db):
        assert len(rich_db.getpairs("S", "L", site={"name": "B"})) == 1
        assert len(rich_db.getpairs("S", "L", ptag={"name": "t2"})) == 1
        assert len(rich_db.getpairs("S", "L", ltag={"name": "t3"})) == 1
        assert len(rich_db.getpairs("S", "L", ltag={"name": "t2"})) == 0

    def test_planar_and_linear_types_required(self, rich_db):
        with pytest.raises(ValueError, match="ptype must be a planar"):
            rich_db.getpairs("L", "S")
        with pytest.raises(ValueError, match="ltype must be a linear"):
            rich_db.getpairs("S", "S2")


class TestInputValidation:
    def test_unknown_structype(self, db):
        with pytest.raises(ValueError, match="no structure"):
            db.getset("X")
        with pytest.raises(ValueError, match="no structure"):
            db.getpairs("S", "X")

    def test_add_fol_wrong_type(self, db):
        site = db.site("A")
        with pytest.raises(TypeError):
            db.add_fol(site, db.structype("S"), Lineation(110, 26))

    def test_add_fol_linear_structype(self, db):
        site = db.site("A")
        with pytest.raises(ValueError):
            db.add_fol(site, db.structype("L"), Foliation(140, 30))

    def test_add_pair_wrong_type(self, db):
        site = db.site("A")
        with pytest.raises(TypeError):
            db.add_pair(site, db.structype("S"), db.structype("L"), Foliation(1, 2))


class TestReadMethods:
    def test_structdata(self, db):
        assert len(db.structdata("S")) == 2
        assert len(db.structdata(db.structype("L"))) == 2

    def test_getset(self, db):
        fols = db.getset("S", store=["site"])
        assert len(fols) == 2
        assert isinstance(fols[0], Foliation)
        assert fols[0]._attrs == {"site": "A"}
        assert isinstance(db.getset("L")[0], Lineation)

    def test_getset_filter(self, db):
        assert len(db.getset("S", site={"name": "A"})) == 2
        assert len(db.getset("S", site={"name": "B"})) == 0

    def test_df(self, db):
        df = db.df("S")
        assert len(df) == 2
        assert list(df["structure"]) == ["S", "S"]
        assert len(db.df("L", apsg=True)["L"]) == 2

    def test_getpairs_filter(self, db):
        assert len(db.getpairs("S", "L", site={"name": "B"})) == 0
        with pytest.warns(UserWarning, match="Too big misfit"):
            pairs = db.getpairs(db.structype("S"), db.structype("L"))
        assert pairs.name == "S-L"
