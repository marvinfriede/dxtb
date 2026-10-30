"""Architecture guards for the functional SCF authority boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_vendored_xitorch_is_physically_absent() -> None:
    """The retired implementation cannot be rediscovered through imports."""
    source = Path(__file__).parents[2] / "src/dxtb/_src/exlibs/xitorch"
    assert not source.exists()
    assert importlib.util.find_spec("dxtb._src.exlibs.xitorch") is None


def test_production_has_no_xitorch_import_or_legacy_implicit_engine() -> None:
    """SCF has no xitorch; retained batch implicit stays functional."""
    source_root = Path(__file__).parents[2] / "src/dxtb"
    for path in source_root.rglob("*.py"):
        content = path.read_text()
        assert "xitorch" not in content, path
        assert "dxtb._src.exlibs.xitorch" not in content, path
        assert "import xitorch" not in content, path
        assert "SelfConsistentFieldImplicit" not in content, path
        assert "SelfConsistentFieldSingleShot" not in content, path

    batch_implicit = (
        source_root / "_src/calculators/batch_implicit.py"
    ).read_text()
    assert "scf.implicit" not in batch_implicit
    assert "scf.unrolling" not in batch_implicit


def test_legacy_solver_selection_hooks_are_absent() -> None:
    """No iterator or calculator can explicitly select the removed engine."""
    source_root = Path(__file__).parents[2] / "src/dxtb"
    for path in (
        source_root / "_src/scf/iterator.py",
        source_root / "_src/calculators/singlepoint.py",
    ):
        content = path.read_text()
        assert "solve_legacy" not in content
        assert "force_legacy_scf" not in content
