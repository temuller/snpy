"""Shared pytest configuration for the SNooPy test suite.

The test modules refer to their data files by bare relative names
(e.g. ``snpy.get_sn('SN2006ax.txt')``) and some of them write files into
the current working directory (``table.dat``, ``temp.snpy``,
``SN2006ax*_lc_B_data.dat``).  This used to require running ``pytest``
from inside the ``tests/`` directory; running it any other way silently
fell through to the SQL backend and blocked on a password prompt.

The fixtures below make the suite independent of the invoking directory:

* every test runs inside an isolated temporary directory (``tmp_path``)
  into which the static data files have been copied, so relative reads
  resolve and writes never touch the source tree;
* ``SNPY_TESTS_DIR`` points at the directory holding the real data files.

An editable install (``pip install -e .``) is still required so that
``import snpy`` picks up the working tree rather than a stale copy in
site-packages.
"""
import os
import shutil

import pytest

# Directory containing this file == the test-data directory.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Files (and directories) that the tests read via relative paths.  They are
# copied into each test's temporary working directory.
DATA_FILES = ["SN2006ax.txt"]


@pytest.fixture(autouse=True)
def _in_tmp_workdir(tmp_path, monkeypatch):
    """Run every test in an isolated dir that contains the data files."""
    for name in DATA_FILES:
        src = os.path.join(TESTS_DIR, name)
        if os.path.exists(src):
            dst = tmp_path / name
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy(src, dst)
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture(autouse=True)
def _offline_ebvgal(monkeypatch):
    """Make Milky-Way E(B-V) look-ups deterministic and offline-safe.

    ``sn.getEBVgal`` reads local SFD dust maps, which the test machine may
    not have.  Pin the value for the test field (SN2006ax) to the SF11 value
    (tests/test_sfd_dust.py checks the maps themselves when available).
    """
    from snpy.utils import sfd_dust

    def _fixed_get_ebv(ra, dec, calibration="SF11", mapdir=None):
        return 0.041

    monkeypatch.setattr(sfd_dust, "get_ebv", _fixed_get_ebv)
    yield


@pytest.fixture
def datadir():
    """Absolute path to the directory holding the SNooPy test data."""
    return TESTS_DIR


@pytest.fixture
def snobj():
    """A freshly imported SN2006ax object (data-relative, cwd-independent).

    ``get_sn`` is given the absolute path so that the load never depends
    on the current working directory and never falls through to the SQL
    backend.
    """
    import snpy
    return snpy.get_sn(os.path.join(TESTS_DIR, "SN2006ax.txt"))
