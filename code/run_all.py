"""Reproduce only the numerical figures and tables used in the report body."""

from __future__ import annotations

import argparse
import csv
from importlib.util import find_spec
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from report_outputs import CSV_FILES, FIGURES, LEGACY_FILES, TABLES, expected_paths

ROOT = Path(__file__).resolve().parents[1]


def preflight() -> None:
    """All numerical inputs are built in; only the Python dependencies are needed."""
    packages = {"numpy": "numpy", "scipy": "scipy", "matplotlib": "matplotlib", "skfem": "scikit-fem"}
    missing = [package for module, package in packages.items() if find_spec(module) is None]
    if missing:
        raise SystemExit(
            "Missing Python packages: " + ", ".join(missing)
            + ". Install them in this Python environment with: "
            "python -m pip install -r requirements.txt"
        )
    print("Preflight passed: all benchmark inputs are defined in Python; no report or LaTeX input is required.", flush=True)


def validate_outputs(work: Path) -> None:
    """Reject missing, empty or out-of-scope generated artifacts before publishing."""
    expected = expected_paths()
    actual = {p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file()}
    if actual != expected:
        raise RuntimeError(
            "Output inventory mismatch. Missing: " + str(sorted(expected - actual))
            + "; unexpected: " + str(sorted(actual - expected))
        )
    for name in expected:
        if (work / name).stat().st_size == 0:
            raise RuntimeError("Empty generated file: " + name)
    for name in CSV_FILES:
        with (work / "data" / name).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            records = list(reader)
        if not records or any(None in row or any(v is None for v in row.values()) for row in records):
            raise RuntimeError("Empty or malformed numerical CSV: " + name)


def publish_outputs(work: Path) -> None:
    """Publish a completed run and remove known legacy appendix/duplicate outputs."""
    for name in sorted(expected_paths()):
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        (work / name).replace(target)
    for name in LEGACY_FILES:
        path = ROOT / name
        if path.is_file() or path.is_symlink():
            path.unlink()
    raw_tables = ROOT / "generated" / "raw_tables"
    if raw_tables.is_symlink():
        raw_tables.unlink()
    elif raw_tables.is_dir():
        shutil.rmtree(raw_tables)


def main() -> None:
    preflight()
    from experiments import run_all as run_baseline
    from depth_benchmarks import main as run_depth
    from variable_diffusion import run as run_variable
    from body_tables import write_supplementary_tables

    # Compute and validate in a temporary directory. A numerical failure leaves
    # the previously published outputs in place for diagnosis and comparison.
    with TemporaryDirectory(prefix=".mth321-run-", dir=ROOT) as directory:
        work = Path(directory)
        for name in ("data", "figures", "generated"):
            (work / name).mkdir()
        generated = work / "generated"
        run_baseline(generated)
        for path in list(generated.iterdir()):
            if path.suffix in (".csv", ".json"):
                path.replace(work / "data" / path.name)
            elif path.suffix == ".png":
                path.replace(work / "figures" / path.name)
        run_depth(work)
        run_variable(work)
        write_supplementary_tables(work)
        validate_outputs(work)
        publish_outputs(work)

    print("All checks passed.")
    print(f"Main-body numerical figures: {len(FIGURES)} PNGs in {ROOT / 'figures'}")
    print(f"Main-body numerical tables: {len(TABLES)} LaTeX tables in {ROOT / 'generated'}")
    print(f"Supporting CSVs: {len(CSV_FILES)} in {ROOT / 'data'}")
    print("Output-to-report mapping: README.md (figure/table indexes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the report body's 18 numerical figures and 22 numerical tables."
    )
    parser.add_argument("--check-inputs", action="store_true",
                        help="Check Python dependencies without running experiments.")
    args = parser.parse_args()
    preflight() if args.check_inputs else main()
