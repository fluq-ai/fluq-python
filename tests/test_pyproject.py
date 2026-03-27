"""Tests for pyproject.toml validity and publishability."""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _load() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


class TestPyproject:
    def test_parseable(self) -> None:
        data = _load()
        assert "project" in data
        assert "build-system" in data

    def test_package_name(self) -> None:
        assert _load()["project"]["name"] == "fluq-sdk"

    def test_version(self) -> None:
        version = _load()["project"]["version"]
        parts = version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_required_metadata(self) -> None:
        proj = _load()["project"]
        assert proj["description"]
        assert proj["license"] == "MIT"
        assert ">=3.11" in proj["requires-python"]

    def test_dependencies(self) -> None:
        deps = _load()["project"]["dependencies"]
        assert any("httpx" in d for d in deps)

    def test_dev_dependencies(self) -> None:
        dev = _load()["project"]["optional-dependencies"]["dev"]
        assert any("pytest" in d for d in dev)
        assert any("respx" in d for d in dev)

    def test_build_system(self) -> None:
        bs = _load()["build-system"]
        assert "hatchling" in bs["requires"]
        assert bs["build-backend"] == "hatchling.build"

    def test_urls(self) -> None:
        urls = _load()["project"]["urls"]
        assert "Repository" in urls
        assert "Homepage" in urls

    def test_classifiers(self) -> None:
        classifiers = _load()["project"]["classifiers"]
        assert any("MIT" in c for c in classifiers)
        assert any("Python :: 3" in c for c in classifiers)

    def test_keywords(self) -> None:
        keywords = _load()["project"]["keywords"]
        assert "fluq" in keywords
        assert "sdk" in keywords

    def test_wheel_packages(self) -> None:
        data = _load()
        packages = data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
        assert "fluq" in packages
