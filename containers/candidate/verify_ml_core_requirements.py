"""Check the supplied core wheel's default requirements inside the ML image.

This reads metadata only: it neither imports ML packages nor checks wheel tags,
resolves dependencies, or validates the snapshot's unrelated dependencies.
"""
import argparse
from email.parser import BytesParser
from importlib import metadata
from pathlib import Path
import platform
import sys
import zipfile

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("core_wheel", type=Path)
    args = parser.parse_args()
    try:
        with zipfile.ZipFile(args.core_wheel) as wheel:
            records = [item for item in wheel.infolist() if item.filename.endswith(".dist-info/METADATA")]
            if len(records) != 1:
                raise ValueError("core wheel must contain exactly one .dist-info/METADATA")
            wheel_metadata = BytesParser().parsebytes(wheel.read(records[0]))
        if canonicalize_name(wheel_metadata.get("Name", "")) != "passage" or not wheel_metadata.get("Version"):
            raise ValueError("core wheel METADATA must identify passage and its Version")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"PASSAGE core requirements check failed: {exc}", file=sys.stderr)
        return 1

    try:
        core = metadata.distribution("passage")
    except metadata.PackageNotFoundError:
        print("PASSAGE core requirements check failed: passage is missing", file=sys.stderr)
        return 1

    errors = []
    installed_metadata = core.metadata
    for field in ("Name", "Version", "Requires-Python", "Requires-Dist"):
        if sorted(installed_metadata.get_all(field, ())) != sorted(wheel_metadata.get_all(field, ())):
            errors.append(f"installed passage {field} differs from supplied core wheel METADATA")
    python_version = platform.python_version()
    requires_python = wheel_metadata.get("Requires-Python")
    if requires_python:
        try:
            supported = SpecifierSet(requires_python).contains(python_version, prereleases=True)
        except InvalidSpecifier:
            errors.append(f"invalid Requires-Python: {requires_python}")
        else:
            if not supported:
                errors.append(f"Requires-Python {requires_python}; running Python {python_version}")

    active = 0
    for declared in wheel_metadata.get_all("Requires-Dist", ()):
        try:
            requirement = Requirement(declared)
        except InvalidRequirement:
            errors.append("invalid Requires-Dist entry in supplied core wheel METADATA")
            continue
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        active += 1
        if requirement.url is not None:
            errors.append(f"{requirement.name}: direct URL origin cannot be verified by this metadata check")
            continue
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            errors.append(f"{declared}: required distribution is missing")
            continue
        try:
            satisfies = requirement.specifier.contains(Version(installed), prereleases=True)
        except InvalidVersion:
            errors.append(f"{declared}: invalid installed version {installed!r}")
        else:
            if not satisfies:
                errors.append(f"{declared}: installed {installed} does not satisfy the core requirement")

    if errors:
        print("PASSAGE core requirements check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"PASSAGE core requirements satisfied ({active} active requirements; Python {python_version}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
