"""
Download the pinned front-end libraries into static/vendor/.

Each npm package tarball is checked against the sha512 integrity hash that the
npm registry publishes for that exact version before any file is taken from
it, and only the listed files are extracted. Run after changing a version:

    python scripts/vendor_assets.py
"""

import base64
import hashlib
import io
import json
import os
import sys
import tarfile
import urllib.request

# package, version, file inside the tarball, destination in static/vendor/
ASSETS = [
    ("htmx.org", "1.9.10", "package/dist/htmx.min.js", "htmx.min.js"),
    ("d3", "7.9.2", "package/dist/d3.min.js", "d3.min.js"),
]

REGISTRY = "https://registry.npmjs.org"
VENDOR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "vendor")


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        data: bytes = response.read()
    return data


def verified_tarball(package: str, version: str) -> bytes:
    meta = json.loads(fetch(f"{REGISTRY}/{package}/{version}"))
    algorithm, expected_b64 = meta["dist"]["integrity"].split("-", 1)
    if algorithm != "sha512":
        sys.exit(f"{package}@{version}: unexpected integrity algorithm {algorithm}")
    data = fetch(meta["dist"]["tarball"])
    actual_b64 = base64.b64encode(hashlib.sha512(data).digest()).decode()
    if actual_b64 != expected_b64:
        sys.exit(f"{package}@{version}: tarball does not match the registry's integrity hash")
    return data


def main() -> None:
    os.makedirs(VENDOR_DIR, exist_ok=True)
    for package, version, member, destination in ASSETS:
        data = verified_tarball(package, version)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            extracted = tar.extractfile(member)
            if extracted is None:
                sys.exit(f"{package}@{version}: {member} not found in the tarball")
            content = extracted.read()
        header = f"/* {package}@{version} ({member.removeprefix('package/')}), vendored by scripts/vendor_assets.py */\n"
        with open(os.path.join(VENDOR_DIR, destination), "wb") as f:
            f.write(header.encode() + content)
        print(f"{package}@{version} -> static/vendor/{destination} ({len(content):,} bytes, integrity verified)")


if __name__ == "__main__":
    main()
