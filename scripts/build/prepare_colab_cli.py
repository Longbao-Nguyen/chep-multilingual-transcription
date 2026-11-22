"""Resolve and pin google-colab-cli 0.8.0 and its Windows x64 Python 3.12 wheels."""
from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET_VERSION = "0.8.0"


def _entry(item: dict) -> dict:
    info = item["download_info"]
    url = info["url"]
    name = url.rsplit("/", 1)[-1]
    request = urllib.request.Request(url, method="HEAD", headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=60) as response:
        size = int(response.headers["Content-Length"])
    return {
        "name": name,
        "url": url,
        "size": size,
        "sha256": info["archive_info"]["hashes"]["sha256"],
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "colab-dependencies-report.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--dry-run",
                "--ignore-installed",
                "--only-binary=:all:",
                "--platform",
                "win_amd64",
                "--python-version",
                "312",
                "--implementation",
                "cp",
                "--abi",
                "cp312",
                "--report",
                str(report_path),
                f"google-colab-cli=={TARGET_VERSION}",
            ],
            check=True,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(_entry, report["install"]))
    with urllib.request.urlopen("https://pypi.org/pypi/pip/25.3/json", timeout=60) as response:
        pip_meta = json.load(response)
    wheel = next(f for f in pip_meta["urls"] if f["packagetype"] == "bdist_wheel")
    pip = {
        "name": wheel["filename"],
        "url": wheel["url"],
        "size": wheel["size"],
        "sha256": wheel["digests"]["sha256"],
    }
    output = ROOT / "config" / "colab_cli.json"
    output.write_text(
        json.dumps(
            {
                "version": TARGET_VERSION,
                "python": "3.12",
                "platform": "win_amd64",
                "files": files,
                "pip": pip,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output} ({len(files)} wheels, {sum(f['size'] for f in files) + pip['size']} bytes)")


if __name__ == "__main__":
    main()
