"""Build a standalone Windows runtime and, optionally, the Inno Setup installer.

Run with Python 3.12 on the build machine. End users never run pip or npm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / ".build"
APP = ROOT / "release" / "app"
PYTHON_VERSION = "3.12.10"


def command(args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, cwd=ROOT, **kwargs)


def download(url, path):
    if not path.exists():
        temp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as response, temp.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        os.replace(temp, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iscc", type=Path)
    parser.add_argument("--reuse-runtime", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or os.name != "nt":
        raise SystemExit("Build with 64-bit Python 3.12 on Windows.")
    BUILD.mkdir(exist_ok=True)
    APP.mkdir(parents=True, exist_ok=True)
    command([sys.executable, ROOT / 'scripts' / 'prepare_installer_components.py'])
    command(["npm.cmd", "ci"])
    command(["npm.cmd", "run", "build"])
    runtime = APP / "runtime"
    if not args.reuse_runtime or not (runtime / "python.exe").exists():
        archive = download(f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip", BUILD / "python-embed.zip")
        runtime.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as package:
            package.extractall(runtime)
        # The upstream embeddable distribution is isolated from installed Python/Conda.
        (runtime / "python312._pth").write_text("python312.zip\n.\nLib/site-packages\n..\nimport site\n", encoding="utf-8")
        requirements = ["-r", ROOT / "requirements-lock.txt"] if (ROOT / "requirements-lock.txt").is_file() else ["-r", ROOT / "requirements.txt"]
        command([sys.executable, "-m", "pip", "install", "--only-binary=:all:", "--upgrade", "--target", runtime / "Lib" / "site-packages", *requirements, "--report", BUILD / "runtime-report.json"])
    report_path = BUILD / 'runtime-report.json'
    if not report_path.is_file():
        report_path = APP / 'runtime-dependencies.json'
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report['install'] = [item for item in report['install'] if not item['metadata']['name'].lower().startswith('nvidia-')]
        (BUILD / 'runtime-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        pins = sorted(f"{item['metadata']['name']}=={item['metadata']['version']}" for item in report["install"])
        (ROOT / "requirements-lock.txt").write_text("# Resolved Windows x64 Python 3.12 runtime\n" + "\n".join(pins) + "\n", encoding="utf-8")
        # Keep the dependency download hashes and licenses with the distribution.
        shutil.copy2(BUILD / "runtime-report.json", APP / "runtime-dependencies.json")
    conversion = APP / 'runtime-conversion'
    if conversion.exists():
        if conversion.is_symlink() or conversion.resolve() != APP.resolve() / 'runtime-conversion':
            raise RuntimeError('Unexpected conversion runtime path; refusing to remove it.')
        shutil.rmtree(conversion)
    (APP / 'conversion-dependencies.json').unlink(missing_ok=True)
    packages = runtime / 'Lib' / 'site-packages'
    for obsolete in [packages / 'nvidia', *packages.glob('nvidia_*.dist-info')]:
        if obsolete.exists():
            if obsolete.is_symlink() or obsolete.resolve().parent != packages.resolve():
                raise RuntimeError('Unexpected GPU dependency path.')
            shutil.rmtree(obsolete)
    # CT2's wheel carries a cuDNN dispatch DLL even though CPU execution does not need it.
    # The optional runtime supplies this DLL when GPU execution is requested.
    (packages / 'ctranslate2' / 'cudnn64_9.dll').unlink(missing_ok=True)
    if any(packages.rglob('cublas*.dll')) or any(packages.rglob('cudnn*.dll')):
        raise RuntimeError('GPU libraries must not be bundled in the CPU core.')
    command([runtime / 'python.exe', '-c', "import importlib.util; assert all(importlib.util.find_spec(name) is None for name in ('torch', 'transformers')); print('NO_CONVERSION_DEPENDENCIES')"])
    for folder in ("local_app", "dist", "config"):
        destination = APP / folder
        if destination.exists():
            # A rebuilt frontend has hashed filenames; stale assets must not ship.
            if destination.is_symlink() or destination.resolve() != APP.resolve() / folder:
                raise RuntimeError(f'Unexpected payload path: {destination}')
            shutil.rmtree(destination)
        shutil.copytree(ROOT / folder, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("README.md", "HUONG_DAN_SU_DUNG.md"):
        (APP / name).unlink(missing_ok=True)
    for name in ("launch.py", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(ROOT / name, APP / name)
    redist = download("https://aka.ms/vs/17/release/vc_redist.x64.exe", BUILD / "vc_redist.x64.exe")
    # Verify Microsoft signatures before executing or shipping the prerequisite.
    command(["powershell", "-NoProfile", "-Command",
             "$s=Get-AuthenticodeSignature -LiteralPath '" + str(redist).replace("'", "''") + "'; if($s.Status -ne 'Valid' -or $s.SignerCertificate.Subject -notmatch 'Microsoft'){throw 'Invalid VC runtime signature'}"])
    command([runtime / "python.exe", "-c", "import fastapi, uvicorn, av, ctranslate2, faster_whisper; print('CPU_CORE_IMPORTS_OK')"])
    if args.iscc:
        command([args.iscc.resolve(), "/Q", ROOT / "installer" / "FrenchTranscript.iss"])
        for path in (ROOT / "release").glob("*.exe"):
            digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            path.with_suffix(".exe.sha256").write_text(digest + "  " + path.name + "\n")
    print(f"Standalone application: {APP}")


if __name__ == "__main__":
    main()
