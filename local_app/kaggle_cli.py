"""Optional, isolated Kaggle CLI. No pip, credentials or GPU libraries are bundled."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from .config import ROOT, atomic_json

FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


class KaggleStopped(Exception):
    pass


def safe_error(output):
    """Auth command output can contain OAuth URLs. Never expose it in logs/UI."""
    text = output.lower()
    if '429' in text or 'quota' in text or 'too many' in text:
        return 'Kaggle quota or capacity is exhausted. Check your account and try later.'
    if '401' in text or 'unauthenticated' in text:
        return 'Kaggle authentication expired. Sign in again.'
    if '403' in text or 'permission' in text or 'verify' in text:
        return 'Kaggle denied access. Check account verification, GPU access and the signed-in account.'
    if 'kaggle auth login' in text or 'could not find kaggle.json' in text:
        return 'Sign in to Kaggle before starting online transcription.'
    return 'The Kaggle request failed. Check your connection and Kaggle account, then retry.'


class KaggleCLI:
    def __init__(self, root):
        self.root = root / 'tools' / 'kaggle'
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog = json.loads((ROOT / 'config/kaggle_cli.json').read_text(encoding='utf-8'))
        self.packages = self.root / ('cli-' + self.catalog['version'] + '-pip')
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.external = shutil.which('kaggle')
        if not self.external:
            path = Path(sys.executable).parent / 'Scripts/kaggle.exe'
            if path.is_file():
                self.external = str(path)
        self.state = {'status': 'idle', 'username': None, 'message': '', 'progress': 0, 'error': None}

    @property
    def installed(self):
        return (self.packages / 'chep-ready.json').is_file() or bool(self.external)

    def snapshot(self):
        with self.lock:
            return {**self.state, 'installed': self.installed,
                    'download_bytes': sum(f['size'] for f in self.catalog['files']) + self.catalog['pip']['size']}

    @property
    def python(self):
        # python.exe has standard streams even when Chep is launched with pythonw.
        console = Path(sys.executable).with_name('python.exe')
        return str(console) if os.name == 'nt' and console.is_file() else sys.executable

    def command(self, args):
        if (self.packages / 'chep-ready.json').is_file():
            code = 'import sys, site; site.addsitedir(sys.argv[1]); sys.path.insert(0, sys.argv.pop(1)); sys.argv[0] = "kaggle"; from kaggle.cli import main; main()'
            return [self.python, '-c', code, str(self.packages), *args]
        if self.external:
            return [self.external, *args]
        raise ValueError('Download Kaggle CLI first.')

    def run(self, args, timeout=120, stopped=None, check=True):
        return self._run_command(self.command(args), timeout, stopped, check)

    def _run_command(self, command, timeout=120, stopped=None, check=True):
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', env=env, creationflags=FLAGS)
        started = time.monotonic()
        try:
            while True:
                if self.stop.is_set() or (stopped and stopped()):
                    raise KaggleStopped()
                if time.monotonic() - started > timeout:
                    raise RuntimeError('Kaggle request timed out. Retry to check the saved remote session.')
                try:
                    output, _ = process.communicate(timeout=.5)
                    break
                except subprocess.TimeoutExpired:
                    pass
            if check and process.returncode:
                raise RuntimeError(safe_error(output))
            return output, process.returncode
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()

    def identity(self):
        version, _ = self.run(['--version'])
        match = re.search(r'Kaggle CLI (\d+)\.(\d+)\.(\d+)', version)
        if not match or tuple(map(int, match.groups())) < (2, 2, 4):
            raise ValueError('Download the supported Kaggle CLI version using Repair Kaggle CLI.')
        # config view exposes the username and paths only; never run token-printing commands.
        output, _ = self.run(['config', 'view'])
        match = re.search(r'^\s*(?:[-*]\s*)?username\s*:\s*([a-zA-Z0-9_-]+)\s*$', output, re.M | re.I)
        if not match:
            raise ValueError('Sign in to Kaggle before starting online transcription.')
        return match.group(1)

    def action(self, action):
        if action not in ('download', 'login', 'check'):
            raise ValueError('Unknown Kaggle action.')
        if action == 'login' and not self.installed:
            raise ValueError('Download Kaggle CLI first, then choose Authenticate Kaggle CLI.')
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError('A Kaggle setup action is already running.')
            self.state.update(status='downloading' if action == 'download' else 'connecting', error=None,
                              message='Downloading Kaggle CLI...' if action == 'download' else 'Connecting to Kaggle...')
            self.thread = threading.Thread(target=self._action, args=(action,), daemon=True)
            self.thread.start()

    def _action(self, action):
        try:
            if action == 'download':
                self.install()
            if action == 'login':
                with self.lock:
                    self.state.update(username=None, message='In the browser, sign in to Kaggle if needed, then approve CLI access. Waiting for approval...')
                # Use the official CLI login; an existing login exits with code 1.
                output, code = self.run(['auth', 'login'], timeout=600, check=False)
                if code and 'already logged-in to Kaggle' not in output:
                    raise RuntimeError(safe_error(output))
            username = self.identity()
            with self.lock:
                self.state.update(username=username, status='ready', message='Connected as ' + username, error=None)
        except KaggleStopped:
            pass
        except Exception as exc:
            with self.lock:
                if str(exc).startswith('Sign in to Kaggle') and self.installed:
                    self.state.update(username=None, status='sign_in_required',
                        message='Authentication required. Choose Authenticate Kaggle CLI.', error=None)
                else:
                    self.state.update(username=None, status='error', message='Authentication did not complete. Try again.', error=str(exc))

    def install(self):
        if sys.version_info[:2] != (3, 12) or os.name != 'nt':
            raise ValueError('The optional CLI package requires Windows x64 and Python 3.12.')
        target = self.root / ('install-' + uuid.uuid4().hex)
        target.mkdir(exist_ok=True)
        wheels = self.root / 'wheels'
        wheels.mkdir(exist_ok=True)
        entries = [self.catalog['pip'], *self.catalog['files']]
        total = sum(f['size'] for f in entries)
        done = 0
        for entry in entries:
            if self.stop.is_set():
                raise KaggleStopped()
            archive = wheels / entry['name']
            digest = hashlib.sha256()
            count = 0
            with urllib.request.urlopen(entry['url'], timeout=30) as response, archive.open('wb') as out:
                while data := response.read(256 * 1024):
                    if self.stop.is_set():
                        raise KaggleStopped()
                    out.write(data)
                    digest.update(data)
                    count += len(data)
                    if count > entry['size']:
                        raise ValueError('Kaggle CLI package size mismatch.')
                    with self.lock:
                        self.state['progress'] = (done + count) / total
            if count != entry['size'] or digest.hexdigest() != entry['sha256']:
                raise ValueError('Kaggle CLI download verification failed. Retry the download.')
            done += count
        with self.lock:
            self.state['message'] = 'Installing Kaggle CLI with pip...'
        # Run the official pip wheel using app Python. Resolve only the verified
        # PyPI wheels, and install outside the read-only app/core environment.
        bootstrap = 'import sys, runpy; sys.path.insert(0, sys.argv.pop(1)); sys.argv[0] = "pip"; runpy.run_module("pip", run_name="__main__")'
        self._run_command([self.python, '-c', bootstrap, str(wheels / self.catalog['pip']['name']),
            '--isolated', 'install', '--disable-pip-version-check', '--no-warn-script-location',
            '--no-index', '--find-links', str(wheels), '--only-binary=:all:',
            '--ignore-installed', '--no-compile', '--target', str(target),
            'kaggle==' + self.catalog['version']], timeout=600)
        if self.stop.is_set():
            raise KaggleStopped()
        if self.packages.exists():
            if self.packages.resolve().parent != self.root.resolve() or self.packages.is_symlink():
                raise ValueError('Unexpected CLI installation path.')
            shutil.rmtree(self.packages)
        os.replace(target, self.packages)
        atomic_json(self.packages / 'chep-ready.json', {'version': self.catalog['version'], 'installer': 'pip'})

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
