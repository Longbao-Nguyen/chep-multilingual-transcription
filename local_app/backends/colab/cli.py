"""Optional, isolated Google Colab CLI manager. Never bundles CLI or credentials."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import webbrowser
from pathlib import Path
from typing import Any

from ...config import ROOT, atomic_json

FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
JSON_PREFIX = 'CHEP_COLAB_JSON:'
AUTH_URL_PREFIX = 'CHEP_COLAB_AUTH_URL:'


class ColabStopped(Exception):
    """Raised when a remote Colab operation is cancelled or stopping."""


def safe_error(output: str) -> str:
    """Never expose OAuth URLs, tokens, or raw Google auth payloads in logs or UI."""
    text = output.lower()
    if any(tok in text for tok in ('412', 'too many assignments', 'too many active sessions', 'allocation refused', 'capacity', 'quota')):
        return 'Colab GPU capacity or quota limit reached. Close unused Colab sessions or try again later.'
    if any(tok in text for tok in ('401', 'unauthenticated', 'invalid_grant', 'invalid oauth', 'invalid authorization code')):
        return 'Google Colab authentication failed or expired. Sign in again.'
    if any(tok in text for tok in ('400', 'entitlement', 'rejected accelerator', 'not have quota')):
        return 'Your Google Colab account cannot currently allocate a T4 GPU runtime.'
    if any(tok in text for tok in ('403', 'forbidden', 'permission')):
        return 'Google Colab denied access. Check your signed-in Google account and try again.'
    if any(tok in text for tok in ('sign in to google colab', 'no valid credentials', 'token not found')):
        return 'Sign in to Google Colab before starting online transcription.'
    return 'The Google Colab request failed. Check your connection and Colab account, then retry.'


from .helpers import LOGIN_HELPER, LOGOUT_HELPER, PASSIVE_AUTH_HELPER
class ColabCLI:
    """Manage optional google-colab-cli wheel installation, OAuth sign-in, and isolated calls."""

    def __init__(self, root: Path) -> None:
        self.root = root / 'tools' / 'colab'
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog = json.loads((ROOT / 'config' / 'colab_cli.json').read_text(encoding='utf-8'))
        self.packages = self.root / ('cli-' + self.catalog['version'] + '-pip')
        self.fallback_token_path = self.root / 'token.json'
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self._login_proc: subprocess.Popen[str] | None = None
        self.state: dict[str, Any] = {
            'status': 'idle',
            'username': None,
            'message': '',
            'progress': 0.0,
            'error': None,
            'auth_url': None,
        }

    def _package_dir(self) -> Path | None:
        if (self.packages / 'chep-ready.json').is_file():
            return self.packages
        override = os.environ.get('CHEP_COLAB_CLI_PACKAGES')
        if override and Path(override).is_dir():
            return Path(override)
        return None

    @property
    def installed(self) -> bool:
        return self._package_dir() is not None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                **self.state,
                'installed': self.installed,
                'download_bytes': sum(f['size'] for f in self.catalog['files']) + self.catalog['pip']['size'],
            }

    @property
    def python(self) -> str:
        console = Path(sys.executable).with_name('python.exe')
        return str(console) if os.name == 'nt' and console.is_file() else sys.executable

    def _helper_command(self, helper_code: str) -> list[str]:
        pkg_dir = self._package_dir()
        if not pkg_dir:
            raise ValueError('Download Colab CLI first.')
        bootstrap = (
            'import site, sys; '
            'p = sys.argv.pop(1); '
            'site.addsitedir(p); '
            'sys.path.insert(0, p); '
            'exec(sys.argv.pop(1))'
        )
        return [self.python, '-c', bootstrap, str(pkg_dir), helper_code]

    def run_helper(
        self,
        helper_code: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: int = 120,
        stopped: Any = None,
    ) -> dict[str, Any]:
        """Run an isolated helper subprocess inside the installed Colab CLI packages."""
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}
        command = self._helper_command(helper_code)
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            creationflags=FLAGS,
        )
        input_json = json.dumps({**(payload or {}), 'fallback_token': str(self.fallback_token_path)})
        started = time.monotonic()
        output = ''
        try:
            assert process.stdin is not None
            process.stdin.write(input_json)
            process.stdin.close()
            while True:
                if self.stop.is_set() or (stopped and stopped()):
                    raise ColabStopped()
                if time.monotonic() - started > timeout:
                    raise RuntimeError('Google Colab request timed out. Check your connection and retry.')
                try:
                    output, _ = process.communicate(timeout=0.4)
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
                try:
                    process.communicate(timeout=2)
                except Exception:
                    pass

        for line in reversed(output.splitlines()):
            if line.startswith(JSON_PREFIX):
                data = json.loads(line[len(JSON_PREFIX):])
                if process.returncode and not data.get('error'):
                    data['error'] = safe_error(output)
                return data
        if process.returncode:
            raise RuntimeError(safe_error(output))
        raise RuntimeError('Google Colab helper did not return a valid response.')

    def identity(self) -> str:
        """Return signed-in Google email or raise ValueError if authentication is required."""
        data = self.run_helper(PASSIVE_AUTH_HELPER, timeout=60)
        if not data.get('authenticated') or not data.get('email'):
            raise ValueError('Sign in to Google Colab before starting online transcription.')
        return str(data['email'])

    def action(self, action: str) -> None:
        if action not in ('download', 'login', 'check', 'cancel', 'logout'):
            raise ValueError('Unknown Colab action.')
        if action == 'cancel':
            self.cancel_login()
            return
        if action == 'login' and not self.installed:
            raise ValueError('Download Colab CLI first, then choose Authenticate Colab CLI.')
        if action == 'logout':
            self.logout()
            return
        with self.lock:
            if self.thread and self.thread.is_alive():
                if self.state.get('status') == 'awaiting_code' and action == 'login':
                    return
                raise ValueError('A Colab setup action is already running.')
            self.state.update(
                status='downloading' if action == 'download' else 'connecting',
                error=None,
                auth_url=None,
                message='Downloading Colab CLI...' if action == 'download' else 'Connecting to Google Colab...',
            )
            self.thread = threading.Thread(target=self._action, args=(action,), daemon=True)
            self.thread.start()

    def _action(self, action: str) -> None:
        try:
            if action == 'download':
                self.install()
            if action == 'login':
                self._interactive_login()
                return
            email = self.identity()
            with self.lock:
                self.state.update(
                    username=email,
                    status='ready',
                    message='Connected as ' + email,
                    error=None,
                    auth_url=None,
                )
        except ColabStopped:
            with self.lock:
                self.state.update(
                    username=None,
                    status='sign_in_required' if self.installed else 'idle',
                    message='',
                    error=None,
                    auth_url=None,
                )
        except Exception as exc:
            with self.lock:
                if str(exc).startswith('Sign in to Google Colab') and self.installed:
                    self.state.update(
                        username=None,
                        status='sign_in_required',
                        message='Authentication required. Choose Authenticate Colab CLI.',
                        error=None,
                        auth_url=None,
                    )
                else:
                    self.state.update(
                        username=None,
                        status='error',
                        message='Authentication did not complete. Try again.',
                        error=str(exc),
                        auth_url=None,
                    )

    def _interactive_login(self) -> None:
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1'}
        command = self._helper_command(LOGIN_HELPER)
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            creationflags=FLAGS,
        )
        with self.lock:
            self._login_proc = process

        lines_q: queue.Queue[str | None] = queue.Queue()

        def reader() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                lines_q.put(line)
            lines_q.put(None)

        threading.Thread(target=reader, daemon=True).start()
        captured: list[str] = []
        auth_url: str | None = None
        started = time.monotonic()

        try:
            assert process.stdin is not None
            process.stdin.write(json.dumps({'fallback_token': str(self.fallback_token_path)}) + '\n')
            process.stdin.flush()

            # Step 1: Wait up to 30s for OAuth URL.
            while time.monotonic() - started < 30:
                if self.stop.is_set():
                    raise ColabStopped()
                try:
                    line = lines_q.get(timeout=0.25)
                except queue.Empty:
                    if process.poll() is not None:
                        break
                    continue
                if line is None:
                    break
                captured.append(line)
                if line.startswith(AUTH_URL_PREFIX):
                    auth_url = line[len(AUTH_URL_PREFIX):].strip()
                    break

            if not auth_url:
                raise RuntimeError(safe_error(''.join(captured)))

            with self.lock:
                self.state.update(
                    username=None,
                    status='awaiting_code',
                    auth_url=auth_url,
                    error=None,
                    message='Sign in to Google in your browser, copy the authorization code, and paste it below.',
                )
            try:
                webbrowser.open(auth_url)
            except Exception:
                pass

            # Step 2: Wait up to 600s for user code submission + token exchange.
            result_data: dict[str, Any] | None = None
            while time.monotonic() - started < 600:
                if self.stop.is_set():
                    raise ColabStopped()
                try:
                    line = lines_q.get(timeout=0.3)
                except queue.Empty:
                    continue
                if line is None:
                    if process.poll() is None:
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                    break
                captured.append(line)
                if line.startswith(JSON_PREFIX):
                    result_data = json.loads(line[len(JSON_PREFIX):])
                    if process.poll() is None:
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                    break
            else:
                raise RuntimeError('Google sign-in timed out. Choose Authenticate Colab CLI to try again.')

            if process.returncode == 0 and result_data and result_data.get('ok'):
                email = str(result_data['email'])
                with self.lock:
                    self.state.update(
                        username=email,
                        status='ready',
                        message='Connected as ' + email,
                        error=None,
                        auth_url=None,
                    )
                return

            msg = (result_data or {}).get('error') or safe_error(''.join(captured))
            raise RuntimeError(msg)
        except ColabStopped:
            with self.lock:
                self.state.update(
                    username=None,
                    status='sign_in_required' if self.installed else 'idle',
                    message='',
                    error=None,
                    auth_url=None,
                )
        except Exception as exc:
            with self.lock:
                # Only preserve non-fatal error if not explicitly cancelled.
                if self.state.get('status') != 'sign_in_required':
                    self.state.update(
                        username=None,
                        status='error',
                        message='Authentication did not complete. Try again.',
                        error=str(exc),
                        auth_url=None,
                    )
        finally:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
                try:
                    process.communicate(timeout=2)
                except Exception:
                    pass
            with self.lock:
                if self._login_proc is process:
                    self._login_proc = None

    def submit_code(self, code: str) -> None:
        cleaned = (code or '').strip()
        if not cleaned:
            raise ValueError('Paste the authorization code from Google before submitting.')
        with self.lock:
            proc = self._login_proc
            if self.state.get('status') != 'awaiting_code' or proc is None or proc.poll() is not None or not proc.stdin:
                raise ValueError('No Google sign-in request is currently waiting for a code.')
            self.state.update(
                status='connecting',
                message='Completing Google sign-in...',
                error=None,
                auth_url=None,
            )
            try:
                proc.stdin.write(cleaned + '\n')
                proc.stdin.flush()
                proc.stdin.close()
            except OSError as exc:
                raise ValueError('The sign-in process closed before receiving the code. Retry authentication.') from exc

    def cancel_login(self) -> None:
        with self.lock:
            proc = self._login_proc
            self.state.update(
                username=None,
                status='sign_in_required' if self.installed else 'idle',
                message='',
                error=None,
                auth_url=None,
            )
            if proc and proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass

    def logout(self) -> None:
        self.cancel_login()
        if self.installed:
            try:
                self.run_helper(LOGOUT_HELPER, timeout=30)
            except Exception:
                pass
        self.fallback_token_path.unlink(missing_ok=True)
        with self.lock:
            self.state.update(
                username=None,
                status='sign_in_required' if self.installed else 'idle',
                message='Signed out of Google Colab.' if self.installed else '',
                error=None,
                auth_url=None,
            )

    def _run_pip_install(self, command: list[str]) -> None:
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            creationflags=FLAGS,
        )
        started = time.monotonic()
        output = ''
        try:
            while True:
                if self.stop.is_set():
                    raise ColabStopped()
                if time.monotonic() - started > 600:
                    raise RuntimeError('Colab CLI installation timed out. Retry the download.')
                try:
                    output, _ = process.communicate(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise RuntimeError('Colab CLI installation failed.')
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def install(self) -> None:
        if sys.version_info[:2] != (3, 12) or os.name != 'nt':
            raise ValueError('The optional CLI package requires Windows x64 and Python 3.12.')
        target = self.root / ('install-' + uuid.uuid4().hex)
        target.mkdir(exist_ok=True)
        wheels = self.root / 'wheels'
        wheels.mkdir(exist_ok=True)
        entries = [self.catalog['pip'], *self.catalog['files']]
        total = sum(f['size'] for f in entries)
        done = 0
        try:
            for entry in entries:
                if self.stop.is_set():
                    raise ColabStopped()
                archive = wheels / entry['name']
                digest = hashlib.sha256()
                count = 0
                with urllib.request.urlopen(entry['url'], timeout=60) as response, archive.open('wb') as out:
                    while data := response.read(256 * 1024):
                        if self.stop.is_set():
                            raise ColabStopped()
                        out.write(data)
                        digest.update(data)
                        count += len(data)
                        if count > entry['size']:
                            raise ValueError('Colab CLI package size mismatch.')
                        with self.lock:
                            self.state['progress'] = (done + count) / total
                if count != entry['size'] or digest.hexdigest() != entry['sha256']:
                    raise ValueError('Colab CLI download verification failed. Retry the download.')
                done += count

            with self.lock:
                self.state['message'] = 'Installing Colab CLI with pip...'
            bootstrap = (
                'import sys, runpy; '
                'sys.path.insert(0, sys.argv.pop(1)); '
                'sys.argv[0] = "pip"; '
                'runpy.run_module("pip", run_name="__main__")'
            )
            self._run_pip_install([
                self.python, '-c', bootstrap, str(wheels / self.catalog['pip']['name']),
                '--isolated', 'install', '--disable-pip-version-check', '--no-warn-script-location',
                '--no-index', '--find-links', str(wheels), '--only-binary=:all:',
                '--ignore-installed', '--no-compile', '--target', str(target),
                'google-colab-cli==' + self.catalog['version'],
            ])
            if self.stop.is_set():
                raise ColabStopped()
            if self.packages.exists():
                if self.packages.resolve().parent != self.root.resolve() or self.packages.is_symlink():
                    raise ValueError('Unexpected CLI installation path.')
                shutil.rmtree(self.packages)
            os.replace(target, self.packages)
            atomic_json(
                self.packages / 'chep-ready.json',
                {'version': self.catalog['version'], 'installer': 'pip'},
            )
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise

    def close(self) -> None:
        self.stop.set()
        self.cancel_login()
        if self.thread:
            self.thread.join(timeout=3)
