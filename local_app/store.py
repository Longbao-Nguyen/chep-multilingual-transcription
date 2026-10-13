from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

ACTIVE = ("LOADING", "PREPROCESSING", "TRANSCRIBING", "FINALIZING", "UPLOADING", "WAITING_GPU", "REMOTE_RUNNING", "DOWNLOADING")


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / "queue.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, size INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'QUEUED', progress REAL NOT NULL DEFAULT 0,
                    duration REAL, processed REAL NOT NULL DEFAULT 0, error TEXT,
                    created REAL NOT NULL, updated REAL NOT NULL, profile TEXT);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT OR IGNORE INTO settings VALUES ('device', 'auto');
                INSERT OR IGNORE INTO settings VALUES ('language', 'fr');
            """)
            columns = {row['name'] for row in db.execute('PRAGMA table_info(jobs)')}
            if 'config' not in columns:
                db.execute('ALTER TABLE jobs ADD COLUMN config TEXT')
            if 'attempts' not in columns:
                db.execute('ALTER TABLE jobs ADD COLUMN attempts TEXT')
            if 'remote' not in columns:
                db.execute('ALTER TABLE jobs ADD COLUMN remote TEXT')
            if 'removed' not in columns:
                db.execute('ALTER TABLE jobs ADD COLUMN removed INTEGER NOT NULL DEFAULT 0')
            if 'cleanup' not in columns:
                db.execute('ALTER TABLE jobs ADD COLUMN cleanup TEXT')
            # Old records used exactly this French model. Do not invent actual runtime metadata.
            from .catalog import selection
            legacy = selection('fr', 'auto')
            legacy['legacy'] = True
            db.execute('UPDATE jobs SET config=? WHERE config IS NULL', (json.dumps(legacy),))
            db.execute('PRAGMA user_version=3')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def recover(self):
        with self.connect() as db:
            # Resume the same remote IDs after restart; do not upload or submit a second run.
            db.execute("UPDATE jobs SET status='QUEUED', error=NULL, updated=? WHERE removed=0 AND status IN ('UPLOADING','WAITING_GPU','REMOTE_RUNNING','DOWNLOADING')", (time.time(),))
            db.execute("UPDATE jobs SET status='INTERRUPTED', error=?, updated=? WHERE removed=0 AND status IN ('LOADING','PREPROCESSING','TRANSCRIBING','FINALIZING')",
                       ("The application stopped before completion. Choose Retry to process this file again.", time.time()))

    def settings(self):
        with self.connect() as db:
            return dict(db.execute("SELECT key,value FROM settings").fetchall())

    def set_device(self, value: str):
        with self.connect() as db:
            db.execute("UPDATE settings SET value=? WHERE key='device'", (value,))

    def set_selection(self, language, device):
        with self.connect() as db:
            db.executemany('UPDATE settings SET value=? WHERE key=?', [(language, 'language'), (device, 'device')])

    def add(self, job_id: str, name: str, size: int, config=None):
        now = time.time()
        if config is None:
            from .catalog import selection
            settings = self.settings()
            config = selection(settings['language'], settings['device'])
        with self.connect() as db:
            db.execute("INSERT INTO jobs (id,name,size,created,updated,config) VALUES (?,?,?,?,?,?)", (job_id, name, size, now, now, json.dumps(config)))

    def list(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM jobs WHERE removed=0 ORDER BY created, rowid")]

    def get(self, job_id: str, include_removed=False):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?" + ("" if include_removed else " AND removed=0"), (job_id,)).fetchone()
            return dict(row) if row else None

    def pending_cleanup(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM jobs WHERE removed=1 ORDER BY updated, rowid")]

    def mark_removed(self, job_id: str):
        """Hide the job atomically while keeping all IDs needed for cleanup."""
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
            if not row:
                return None
            if not row['removed']:
                try:
                    remote = json.loads(row['remote'] or '{}')
                except (TypeError, ValueError):
                    remote = {'invalid_metadata': True}
                cleanup = {'source': False, 'outputs': False, 'work': False,
                           'remote': not bool(remote) or bool(remote.get('cleaned')),
                           'errors': {}}
                db.execute('UPDATE jobs SET removed=1, cleanup=?, updated=? WHERE id=?',
                           (json.dumps(cleanup), time.time(), job_id))
            return dict(db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())

    def update(self, job_id: str, **values):
        allowed = {"status", "progress", "duration", "processed", "error", "profile", "attempts", "remote", "cleanup"}
        if not values.keys() <= allowed:
            raise ValueError("Unknown job field")
        values["updated"] = time.time()
        with self.connect() as db:
            db.execute(f"UPDATE jobs SET {','.join(k+'=?' for k in values)} WHERE id=?", [*values.values(), job_id])

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE removed=0 AND status='QUEUED' ORDER BY created,rowid LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='LOADING', error=NULL, progress=0, processed=0, updated=? WHERE id=?", (time.time(), row["id"]))
            return dict(row)

    def delete(self, job_id: str):
        with self.connect() as db:
            db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
