"""Transactional call quota and resumable job journal. SQLite is authoritative."""

from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3

from .core import canonical


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self, path: Path, limit: int):
        if limit < 1:
            raise ValueError("Call cap must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, stage TEXT NOT NULL, spec TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'planned', result TEXT,
                    started TEXT, finished TEXT);
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    started TEXT NOT NULL);
            """)
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('call_cap', ?)", (str(limit),))
            stored = int(db.execute("SELECT value FROM metadata WHERE key='call_cap'").fetchone()[0])
            if stored != limit:
                raise ValueError("Existing campaign call cap is immutable; cannot silently increase it")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def register(self, stage: str, spec: dict) -> str:
        text = canonical(spec)
        key = hashlib.sha256(canonical([stage, spec]).encode()).hexdigest()
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO jobs(id,stage,spec) VALUES (?,?,?)", (key, stage, text))
        return key

    def claim(self, key: str) -> bool:
        # Reservation is persisted before sending. Interrupted/uncertain calls stay counted.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM jobs WHERE id=?", (key,)).fetchone()
            if row is None:
                raise KeyError(key)
            if row[0] != "planned":
                return False
            used = db.execute("SELECT count(*) FROM attempts").fetchone()[0]
            cap = int(db.execute("SELECT value FROM metadata WHERE key='call_cap'").fetchone()[0])
            if used >= cap:
                return False
            db.execute("INSERT INTO attempts(job_id,started) VALUES (?,?)", (key, now()))
            db.execute("UPDATE jobs SET status='running',started=? WHERE id=?", (now(), key))
        return True

    def finish(self, key: str, result: dict):
        status = "ok" if result.get("status") == "ok" else "error"
        with self.connect() as db:
            changed = db.execute("UPDATE jobs SET status=?,result=?,finished=? WHERE id=? AND status='running'",
                                 (status, canonical(result), now(), key)).rowcount
            if changed != 1:
                raise ValueError("Only a claimed job may be completed")

    def rows(self, stage: str | None = None) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM jobs" + (" WHERE stage=?" if stage else "") + " ORDER BY id",
                              (stage,) if stage else ()).fetchall()
        return [{"request_id": r["id"], "stage": r["stage"], "job_status": r["status"],
                 "spec": json.loads(r["spec"]), "result": json.loads(r["result"]) if r["result"] else None,
                 "started_utc": r["started"], "finished_utc": r["finished"]} for r in rows]

    def stats(self) -> dict:
        with self.connect() as db:
            counts = dict(db.execute("SELECT status,count(*) FROM jobs GROUP BY status").fetchall())
            used = db.execute("SELECT count(*) FROM attempts").fetchone()[0]
            cap = int(db.execute("SELECT value FROM metadata WHERE key='call_cap'").fetchone()[0])
        return {"call_cap": cap, "reserved_calls": used, "remaining_calls": cap - used, "jobs": counts}
