"""SQLite store for community outbreak reports."""
import os, sqlite3
from contextlib import closing

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "reports.db")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with closing(_conn()) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                crop TEXT NOT NULL,
                severity INTEGER NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                village TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )""")
        c.commit()


def risk_of(severity: int) -> str:
    return "high" if severity >= 68 else "watch" if severity >= 45 else "ok"


def add_report(label, crop, severity, lat, lon, village=None):
    with closing(_conn()) as c:
        cur = c.execute(
            "INSERT INTO reports (label,crop,severity,lat,lon,village) VALUES (?,?,?,?,?,?)",
            (label, crop, severity, lat, lon, village))
        c.commit()
        rid = cur.lastrowid
        row = c.execute("SELECT * FROM reports WHERE id=?", (rid,)).fetchone()
        return _row(row)


def list_reports(limit=500):
    with closing(_conn()) as c:
        rows = c.execute("SELECT * FROM reports ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [_row(r) for r in rows]


def stats():
    reps = list_reports()
    villages = len({r["village"] for r in reps if r["village"]})
    alerts = len([r for r in reps if r["risk"] == "high"])
    return {"reports_total": len(reps), "villages": villages, "alerts": alerts, "reports": reps}


def _row(r):
    d = dict(r)
    d["risk"] = risk_of(d["severity"])
    return d
