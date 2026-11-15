"""SQLite 访问层：数据集元信息、对齐运行结果。命名卷 tsgrid-data:/app/data。"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

from . import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    sha256     TEXT UNIQUE,
    source     TEXT,
    rows       INTEGER,
    files_json TEXT,
    inventory_json TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    dataset_id  TEXT,
    device_id   TEXT,
    metric      TEXT,
    threshold_s REAL,
    created_at  TEXT,
    payload_json TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
-- 发布版本：同一份 (数据集, 设备, 指标, 阈值, 内容哈希) 幂等；只有全部作业完成后才原子切换 active。
CREATE TABLE IF NOT EXISTS publications (
    pub_id        TEXT PRIMARY KEY,
    dataset_id    TEXT NOT NULL,
    device_id     TEXT NOT NULL,
    metric        TEXT NOT NULL,
    threshold_s   REAL NOT NULL,
    content_hash  TEXT NOT NULL,
    status        TEXT NOT NULL,            -- running | published | failed
    is_active     INTEGER NOT NULL DEFAULT 0,
    segment_count INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    payload_json  TEXT,
    created_at    TEXT,
    updated_at    TEXT,
    UNIQUE(dataset_id, device_id, metric, threshold_s, content_hash)
);
-- 分段作业：每个采样档位分段一行，完成后持久化结果切片；续跑时已完成作业不重算。
CREATE TABLE IF NOT EXISTS segment_jobs (
    pub_id        TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    interval_s    INTEGER NOT NULL,
    status        TEXT NOT NULL,            -- pending | done | failed
    attempts      INTEGER NOT NULL DEFAULT 0,
    result_json   TEXT,
    error         TEXT,
    updated_at    TEXT,
    PRIMARY KEY (pub_id, segment_index)
);
CREATE INDEX IF NOT EXISTS idx_publications_active
    ON publications(dataset_id, device_id, metric, is_active);
"""

# 发布状态机允许的状态
PUB_RUNNING = "running"
PUB_PUBLISHED = "published"
PUB_FAILED = "failed"
JOB_PENDING = "pending"
JOB_DONE = "done"
JOB_FAILED = "failed"


def _connect() -> sqlite3.Connection:
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def upsert_dataset(dataset) -> str:
    """按内容 sha256 幂等：同一份数据重复导入返回既有 dataset_id。"""
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT dataset_id FROM datasets WHERE sha256 = ?", (dataset.sha256,)).fetchone()
        if row:
            return row["dataset_id"]
        conn.execute(
            "INSERT INTO datasets (dataset_id, sha256, source, rows, files_json, inventory_json, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                dataset.dataset_id,
                dataset.sha256,
                dataset.source,
                dataset.rows,
                json.dumps(dataset.files, ensure_ascii=False),
                json.dumps(dataset.inventory(), ensure_ascii=False),
                _now(),
            ),
        )
        return dataset.dataset_id


def list_datasets() -> list[dict]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT dataset_id, sha256, source, rows, files_json, inventory_json, created_at"
            " FROM datasets ORDER BY created_at"
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["files"] = json.loads(item.pop("files_json") or "[]")
        item["inventory"] = json.loads(item.pop("inventory_json") or "[]")
        out.append(item)
    return out


def save_run(dataset_id: str, payload: dict) -> str:
    init_db()
    run_id = "run-" + uuid.uuid4().hex[:10]
    with _connect() as conn:
        conn.execute(
            "INSERT INTO runs (run_id, dataset_id, device_id, metric, threshold_s, created_at, payload_json)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                run_id,
                dataset_id,
                payload.get("device_id"),
                payload.get("metric"),
                payload.get("threshold_s"),
                _now(),
                json.dumps(payload, ensure_ascii=False),
            ),
        )
    return run_id


def get_run(run_id: str) -> dict | None:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        return None
    payload = json.loads(row["payload_json"])
    payload["run_id"] = row["run_id"]
    payload["created_at"] = row["created_at"]
    return payload


def list_runs(limit: int = 20) -> list[dict]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT run_id, device_id, metric, threshold_s, created_at FROM runs"
            " ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_meta(key: str) -> str | None:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(key: str, value: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("INSERT INTO meta (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


# ---------------------------------------------------------------------------
# 发布版本与分段作业（断点续跑 / 原子切换 / 幂等重跑）
# ---------------------------------------------------------------------------

def create_publication(
    pub_id: str,
    dataset_id: str,
    device_id: str,
    metric: str,
    threshold_s: float,
    content_hash: str,
    segment_plans: list[dict],
) -> dict:
    """登记一个新的发布版本及其全部分段作业（初始 pending）。

    同 content_hash 的版本若已存在则直接返回（幂等：重跑同一批不新建版本、不重算已完成分段）。
    """
    init_db()
    now = _now()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM publications WHERE dataset_id=? AND device_id=? AND metric=? AND threshold_s=? AND content_hash=?",
            (dataset_id, device_id, metric, threshold_s, content_hash),
        ).fetchone()
        if row:
            return _publication_dict(row)
        conn.execute(
            "INSERT INTO publications (pub_id, dataset_id, device_id, metric, threshold_s, content_hash,"
            " status, is_active, segment_count, created_at, updated_at) VALUES (?,?,?,?,?,?,?,0,?,?,?)",
            (pub_id, dataset_id, device_id, metric, threshold_s, content_hash, PUB_RUNNING, len(segment_plans), now, now),
        )
        for plan in segment_plans:
            conn.execute(
                "INSERT INTO segment_jobs (pub_id, segment_index, interval_s, status, attempts, updated_at)"
                " VALUES (?,?,?,?,0,?)",
                (pub_id, plan["index"], plan["interval_s"], JOB_PENDING, now),
            )
        row = conn.execute("SELECT * FROM publications WHERE pub_id=?", (pub_id,)).fetchone()
        return _publication_dict(row)


def get_publication(pub_id: str) -> dict | None:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM publications WHERE pub_id=?", (pub_id,)).fetchone()
    return _publication_dict(row) if row else None


def get_publication_by_hash(
    dataset_id: str, device_id: str, metric: str, threshold_s: float, content_hash: str
) -> dict | None:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM publications WHERE dataset_id=? AND device_id=? AND metric=? AND threshold_s=? AND content_hash=?",
            (dataset_id, device_id, metric, threshold_s, content_hash),
        ).fetchone()
    return _publication_dict(row) if row else None


def list_segment_jobs(pub_id: str) -> list[dict]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM segment_jobs WHERE pub_id=? ORDER BY segment_index", (pub_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_segment_job(pub_id: str, segment_index: int) -> dict | None:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM segment_jobs WHERE pub_id=? AND segment_index=?", (pub_id, segment_index)
        ).fetchone()
    return dict(row) if row else None


def mark_segment_done(pub_id: str, segment_index: int, result: dict) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE segment_jobs SET status=?, result_json=?, attempts=attempts+1, error=NULL, updated_at=?"
            " WHERE pub_id=? AND segment_index=?",
            (JOB_DONE, json.dumps(result, ensure_ascii=False), _now(), pub_id, segment_index),
        )
        conn.execute("UPDATE publications SET updated_at=? WHERE pub_id=?", (_now(), pub_id))


def mark_segment_failed(pub_id: str, segment_index: int, error: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE segment_jobs SET status=?, attempts=attempts+1, error=?, updated_at=?"
            " WHERE pub_id=? AND segment_index=?",
            (JOB_FAILED, error, _now(), pub_id, segment_index),
        )


def mark_publication_failed(pub_id: str, error: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE publications SET status=?, error=?, updated_at=? WHERE pub_id=?",
            (PUB_FAILED, error, _now(), pub_id),
        )


def publish_atomically(pub_id: str, payload: dict) -> dict:
    """原子切换：在单个事务里写入完整 payload、置 published 并把 active 指针从旧版切到新版。

    任何一步失败整个事务回滚——旧 active 版本保持不变（失败保留旧版）。
    """
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM publications WHERE pub_id=?", (pub_id,)).fetchone()
        if row is None:
            raise KeyError(f"发布版本 {pub_id} 不存在")
        pending = conn.execute(
            "SELECT COUNT(*) AS c FROM segment_jobs WHERE pub_id=? AND status != ?",
            (pub_id, JOB_DONE),
        ).fetchone()
        if pending["c"] > 0:
            raise RuntimeError(f"仍有 {pending['c']} 个分段作业未完成，不能原子发布")
        conn.execute(
            "UPDATE publications SET status=?, payload_json=?, error=NULL, is_active=0, updated_at=? WHERE pub_id=?",
            (PUB_PUBLISHED, json.dumps(payload, ensure_ascii=False), _now(), pub_id),
        )
        # 摘掉同 (数据集,设备,指标) 的旧 active，再指向新版——同事务提交，外部只会看到新版。
        conn.execute(
            "UPDATE publications SET is_active=0 WHERE dataset_id=? AND device_id=? AND metric=? AND pub_id != ?",
            (row["dataset_id"], row["device_id"], row["metric"], pub_id),
        )
        conn.execute("UPDATE publications SET is_active=1 WHERE pub_id=?", (pub_id,))
        out = conn.execute("SELECT * FROM publications WHERE pub_id=?", (pub_id,)).fetchone()
    return _publication_dict(out)


def get_active_publication(dataset_id: str, device_id: str, metric: str) -> dict | None:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM publications WHERE dataset_id=? AND device_id=? AND metric=? AND is_active=1",
            (dataset_id, device_id, metric),
        ).fetchone()
    return _publication_dict(row) if row else None


def list_publications(dataset_id: str | None = None) -> list[dict]:
    init_db()
    with _connect() as conn:
        if dataset_id:
            rows = conn.execute(
                "SELECT * FROM publications WHERE dataset_id=? ORDER BY device_id, metric, created_at",
                (dataset_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM publications ORDER BY dataset_id, device_id, metric, created_at"
            ).fetchall()
    return [_publication_dict(r) for r in rows]


def _publication_dict(row: sqlite3.Row) -> dict:
    item = dict(row)
    if item.get("payload_json"):
        item["payload"] = json.loads(item["payload_json"])
    else:
        item["payload"] = None
    item.pop("payload_json", None)
    item["is_active"] = bool(item["is_active"])
    return item
