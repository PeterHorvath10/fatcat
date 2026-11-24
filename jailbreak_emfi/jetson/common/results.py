"""Result files written on the Jetsons: JSONL records, locked TSV logs, atomic JSON documents.

Every write is flushed and fsync'ed before returning: an EM pulse can reset the board at any moment, and a
trial that was acknowledged must not lose its record.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def new_run_id() -> str:
    """Identifier of one worker process run (UTC start time + random suffix)."""
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:12]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Records of a JSONL file; torn lines (board reset mid-write) are skipped."""
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def append_tsv(path: Path, header: str, records: Iterable[str]) -> None:
    """Append records under an exclusive lock; write the header if the file is new or empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(f"{header}\n")
        for record in records:
            handle.write(f"{record}\n")
        handle.flush()
        os.fsync(handle.fileno())
    if created:
        fsync_directory(path.parent)


def write_json_atomic(path: Path, document: Any, indent: int | None = 2) -> None:
    """Replace path with document (temporary file + rename), durable when it returns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(document, indent=indent) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)
