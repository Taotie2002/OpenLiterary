"""Persistent, timestamped task logging for SagaTrans.

The logger is intentionally dependency-free.  It writes an append-only JSONL
stream for machines and a parallel human-readable text stream.  Both files
carry the same ``run_id`` and every event has a millisecond ISO-8601 timestamp
with an explicit ``+08:00`` offset.

The module is also mirrored into the generated single-file snapshot by
``scripts/build_integrated.py``; keep the public names stable.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RUN_ID_ENV = "SAGATRANS_RUN_ID"
LOG_ENABLED_ENV = "SAGATRANS_TASK_LOG_ENABLED"
LOG_DIR_ENV = "SAGATRANS_TASK_LOG_DIR"
DEFAULT_LOG_DIR = Path("output") / "logs" / "runs"
CST = _dt.timezone(_dt.timedelta(hours=8), name="Asia/Shanghai")

_MANDATORY_FIELDS = ("ts", "seq", "run_id", "command", "stage", "event", "status")
_TEXT_CONTENT_KEYS = {
    "text", "source_text", "target_text", "translated_text", "raw_text",
    "prompt", "response", "content", "body", "literary", "final_text",
}
_SECRET_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|auth[_-]?token|access[_-]?token|"
    r"secret|password|passwd|moark[_-]?key|deepseek[_-]?key|minimax[_-]?key)"
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"),
    re.compile(r"(?i)((?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)


class TaskLoggingError(RuntimeError):
    """Raised when a required task log cannot be created or written."""


def now_iso() -> str:
    """Return a millisecond ISO-8601 timestamp in Asia/Shanghai time."""
    return _dt.datetime.now(CST).isoformat(timespec="milliseconds")


def new_run_id() -> str:
    """Create a sortable, process-unique run id."""
    stamp = _dt.datetime.now(CST).strftime("%Y%m%dT%H%M%S")
    suffix = int(time.time_ns() % 100000)
    return f"{stamp}-{os.getpid()}-{suffix:05d}"


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _redact_text(value: str, limit: int = 2000) -> str:
    text = str(value)
    for pattern in _SECRET_PATTERNS:
        def _replace(match: re.Match) -> str:
            if match.lastindex:
                return str(match.group(1)) + "[REDACTED]"
            return "[REDACTED]"
        text = pattern.sub(_replace, text)
    if len(text) > limit:
        text = text[:limit] + "...[truncated]"
    return text


def _sanitize(value: Any, key: str = "") -> Any:
    """Redact secrets and never persist complete source/translation bodies."""
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for item_key, item_value in value.items():
            out[str(item_key)] = _sanitize(item_value, str(item_key))
        return out
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, key) for item in value]
    if isinstance(value, str):
        text = _redact_text(value)
        if key.lower() in _TEXT_CONTENT_KEYS and len(text) > 240:
            return text[:240] + "...[content truncated]"
        return text
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class TaskLogger:
    """Append-only task log with cross-process sequence allocation.

    A single ``run_id`` may be shared by a shell parent and several Python
    children.  The sequence counter is protected by ``fcntl.flock`` so events
    from all processes form one monotonic stream.
    """

    def __init__(
        self,
        root: Path,
        command: str,
        run_id: Optional[str] = None,
        read_only: bool = False,
        enabled: bool = True,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.command = str(command or "unknown")
        self.run_id = str(run_id or os.environ.get(RUN_ID_ENV) or new_run_id())
        os.environ[RUN_ID_ENV] = self.run_id
        self.read_only = bool(read_only)
        self.enabled = bool(enabled)
        self._explicit_log_dir = Path(log_dir) if log_dir else None
        self.log_dir = (self._explicit_log_dir if self._explicit_log_dir
                        else self.root / DEFAULT_LOG_DIR)
        self.jsonl_path = self.log_dir / f"{self.run_id}.jsonl"
        self.text_path = self.log_dir / f"{self.run_id}.log"
        self.seal_path = self.log_dir / f"{self.run_id}.sha256"
        self._seq_path = self.log_dir / f"{self.run_id}.seq"
        self._lock_path = self.log_dir / f"{self.run_id}.lock"
        self._jsonl = None
        self._text = None
        self._seq: Optional[int] = None
        self._started = False
        self._closed = False
        self._last_status = "running"
        self._started_monotonic: Optional[float] = None
        self._last_ts = ""
        if self.enabled:
            self._open()

    def _open(self) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl = open(self.jsonl_path, "a", encoding="utf-8", buffering=1)
            self._text = open(self.text_path, "a", encoding="utf-8", buffering=1)
        except OSError as exc:
            raise TaskLoggingError(f"无法创建任务日志: {exc}") from exc

    def _next_seq(self) -> int:
        # Re-read the sidecar on every event: several Python children may share
        # one run_id, and an in-process cached counter would otherwise reuse
        # sequence numbers after another process has advanced the stream.
        disk_seq = 0
        try:
            disk_seq = int(self._seq_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            disk_seq = 0
        self._seq = max(self._seq or 0, disk_seq)
        self._seq += 1
        try:
            _atomic_write_text(self._seq_path, str(self._seq))
        except OSError:
            pass
        return self._seq

    def _format_text(self, payload: Dict[str, Any]) -> str:
        parts = [
            f"[{payload.get('ts', '')}]",
            str(payload.get("event") or ""),
            f"stage={payload.get('stage') or '-'}",
            f"status={payload.get('status') or '-'}",
        ]
        for key in ("chapter", "chunk_id", "attempt", "artifact", "error_class"):
            value = payload.get(key)
            if value not in (None, ""):
                parts.append(f"{key}={value}")
        message = str(payload.get("message") or "")
        if message:
            parts.append(message)
        return " ".join(parts).replace("\n", " ").replace("\r", " ")

    def event(self, event: str, *, stage: str = "", status: str = "ok",
              message: str = "", **fields: Any) -> Optional[Dict[str, Any]]:
        """Append one event.  Returns the emitted payload, or None if disabled."""
        if not self.enabled:
            return None
        if self._closed:
            raise TaskLoggingError("任务日志已关闭，不能再写入事件")
        payload: Dict[str, Any] = dict(fields)
        payload.update({
            "ts": now_iso(),
            "run_id": self.run_id,
            "command": self.command,
            "stage": str(stage or ""),
            "event": str(event or ""),
            "status": str(status or ""),
            "message": str(message or ""),
        })
        payload = _sanitize(payload)
        # Reserved fields are always present even if a caller supplied them.
        for key in _MANDATORY_FIELDS:
            payload.setdefault(key, "")
        try:
            import fcntl
            with open(self._lock_path, "a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    payload["seq"] = self._next_seq()
                    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    assert self._jsonl is not None and self._text is not None
                    self._jsonl.write(line + "\n")
                    self._jsonl.flush()
                    self._text.write(self._format_text(payload) + "\n")
                    self._text.flush()
                    self._last_ts = str(payload["ts"])
                    self._last_status = str(payload.get("status") or "")
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except Exception as exc:
            if self.read_only:
                print(f"⚠️ [TaskLog] 写日志失败（只读命令继续）: {exc}", file=sys.stderr)
                return payload
            raise TaskLoggingError(f"写任务日志失败: {exc}") from exc
        return payload

    def start(self, **fields: Any) -> Optional[Dict[str, Any]]:
        if self._started:
            return None
        self._started = True
        self._started_monotonic = time.monotonic()
        fields.setdefault("argv", [str(item) for item in sys.argv])
        fields.setdefault("pid", os.getpid())
        fields.setdefault("cwd", str(Path.cwd()))
        fields.setdefault("python", sys.executable)
        fields.setdefault("jsonl_path", str(self.jsonl_path))
        fields.setdefault("log_path", str(self.text_path))
        return self.event("run_start", stage="RUN", status="running", **fields)

    def __enter__(self) -> "TaskLogger":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.finish(0)
        elif issubclass(exc_type, KeyboardInterrupt):
            self.finish(130, status="interrupted")
        elif issubclass(exc_type, SystemExit):
            code = getattr(exc, "code", 1)
            if code in (None, 0):
                self.finish(0)
            elif code == 130:
                self.finish(130, status="interrupted")
            else:
                self.finish(int(code) if isinstance(code, int) else 1)
        else:
            self.finish(1, status="failed", error_class=type(exc).__name__)
        return False

    def finish(self, exit_code: int = 0, status: Optional[str] = None,
               **fields: Any) -> Optional[Dict[str, Any]]:
        if self._closed:
            return None
        if status is None:
            if exit_code == 0:
                status = "ok"
            elif exit_code == 130:
                status = "interrupted"
            else:
                status = "failed"
        duration_ms = 0
        if self._started_monotonic is not None:
            duration_ms = int((time.monotonic() - self._started_monotonic) * 1000)
        payload = self.event(
            "run_end", stage="RUN", status=str(status),
            exit_code=int(exit_code), duration_ms=duration_ms, **fields)
        try:
            if self._jsonl is not None:
                self._jsonl.flush()
                os.fsync(self._jsonl.fileno())
            if self._text is not None:
                self._text.flush()
                os.fsync(self._text.fileno())
        except (OSError, ValueError):
            pass
        self.close(seal=True)
        return payload

    def close(self, seal: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        for handle in (self._jsonl, self._text):
            try:
                if handle is not None:
                    handle.close()
            except OSError:
                pass
        self._jsonl = None
        self._text = None
        if seal and self.enabled:
            self.seal()

    def seal(self) -> Optional[Dict[str, Any]]:
        """Write the post-run hash envelope for both log streams."""
        if not self.jsonl_path.exists() and not self.text_path.exists():
            return None
        payload = {
            "run_id": self.run_id,
            "command": self.command,
            "sealed_at": now_iso(),
            "status": self._last_status,
            "jsonl": str(self.jsonl_path.relative_to(self.root))
            if self.jsonl_path.is_relative_to(self.root) else str(self.jsonl_path),
            "log": str(self.text_path.relative_to(self.root))
            if self.text_path.is_relative_to(self.root) else str(self.text_path),
            "jsonl_sha256": sha256_file(self.jsonl_path) if self.jsonl_path.exists() else "",
            "log_sha256": sha256_file(self.text_path) if self.text_path.exists() else "",
            "seq_last": self._seq or 0,
            "last_ts": self._last_ts,
        }
        try:
            _atomic_write_text(
                self.seal_path,
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
        except OSError as exc:
            if not self.read_only:
                raise TaskLoggingError(f"无法封存任务日志哈希: {exc}") from exc
            print(f"⚠️ [TaskLog] 封存失败（只读命令继续）: {exc}", file=sys.stderr)
            return None
        return payload


def make_logger(root: Path, command: str, read_only: bool = False,
                enabled: Optional[bool] = None,
                run_id: Optional[str] = None,
                log_dir: Optional[Path] = None) -> TaskLogger:
    """Create a logger according to the command's failure policy.

    Write commands fail closed when the primary log cannot be created.  Read-only
    diagnostics keep running but report the degradation on stderr.
    """
    if enabled is None:
        env_value = os.environ.get(LOG_ENABLED_ENV, "").strip().lower()
        enabled = env_value not in {"0", "false", "off", "no"}
    try:
        return TaskLogger(root=root, command=command, run_id=run_id,
                          read_only=read_only, enabled=bool(enabled),
                          log_dir=log_dir)
    except TaskLoggingError:
        if read_only:
            print("⚠️ [TaskLog] 任务日志不可写；只读命令继续运行", file=sys.stderr)
            return TaskLogger(root=root, command=command, run_id=run_id,
                              read_only=True, enabled=False, log_dir=log_dir)
        raise


_ACTIVE_LOGGER: Optional[TaskLogger] = None


def set_active_logger(logger: Optional[TaskLogger]) -> None:
    global _ACTIVE_LOGGER
    _ACTIVE_LOGGER = logger


def get_active_logger() -> Optional[TaskLogger]:
    return _ACTIVE_LOGGER


def log_event(event: str, *, stage: str = "", status: str = "ok",
              message: str = "", **fields: Any) -> Optional[Dict[str, Any]]:
    logger = _ACTIVE_LOGGER
    if logger is None or not logger.enabled:
        return None
    return logger.event(event, stage=stage, status=status, message=message, **fields)


def log_artifact(path: Path, *, stage: str = "", chunk_id: str = "",
                 kind: str = "", status: str = "ok") -> Optional[Dict[str, Any]]:
    """Record an artifact write when a task logger is active."""
    logger = _ACTIVE_LOGGER
    if logger is None or not logger.enabled:
        return None
    artifact_path = Path(path)
    digest = ""
    try:
        if artifact_path.exists() and artifact_path.is_file():
            digest = "sha256:" + sha256_file(artifact_path)
    except OSError:
        digest = ""
    return logger.event(
        "artifact_written", stage=stage, status=status,
        chunk_id=chunk_id or "", artifact=str(artifact_path),
        artifact_sha256=digest, artifact_kind=kind or "",
    )


def validate_jsonl(path: Path, expected_run_id: Optional[str] = None,
                   require_run_start: bool = True,
                   require_run_end: bool = True) -> Dict[str, Any]:
    """Validate ordering, timestamps and mandatory fields in a run JSONL file."""
    errors: List[str] = []
    seqs: List[int] = []
    run_ids = set()
    events: List[str] = []
    path = Path(path)
    if not path.exists():
        return {"ok": False, "errors": ["missing_jsonl"], "events": []}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {"ok": False, "errors": [f"read_error:{exc}"], "events": []}
    for line_no, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line{line_no}:json_error:{exc.msg}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"line{line_no}:not_object")
            continue
        missing = [key for key in _MANDATORY_FIELDS if key not in payload]
        if missing:
            errors.append(f"line{line_no}:missing:{','.join(missing)}")
        ts = str(payload.get("ts") or "")
        try:
            parsed_ts = _dt.datetime.fromisoformat(ts)
            if parsed_ts.tzinfo is None:
                errors.append(f"line{line_no}:ts_without_tz")
        except ValueError:
            errors.append(f"line{line_no}:bad_ts")
        try:
            seqs.append(int(payload.get("seq")))
        except (TypeError, ValueError):
            errors.append(f"line{line_no}:bad_seq")
        if payload.get("run_id"):
            run_ids.add(str(payload.get("run_id")))
        if payload.get("event"):
            events.append(str(payload.get("event")))
    for index in range(1, len(seqs)):
        if seqs[index] != seqs[index - 1] + 1:
            errors.append(f"seq_gap:{seqs[index - 1]}->{seqs[index]}")
    if expected_run_id and run_ids and run_ids != {expected_run_id}:
        errors.append("run_id_mismatch")
    if require_run_start and not any(event == "run_start" for event in events):
        errors.append("run_start_missing")
    if require_run_end and not any(event == "run_end" for event in events):
        errors.append("run_end_missing")
    return {
        "ok": not errors,
        "errors": errors,
        "events": events,
        "seq_first": seqs[0] if seqs else 0,
        "seq_last": seqs[-1] if seqs else 0,
        "run_ids": sorted(run_ids),
    }


def prune_runs(log_dir: Path, keep_last: int = 30, keep_days: int = 90,
               current_run_id: str = "") -> List[str]:
    """Delete old run artifacts while preserving the active run.

    Returns the list of removed run ids.  This function is explicit; callers
    decide when retention is safe.
    """
    directory = Path(log_dir)
    if not directory.is_dir():
        return []
    now = time.time()
    files = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    removed: List[str] = []
    for index, jsonl_path in enumerate(files):
        run_id = jsonl_path.stem
        if run_id == current_run_id:
            continue
        try:
            age_days = (now - jsonl_path.stat().st_mtime) / 86400.0
        except OSError:
            continue
        if index < keep_last and age_days <= keep_days:
            continue
        for suffix in (".jsonl", ".log", ".sha256", ".seq", ".lock"):
            target = directory / f"{run_id}{suffix}"
            try:
                if target.exists():
                    target.unlink()
            except OSError:
                pass
        removed.append(run_id)
    return removed


__all__ = [
    "CST", "RUN_ID_ENV", "LOG_ENABLED_ENV", "LOG_DIR_ENV",
    "DEFAULT_LOG_DIR", "TaskLoggingError", "TaskLogger",
    "now_iso", "new_run_id", "sha256_file", "make_logger",
    "set_active_logger", "get_active_logger", "log_event", "log_artifact",
    "validate_jsonl", "prune_runs",
]
