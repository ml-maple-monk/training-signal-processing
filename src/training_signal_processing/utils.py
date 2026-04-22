from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def utc_timestamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def utc_isoformat() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def read_jsonl_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise ValueError(f"JSONL input file not found: {path}")
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"JSONL input line {line_number} must be a JSON object in {path}"
            )
        rows.append(value)
    return rows


def join_s3_key(prefix: str, suffix: str) -> str:
    return f"{prefix.rstrip('/')}/{suffix.lstrip('/')}"


def make_s3_url(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


def compute_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_json_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2).encode("utf-8")
