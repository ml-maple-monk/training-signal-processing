from __future__ import annotations

from pathlib import Path

import pyarrow.fs as pafs

from ...core.storage import ObjectStore


class LocalParquetObjectStore(ObjectStore):
    """ObjectStore facade over a local mirror shaped like <root>/<bucket>/<key>."""

    def __init__(self, root: Path, *, bucket: str) -> None:
        self.root = root.expanduser().resolve()
        self.bucket = bucket
        self.storage_root = self.root / bucket

    def exists(self, key: str) -> bool:
        return (self.storage_root / key.lstrip("/")).exists()

    def list_keys(self, prefix: str) -> list[str]:
        prefix = prefix.strip().lstrip("/")
        prefix_root = self.storage_root / prefix
        if not prefix_root.exists():
            return []
        return sorted(
            path.relative_to(self.storage_root).as_posix()
            for path in prefix_root.rglob("*")
            if path.is_file()
        )

    def read_bytes(self, key: str) -> bytes:
        return (self.storage_root / key.lstrip("/")).read_bytes()

    def write_bytes(self, key: str, body: bytes) -> None:
        path = self.storage_root / key.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)

    def upload_file(self, path: Path, key: str) -> None:
        self.write_bytes(key, path.read_bytes())

    def make_url(self, key: str) -> str:
        return (self.storage_root / key.lstrip("/")).as_uri()

    def build_pyarrow_filesystem(self) -> pafs.FileSystem:
        return pafs.SubTreeFileSystem(
            self.root.as_posix(),
            pafs.LocalFileSystem(),
        )
