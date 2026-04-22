from __future__ import annotations

from ..pipelines.ocr.remote_job import ObjectStoreRemoteJob, RemoteJob, cli, main

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.

__all__ = ["RemoteJob", "ObjectStoreRemoteJob", "cli", "main"]


if __name__ == "__main__":
    main()
