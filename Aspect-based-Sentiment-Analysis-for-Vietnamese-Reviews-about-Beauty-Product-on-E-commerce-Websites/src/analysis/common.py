"""Shared byte-level IO for analysis; no model or evaluator dependencies."""

import csv
import hashlib
import io
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    """Return the SHA-256 of a file without loading the entire file into memory."""
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def csv_bytes(rows, fields=None, *, extrasaction="raise"):
    """Preserve the existing UTF-8, CRLF and blank-null CSV representation."""
    if fields is None:
        fields = rows[0].keys()
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction=extrasaction)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_once(path, payload):
    """Atomically create bytes, preserve identical files, reject different bytes."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Existing TEST-02B artifact differs; preserving it: {path}")
        print("PRESERVED identical existing artifact", path, flush=True)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
