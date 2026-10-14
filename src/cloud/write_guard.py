"""Refuse any GCS write outside one allowed prefix. Installed by main.py when
CADBUSI_WRITE_PREFIX is set, which every Vertex job launched from src/cloud/ does.

Patches the google-cloud-storage client itself (uploads, deletes, copies, renames,
rewrites, composes), so it covers storage_adapter and every module that talks to
the bucket directly. Reads are untouched: a test run may read the main database
and the raw downloads, but can only write under its own run folder.
"""
from __future__ import annotations

import functools
import os

_installed = False


def _check(bucket_name: str, blob_name: str, prefix: str):
    name = (blob_name or "").replace("//", "/").lstrip("/")
    if not name.startswith(prefix):
        raise PermissionError(f"write guard: refusing to write gs://{bucket_name}/{name} "
                              f"(only {prefix!r} is writable in this run)")


def install(prefix: str | None = None):
    """Allow writes only to blobs whose name starts with `prefix` (a path inside the bucket)."""
    global _installed
    prefix = (prefix or os.environ.get("CADBUSI_WRITE_PREFIX", "")).strip("/") + "/"
    if _installed or prefix == "/":
        return
    from google.cloud.storage import Blob, Bucket

    def blob_method(fn):
        @functools.wraps(fn)
        def wrapped(self, *a, **kw):
            _check(self.bucket.name, self.name, prefix)
            return fn(self, *a, **kw)
        return wrapped

    for m in ("upload_from_string", "upload_from_filename", "upload_from_file", "delete", "compose",
              "create_resumable_upload_session", "open"):
        if hasattr(Blob, m):
            orig = getattr(Blob, m)
            if m == "open":  # reading is fine; only writable handles are guarded
                @functools.wraps(orig)
                def guarded_open(self, mode="r", *a, _orig=orig, **kw):
                    if "w" in mode:
                        _check(self.bucket.name, self.name, prefix)
                    return _orig(self, mode, *a, **kw)
                setattr(Blob, m, guarded_open)
            else:
                setattr(Blob, m, blob_method(orig))

    orig_rewrite = Blob.rewrite

    @functools.wraps(orig_rewrite)
    def rewrite(self, source, *a, **kw):  # self is the destination
        _check(self.bucket.name, self.name, prefix)
        return orig_rewrite(self, source, *a, **kw)
    Blob.rewrite = rewrite

    orig_copy = Bucket.copy_blob

    @functools.wraps(orig_copy)
    def copy_blob(self, blob, destination_bucket, new_name=None, *a, **kw):
        _check(destination_bucket.name, new_name or blob.name, prefix)
        return orig_copy(self, blob, destination_bucket, new_name, *a, **kw)
    Bucket.copy_blob = copy_blob

    orig_rename = Bucket.rename_blob

    @functools.wraps(orig_rename)
    def rename_blob(self, blob, new_name, *a, **kw):
        _check(self.name, blob.name, prefix)
        _check(self.name, new_name, prefix)
        return orig_rename(self, blob, new_name, *a, **kw)
    Bucket.rename_blob = rename_blob

    orig_delete_blob = Bucket.delete_blob

    @functools.wraps(orig_delete_blob)
    def delete_blob(self, blob_name, *a, **kw):
        _check(self.name, blob_name, prefix)
        return orig_delete_blob(self, blob_name, *a, **kw)
    Bucket.delete_blob = delete_blob

    orig_delete_blobs = Bucket.delete_blobs

    @functools.wraps(orig_delete_blobs)
    def delete_blobs(self, blobs, *a, **kw):
        for b in blobs:
            _check(self.name, getattr(b, "name", b), prefix)
        return orig_delete_blobs(self, blobs, *a, **kw)
    Bucket.delete_blobs = delete_blobs

    _installed = True
    print(f"[write guard] GCS writes restricted to {prefix}")
