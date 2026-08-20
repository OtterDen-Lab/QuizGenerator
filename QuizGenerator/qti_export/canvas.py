"""Canvas Content Migrations API support for generated QTI archives."""
from __future__ import annotations

import time
from pathlib import Path

import requests


class QtiUploadError(RuntimeError):
  """Canvas rejected or failed a QTI content migration."""


def upload_qti_package(course, package_path: str | Path, *, timeout: float = 300) -> object:
  """Upload a QTI package through Canvas's required two-step migration flow."""
  package = Path(package_path)
  migration = course.create_content_migration(
    "qti_converter",
    pre_attachment={"name": package.name, "size": package.stat().st_size},
  )
  attachment = getattr(migration, "pre_attachment", None)
  if not attachment or not attachment.get("upload_url"):
    message = attachment.get("message", "Canvas did not provide an upload URL") if attachment else "Canvas did not provide upload details"
    raise QtiUploadError(message)
  with package.open("rb") as handle:
    response = requests.post(
      attachment["upload_url"],
      data=attachment.get("upload_params", {}),
      files={"file": (package.name, handle, "application/zip")},
      timeout=60,
    )
  response.raise_for_status()

  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    progress_url = getattr(migration, "progress_url", None)
    if not progress_url:
      return migration
    progress = migration.get_progress()
    state = getattr(progress, "workflow_state", None)
    if state == "completed":
      return migration
    if state in {"failed", "failure"}:
      raise QtiUploadError(getattr(progress, "message", None) or "Canvas QTI migration failed")
    time.sleep(1)
  raise QtiUploadError(f"Timed out waiting for Canvas QTI migration after {timeout:g} seconds")
