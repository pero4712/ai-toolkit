"""
Consistent job-state snapshots for the UI's "Export State".

A running job keeps writing its folder (checkpoints, an optimizer.pt that is
overwritten in place on every save, a WAL-mode loss_log.db, an append-only
loss_events.jsonl), so copying it from outside can pair a checkpoint with
another step's optimizer, tear a file mid-write, or miss rows still in the
WAL. Instead the UI drops ``<save_root>/.export/request.json`` and the trainer
answers it at the next step boundary, when no save or log write is in flight:
it saves a checkpoint at that step, then freezes everything a resume needs
into ``.export/snapshot/``. The UI zips the snapshot while training goes on.

Layout written here (shared with ui/cron/jobState.ts):
    .export/trainer.json          written at train start: this run answers requests
    .export/request.json          {"id": ...}                    (UI -> trainer)
    .export/snapshot/done.json    {"id", "step", "files", ...}   (trainer -> UI)
    .export/snapshot_error.json   {"id", "error"}                (on failure)
"""

import glob
import json
import os
import shutil
import sqlite3
import time
from typing import List, Optional

from toolkit.logging_aitk import prune_steps_after

EXPORT_DIR = ".export"
REQUEST_FILE = "request.json"
SNAPSHOT_DIR = "snapshot"
DONE_FILE = "done.json"
ERROR_FILE = "snapshot_error.json"
TRAINER_MARKER = "trainer.json"

LOSS_DB = "loss_log.db"
LOSS_EVENTS = "loss_events.jsonl"
# copied as-is; optimizer.pt is rewritten in place on every save, so it must
# be a real copy (a hard link would follow later overwrites)
COPY_FILES = ["optimizer.pt", "config.yaml", "loss_analysis.json"]


def _export_dir(save_root: str) -> str:
    return os.path.join(save_root, EXPORT_DIR)


def write_trainer_marker(save_root: str) -> None:
    """Tell the UI this run answers export requests (older runs never would)."""
    export_dir = _export_dir(save_root)
    os.makedirs(export_dir, exist_ok=True)
    _write_json_atomic(
        os.path.join(export_dir, TRAINER_MARKER),
        {"pid": os.getpid(), "started_at": time.time(), "snapshot_version": 1},
    )


def read_request(save_root: str) -> Optional[dict]:
    """The pending export request, or None. Cheap enough to call every step."""
    return _read_request_file(save_root, REQUEST_FILE)

def _read_request_file(save_root: str, name: str) -> Optional[dict]:
    path = os.path.join(_export_dir(save_root), name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            req = json.load(f)
        return req if isinstance(req, dict) else {}
    except (OSError, ValueError):
        # half-written by the UI; pick it up next step
        return None


def _write_json_atomic(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _link_or_copy(src: str, dst: str) -> None:
    # checkpoints are written once under a step-specific name and never
    # modified, so a hard link is a free, stable snapshot; fall back to a copy
    # on filesystems without links
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _snapshot_loss_db(src_path: str, dst_path: str, step: int) -> None:
    # the backup API reads a consistent view including the WAL, which a plain
    # file copy of loss_log.db would miss
    src = sqlite3.connect(src_path, timeout=30.0)
    dst = sqlite3.connect(dst_path, isolation_level=None)
    try:
        src.backup(dst)
        prune_steps_after(dst, step)
    finally:
        dst.close()
        src.close()


def _snapshot_loss_events(src_path: str, dst_path: str, step: int) -> None:
    # append-only and never pruned on resume, so cut it at the snapshot step
    # or a resumed run would log the steps after it twice
    with open(src_path, "r", encoding="utf-8", errors="replace") as src, \
            open(dst_path, "w", encoding="utf-8") as dst:
        for line in src:
            try:
                rec_step = json.loads(line).get("step")
            except ValueError:
                continue
            if isinstance(rec_step, int) and rec_step <= step:
                dst.write(line)


def step_artifacts(save_root: str, step: int) -> List[str]:
    """Names of everything the save at ``step`` wrote (multi-expert files,
    adapters, diffusers checkpoint folders)."""
    tag = str(step).zfill(9)
    names = []
    for path in glob.glob(os.path.join(save_root, f"*_{tag}*")):
        name = os.path.basename(path)
        if not name.startswith("."):
            names.append(name)
    return sorted(names)


def build_snapshot(save_root: str, step: int, request: dict, loss_db_path: Optional[str] = None) -> dict:
    """Freeze the state for ``step`` into .export/snapshot/ and answer the request."""
    export_dir = _export_dir(save_root)
    final_dir = os.path.join(export_dir, SNAPSHOT_DIR)
    tmp_dir = final_dir + ".tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)

    files: List[str] = []
    for name in step_artifacts(save_root, step):
        src = os.path.join(save_root, name)
        dst = os.path.join(tmp_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, copy_function=_link_or_copy)
        else:
            _link_or_copy(src, dst)
        files.append(name)
    if not files:
        raise RuntimeError(f"no checkpoint files found for step {step}")

    for name in COPY_FILES:
        src = os.path.join(save_root, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(tmp_dir, name))
            files.append(name)

    loss_db_path = loss_db_path or os.path.join(save_root, LOSS_DB)
    if os.path.isfile(loss_db_path):
        _snapshot_loss_db(loss_db_path, os.path.join(tmp_dir, LOSS_DB), step)
        files.append(LOSS_DB)

    events_path = os.path.join(save_root, LOSS_EVENTS)
    if os.path.isfile(events_path):
        _snapshot_loss_events(events_path, os.path.join(tmp_dir, LOSS_EVENTS), step)
        files.append(LOSS_EVENTS)

    done = {
        "id": request.get("id"),
        "step": step,
        "files": files,
        "created_at": time.time(),
    }
    # done.json last: its presence means the snapshot is complete
    _write_json_atomic(os.path.join(tmp_dir, DONE_FILE), done)
    shutil.rmtree(final_dir, ignore_errors=True)
    os.replace(tmp_dir, final_dir)
    clear_request(save_root)
    return done


def report_error(save_root: str, request: dict, error: str) -> None:
    export_dir = _export_dir(save_root)
    os.makedirs(export_dir, exist_ok=True)
    _write_json_atomic(
        os.path.join(export_dir, ERROR_FILE),
        {"id": request.get("id"), "error": error, "created_at": time.time()},
    )
    clear_request(save_root)


def clear_request(save_root: str) -> None:
    try:
        os.remove(os.path.join(_export_dir(save_root), REQUEST_FILE))
    except OSError:
        pass
