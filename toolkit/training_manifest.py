"""
Training manifest join for loss attribution.

Video LoRA Studio writes ``manifest.json`` (``{"session_id", "fps", "rows"}``)
beside the dataset folders of every training export. Each row carries the
per-file semantic attribution captured BEFORE folder consolidation, so loss
cohorts can be keyed by what a clip *is* (semantic group, phase, level, take)
rather than by which pooled target folder it landed in.

Files without a manifest row keep their folder-derived attribution. Lookups
never raise: a missing or malformed manifest simply yields no joins.
"""

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

MANIFEST_FILENAME = "manifest.json"


def _s(row: dict, key: str) -> str:
    v = row.get(key)
    return "" if v is None else str(v)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


@dataclass
class ManifestAttribution:
    file: str                 # manifest join key, "<folder>/<name>" relative to export root
    segment_id: str
    semantic_group: str
    semantic_group_id: str
    phase_type: str
    phase_id: str
    level: str
    source_take: str
    profile_id: str
    profile_version: str
    window: Optional[int]     # 1-based index for physically pre-cut multiclip files
    weight: Optional[float]   # design provenance only, never a training input
    duplicate_of: str         # base row's `file` for _dupN share copies
    manifest_hash: str
    session_id: str
    # identity resolved through duplicate_of: share copies collapse onto their base
    base_file: str
    base_segment_id: str

    @property
    def is_duplicate(self) -> bool:
        return bool(self.duplicate_of)

    @property
    def clip_name(self) -> str:
        """Human-readable clip id (base file stem), matching today's source_id style."""
        return _stem(self.base_file)


class ManifestIndex:
    """All manifests found beside the configured dataset folders, joined by path."""

    def __init__(self) -> None:
        # manifest path -> {"dir", "rows_by_file", "session_id", "hash", "row_count"}
        self._manifests: Dict[str, dict] = {}
        self._path_cache: Dict[str, Optional[ManifestAttribution]] = {}
        # design-side expectations for the exposure check
        self.design_by_group: Dict[str, dict] = {}
        self.design_by_take: Dict[str, dict] = {}
        self.total_files: int = 0
        # design_shares header: semantic group -> intended effective share (0-1),
        # the exact numbers the studio's weight hierarchy computed at export.
        # Older manifests lack the key; empty dict = fall back to file counts.
        self.design_shares: Dict[str, float] = {}
        self.design_shares_present: bool = False
        # folder basename -> semantic groups / file counts, for the check-2
        # cardinality tripwire (a pooled folder whose rows all claim one group)
        self.groups_by_folder: Dict[str, List[str]] = {}
        self.files_by_folder: Dict[str, int] = {}

    # ------------------------------------------------------------------ load

    @classmethod
    def load_for_folders(cls, folder_paths: Iterable[Optional[str]]) -> "ManifestIndex":
        index = cls()
        seen = set()
        for folder in folder_paths:
            if not folder:
                continue
            folder = os.path.abspath(folder)
            # the manifest sits at the export root (parent of the dataset folders);
            # also accept one inside the folder itself
            for candidate_dir in (os.path.dirname(folder), folder):
                manifest_path = os.path.join(candidate_dir, MANIFEST_FILENAME)
                if manifest_path in seen:
                    continue
                seen.add(manifest_path)
                if os.path.isfile(manifest_path):
                    index._load_manifest(manifest_path)
        index._build_design()
        return index

    def _load_manifest(self, manifest_path: str) -> None:
        try:
            with open(manifest_path, "rb") as f:
                raw = f.read()
            data = json.loads(raw.decode("utf-8"))
        except Exception as e:
            print(f"[manifest] could not read {manifest_path}: {e}")
            return
        rows = data.get("rows") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            print(f"[manifest] {manifest_path} has no rows list; ignoring")
            return
        rows_by_file: Dict[str, dict] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            file_key = _s(row, "file").replace("\\", "/")
            if file_key:
                rows_by_file[file_key] = row
        self._manifests[manifest_path] = {
            "dir": os.path.dirname(manifest_path),
            "rows_by_file": rows_by_file,
            "session_id": _s(data, "session_id"),
            "hash": hashlib.sha256(raw).hexdigest()[:12],
            "row_count": len(rows_by_file),
            "fps": data.get("fps"),
            "design_shares": data.get("design_shares")
            if isinstance(data.get("design_shares"), dict) else None,
        }

    def _build_design(self) -> None:
        by_group: Dict[str, int] = defaultdict(int)
        take_bases: Dict[str, set] = defaultdict(set)
        take_copies: Dict[str, int] = defaultdict(int)
        folder_groups: Dict[str, set] = defaultdict(set)
        folder_files: Dict[str, int] = defaultdict(int)
        total = 0
        for m in self._manifests.values():
            shares = m.get("design_shares")
            if shares:
                self.design_shares_present = True
                for g, v in shares.items():
                    try:
                        self.design_shares[str(g)] = float(v)
                    except (TypeError, ValueError):
                        pass
            for file_key, row in m["rows_by_file"].items():
                total += 1
                group = _s(row, "semantic_group")
                if group:
                    # duplicates count as themselves: duplication IS the share system
                    by_group[group] += 1
                take = _s(row, "source_take")
                if take:
                    take_copies[take] += 1
                    take_bases[take].add(_s(row, "duplicate_of") or file_key)
                if "/" in file_key:
                    folder = file_key.split("/", 1)[0]
                    folder_files[folder] += 1
                    if group:
                        folder_groups[folder].add(group)
        self.total_files = total
        self.design_by_group = {g: {"files": n} for g, n in by_group.items()}
        self.design_by_take = {
            t: {"files": len(take_bases[t]), "copies": take_copies[t]} for t in take_copies
        }
        self.groups_by_folder = {f: sorted(gs) for f, gs in folder_groups.items()}
        self.files_by_folder = dict(folder_files)

    # ---------------------------------------------------- load-time checks

    def load_time_checks(self, share_tolerance: float = 1.5) -> Dict[str, object]:
        """Check 1 (manifest vs design) and check 2's load-time cardinality leg.

        Evaluated at step 0, before a single draw. A red here is an EXPORT
        BUG: the studio's share/attribution machinery balanced or labeled the
        wrong population. Triggers: a design group with zero manifest rows, a
        rows group absent from design, or a per-group file share off its
        design share beyond *share_tolerance* relative. The cardinality leg:
        a folder pooling many files under ONE manifest group while design
        lists groups that have no rows anywhere.

        Without a ``design_shares`` header (older manifests) the manifest is
        its own design; the vs-design triggers cannot fire and the report
        says so via ``design_source``.
        """
        if not self.loaded:
            return {"status": "no_manifest", "design_source": None, "findings": []}
        findings: List[dict] = []
        rows_share = {
            g: d["files"] / self.total_files
            for g, d in self.design_by_group.items()
        } if self.total_files else {}

        design_source = "design_shares" if self.design_shares_present else "manifest_fallback"
        if self.design_shares_present:
            for g, share in sorted(self.design_shares.items()):
                if g not in self.design_by_group:
                    findings.append({
                        "severity": "red",
                        "check": "design_group_zero_rows",
                        "detail": f"design group '{g}' (share {share:.3f}) has zero manifest rows",
                    })
            for g in sorted(self.design_by_group):
                if g not in self.design_shares:
                    findings.append({
                        "severity": "red",
                        "check": "rows_group_absent_from_design",
                        "detail": f"manifest group '{g}' ({self.design_by_group[g]['files']} files) absent from design_shares",
                    })
            for g, share in sorted(self.design_shares.items()):
                realized = rows_share.get(g)
                if realized is None or share <= 0:
                    continue
                rel = realized / share if share else None
                if rel is not None and (rel > share_tolerance or rel < 1.0 / share_tolerance):
                    findings.append({
                        "severity": "red",
                        "check": "file_share_vs_design_share",
                        "detail": (
                            f"group '{g}': manifest file share {realized:.3f} vs design share "
                            f"{share:.3f} ({rel:.2f}x, tolerance {share_tolerance}x)"
                        ),
                    })
            # check 2, load-time leg: single-group pooled folder + orphaned design groups
            orphaned = [g for g in self.design_shares if g not in self.design_by_group]
            if orphaned:
                for folder, groups in sorted(self.groups_by_folder.items()):
                    files = self.files_by_folder.get(folder, 0)
                    if files > 10 and len(groups) == 1:
                        findings.append({
                            "severity": "red",
                            "check": "single_group_pooled_folder",
                            "detail": (
                                f"folder '{folder}' pools {files} files under one manifest group "
                                f"'{groups[0]}' while design groups {orphaned} have no rows anywhere "
                                "-- export-attribution shape"
                            ),
                        })

        by_group = []
        for g in sorted(set(self.design_shares) | set(self.design_by_group)):
            by_group.append({
                "name": g,
                "files": self.design_by_group.get(g, {}).get("files", 0),
                "manifest_share": round(rows_share[g], 4) if g in rows_share else None,
                "design_share": round(self.design_shares[g], 4) if g in self.design_shares else None,
            })
        folders = [
            {
                "folder": f,
                "files": self.files_by_folder.get(f, 0),
                "manifest_groups": len(gs),
                "groups": gs,
            }
            for f, gs in sorted(self.groups_by_folder.items())
        ]
        return {
            "status": "red" if findings else "ok",
            "design_source": design_source,
            "findings": findings,
            "by_group": by_group,
            "folders": folders,
        }

    # ---------------------------------------------------------------- query

    @property
    def loaded(self) -> bool:
        return len(self._manifests) > 0

    def provenance(self) -> List[dict]:
        return [
            {
                "path": path,
                "session_id": m["session_id"],
                "hash": m["hash"],
                "rows": m["row_count"],
                "fps": m["fps"],
            }
            for path, m in sorted(self._manifests.items())
        ]

    def lookup(self, file_path: Optional[str]) -> Optional[ManifestAttribution]:
        if not file_path or not self._manifests:
            return None
        abs_path = os.path.abspath(file_path)
        if abs_path in self._path_cache:
            return self._path_cache[abs_path]
        result = self._resolve(abs_path)
        self._path_cache[abs_path] = result
        return result

    def _resolve(self, abs_path: str) -> Optional[ManifestAttribution]:
        # exact relative-path join against each manifest's directory
        for manifest_path, m in self._manifests.items():
            try:
                rel = os.path.relpath(abs_path, m["dir"]).replace("\\", "/")
            except ValueError:
                # different drive on Windows
                continue
            row = m["rows_by_file"].get(rel)
            if row is None:
                # "<folder>/<name>" for a file two levels below the manifest dir
                short = "/".join(abs_path.replace("\\", "/").split("/")[-2:])
                row = m["rows_by_file"].get(short)
            if row is not None:
                return self._attribution(row, m)
        # stem fallback: only when the stem is unique across all manifests
        stem = _stem(abs_path)
        matches = []
        for m in self._manifests.values():
            for file_key, row in m["rows_by_file"].items():
                if _stem(file_key) == stem:
                    matches.append((row, m))
        if len(matches) == 1:
            return self._attribution(*matches[0])
        return None

    def _attribution(self, row: dict, m: dict) -> ManifestAttribution:
        file_key = _s(row, "file").replace("\\", "/")
        duplicate_of = _s(row, "duplicate_of").replace("\\", "/")
        base_row = m["rows_by_file"].get(duplicate_of) if duplicate_of else None
        base_file = (duplicate_of or file_key)
        base_segment_id = _s(base_row, "segment_id") if base_row else _s(row, "segment_id")

        window_raw = row.get("window")
        try:
            window = int(window_raw) if window_raw not in (None, "") else None
        except (TypeError, ValueError):
            window = None
        weight_raw = row.get("weight")
        try:
            weight = float(weight_raw) if weight_raw not in (None, "") else None
        except (TypeError, ValueError):
            weight = None

        return ManifestAttribution(
            file=file_key,
            segment_id=_s(row, "segment_id"),
            semantic_group=_s(row, "semantic_group"),
            semantic_group_id=_s(row, "semantic_group_id"),
            phase_type=_s(row, "phase_type"),
            phase_id=_s(row, "phase_id"),
            level=_s(row, "level"),
            source_take=_s(row, "source_take"),
            profile_id=_s(row, "profile_id"),
            profile_version=_s(row, "profile_version"),
            window=window,
            weight=weight,
            duplicate_of=duplicate_of,
            manifest_hash=m["hash"],
            session_id=m["session_id"],
            base_file=base_file,
            base_segment_id=base_segment_id or _s(row, "segment_id"),
        )
