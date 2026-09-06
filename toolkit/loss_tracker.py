"""
Structured loss logging for AI-Toolkit.

Provides per-sample loss tracking with rolling aggregates, breakdowns by
dataset group and noise bucket, and periodic "worst videos" reports.

Enabled via `structured_loss: true` in the logging config.  Zero overhead
when disabled.
"""

import dataclasses
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, IO, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_metric_name(name: str) -> str:
    """Basic character normalization for W&B metric keys.

    NOT to be called directly for dataset group names — use
    ``LossTracker._get_sanitized_group()`` instead so collision detection
    works.
    """
    name = name.replace("/", "_").replace("\\", "_").replace(" ", "_").replace("-", "_")
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)
    return name


def _bucket_timestep(
    t: int,
    num_train_timesteps: int,
    edges: Optional[List[int]] = None,
) -> str:
    """Classify a timestep into low / mid / high noise bucket."""
    if edges is None or len(edges) != 2 or edges[0] >= edges[1]:
        edges = [num_train_timesteps // 3, 2 * num_train_timesteps // 3]
    if t < edges[0]:
        return "low"
    elif t < edges[1]:
        return "mid"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LossLoggingConfig:
    enabled: bool = False
    debug: bool = False           # include captions in JSONL
    write_jsonl: bool = True
    worst_every: int = 500        # periodic table frequency (steps)
    max_tracked_videos: int = 2000
    worst_min_count: int = 10     # min samples before video appears in worst list
    noise_bucket_edges: Optional[List[int]] = None
    loss_outlier_sigma: float = 2.0   # z-score threshold for [OUTLIER] flag
    worst_per_group_n: int = 5        # top-N worst clips per group
    caption_max_len: int = 80         # truncated caption length stored on VideoStats

    @classmethod
    def from_logging_config(cls, lc: Any) -> "LossLoggingConfig":
        edges = getattr(lc, "structured_loss_noise_bucket_edges", None)
        if edges is not None:
            edges = [int(x) for x in edges]
        return cls(
            enabled=getattr(lc, "structured_loss", False),
            debug=getattr(lc, "structured_loss_debug", False),
            write_jsonl=getattr(lc, "structured_loss_jsonl", True),
            worst_every=getattr(lc, "structured_loss_worst_every", 500),
            max_tracked_videos=getattr(lc, "structured_loss_max_videos", 2000),
            worst_min_count=getattr(lc, "structured_loss_worst_min_count", 10),
            noise_bucket_edges=edges,
            loss_outlier_sigma=getattr(lc, "structured_loss_outlier_sigma", 2.0),
            worst_per_group_n=getattr(lc, "structured_loss_worst_per_group", 5),
            caption_max_len=getattr(lc, "structured_loss_caption_len", 80),
        )


# ---------------------------------------------------------------------------
# Per-sample event
# ---------------------------------------------------------------------------

_RE_DUP = re.compile(r'_dup\d+$')
_RE_WINDOW = re.compile(r'_window\d+$')


def _normalize_source_id(source_id: str) -> str:
    """Strip ``_dupN`` suffix so duplicated clips merge into one entry."""
    return _RE_DUP.sub('', source_id)


def _parent_source_id(source_id: str) -> Optional[str]:
    """Return parent id if *source_id* has a ``_windowN`` suffix, else None."""
    normed = _normalize_source_id(source_id)
    if _RE_WINDOW.search(normed):
        return _RE_WINDOW.sub('', normed)
    return None


@dataclasses.dataclass
class LossEvent:
    step: int
    sample_idx: int
    loss_raw: float       # before any weighting
    loss_final: float     # after loss_multiplier / SNR weighting
    dataset_group: str    # original unsanitized name
    source_id: str        # 8-char hex hash
    source_path: str      # full path (debug only)
    is_reg: bool
    timestep: int
    timestep_bucket: str  # low / mid / high
    boundary_index: Optional[int]
    loss_multiplier: float
    token_dropout_rate: float
    caption_dropout_rate: float
    is_caption_dropped: bool
    caption: Optional[str] = None
    # DOP/blank preservation loss events: aggregated in a separate metric
    # family so they never contaminate the normal-loss stats or video stats
    is_preservation: bool = False
    # --- manifest attribution (toolkit/training_manifest.py); None when unjoined ---
    folder_group: Optional[str] = None      # folder-derived name; dataset_group may be semantic
    semantic_group: Optional[str] = None
    phase_type: Optional[str] = None
    phase_id: Optional[str] = None
    level: Optional[str] = None
    source_take: Optional[str] = None
    segment_id: Optional[str] = None
    duplicate_of: Optional[str] = None
    clip_key: Optional[str] = None          # stable clip identity (base segment id); dups collapse
    clip_name: Optional[str] = None         # display name for clip_key (base file stem)
    is_manifest_joined: bool = False
    # runtime sliding-window draw: the one signal only the trainer can log
    window_start: Optional[int] = None
    window_interval: Optional[int] = None


# ---------------------------------------------------------------------------
# EMA scalar
# ---------------------------------------------------------------------------

class EMAScalar:
    __slots__ = ("alpha", "value", "_initialized")

    def __init__(self, span: int) -> None:
        self.alpha = 2.0 / (span + 1)
        self.value = 0.0
        self._initialized = False

    def update(self, x: float) -> None:
        if not self._initialized:
            self.value = x
            self._initialized = True
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value


# ---------------------------------------------------------------------------
# Per-video rolling stats
# ---------------------------------------------------------------------------

class VideoStats:
    __slots__ = (
        "source_id",
        "dataset_group",
        "dataset_group_key",
        "is_reg",
        "losses",
        "losses_raw",
        "idx",
        "count",
        "sum",
        "sum_raw",
        "total_count",
        "last_seen_step",
        "caption",
        "source_path",
        "segment_id",
        "trend_means",
        "trend_steps",
        "was_outlier",
    )

    def __init__(
        self,
        source_id: str,
        dataset_group: str,
        dataset_group_key: str,
        is_reg: bool,
        window: int = 50,
    ) -> None:
        self.source_id = source_id
        self.dataset_group = dataset_group        # original human-friendly name
        self.dataset_group_key = dataset_group_key  # sanitized metric key
        self.is_reg = is_reg
        self.losses: List[float] = [0.0] * window
        self.losses_raw: List[float] = [0.0] * window
        self.idx = 0
        self.count = 0
        self.sum = 0.0
        self.sum_raw = 0.0
        self.total_count = 0
        self.last_seen_step = 0
        self.caption: Optional[str] = None
        # media file path so the UI can play the clip from a worst-list row
        self.source_path: Optional[str] = None
        # manifest segment id (stable across re-exports); None for unjoined clips
        self.segment_id: Optional[str] = None
        self.trend_means: List[float] = []
        self.trend_steps: List[int] = []
        self.was_outlier: bool = False

    def add(self, loss_final: float, loss_raw: float, step: int) -> None:
        if self.count >= len(self.losses):
            self.sum -= self.losses[self.idx]
            self.sum_raw -= self.losses_raw[self.idx]
        self.losses[self.idx] = loss_final
        self.losses_raw[self.idx] = loss_raw
        self.sum += loss_final
        self.sum_raw += loss_raw
        self.idx = (self.idx + 1) % len(self.losses)
        self.count = min(self.count + 1, len(self.losses))
        self.total_count += 1
        self.last_seen_step = step

    @property
    def mean(self) -> float:
        return self.sum / max(self.count, 1)

    @property
    def mean_raw(self) -> float:
        return self.sum_raw / max(self.count, 1)

    @property
    def p90(self) -> float:
        if self.count == 0:
            return 0.0
        vals = sorted(self.losses[: self.count])
        k = max(1, math.ceil(0.9 * self.count))
        return vals[k - 1]

    def record_trend(self, step: int) -> None:
        """Snapshot current mean for trend tracking. Skips if already recorded at this step."""
        if self.trend_steps and self.trend_steps[-1] == step:
            return
        self.trend_means.append(self.mean)
        self.trend_steps.append(step)
        if len(self.trend_means) > 5:
            self.trend_means.pop(0)
            self.trend_steps.pop(0)

    @property
    def trend_slope(self) -> float:
        """Normalized linear regression slope over trend_means.

        Returns slope / abs(y_mean) for scale-independent classification.
        Returns 0.0 if fewer than 2 data points.
        """
        n = len(self.trend_means)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(self.trend_means) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(self.trend_means))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den == 0:
            return 0.0
        raw_slope = num / den
        return raw_slope / max(abs(y_mean), 1e-6)

    def trend_direction(self, min_count: int) -> str:
        """Classify trend as up/down/flat/~ based on normalized slope."""
        if self.total_count < min_count or len(self.trend_means) < 2:
            return "~"
        slope = self.trend_slope
        if slope > 0.01:
            return "up"
        elif slope < -0.01:
            return "down"
        return "flat"

    @property
    def trend_delta(self) -> float:
        """Absolute change from oldest to newest trend point."""
        if len(self.trend_means) < 2:
            return 0.0
        return self.trend_means[-1] - self.trend_means[0]


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class LossTracker:
    def __init__(self, config: LossLoggingConfig, save_root: str) -> None:
        self.config = config
        self.enabled = config.enabled

        # Rolling EMAs for aggregate loss
        self._ema_50 = EMAScalar(span=50)
        self._ema_200 = EMAScalar(span=200)
        self._ema_1000 = EMAScalar(span=1000)

        # Reg/concept split EMAs (keyed by is_reg bool)
        self._reg_ema_50: Dict[bool, EMAScalar] = {False: EMAScalar(50), True: EMAScalar(50)}
        self._reg_ema_200: Dict[bool, EMAScalar] = {False: EMAScalar(200), True: EMAScalar(200)}
        self._reg_ema_1000: Dict[bool, EMAScalar] = {False: EMAScalar(1000), True: EMAScalar(1000)}

        # Per-group EMAs split by is_reg: (sanitized_key, is_reg) → EMA
        self._group_emas_by_reg: Dict[Tuple[str, bool], EMAScalar] = {}
        self._group_sample_counts_by_reg: Dict[Tuple[str, bool], int] = defaultdict(int)
        # pre-multiplier loss EMAs and the group's loss_multiplier, so the
        # snapshot can show cross-group-comparable numbers
        self._group_raw_emas_by_reg: Dict[Tuple[str, bool], EMAScalar] = {}
        self._group_multiplier_by_reg: Dict[Tuple[str, bool], float] = {}

        # Per-noise-bucket split: (bucket, is_reg) → EMA
        self._bucket_emas_by_reg: Dict[Tuple[str, bool], EMAScalar] = {}
        self._bucket_sample_counts_by_reg: Dict[Tuple[str, bool], int] = defaultdict(int)

        # Per-boundary split: (bidx, is_reg) → EMA
        self._boundary_emas_by_reg: Dict[Tuple[int, bool], EMAScalar] = {}
        self._boundary_sample_counts_by_reg: Dict[Tuple[int, bool], int] = defaultdict(int)

        # Per-(group, bucket, is_reg) matrix EMAs for cross-tabulation
        self._matrix_emas_by_reg: Dict[Tuple[str, str, bool], EMAScalar] = {}

        # Preservation-loss aggregates (DOP / blank prompt preservation).
        # Same key structure as the normal-loss families above, but kept fully
        # separate so preservation pressure is measurable per group / noise
        # bucket / expert boundary without touching the normal stats.
        self._pres_ema_by_reg: Dict[bool, EMAScalar] = {False: EMAScalar(200), True: EMAScalar(200)}
        self._pres_group_emas_by_reg: Dict[Tuple[str, bool], EMAScalar] = {}
        self._pres_group_sample_counts_by_reg: Dict[Tuple[str, bool], int] = defaultdict(int)
        self._pres_bucket_emas_by_reg: Dict[Tuple[str, bool], EMAScalar] = {}
        self._pres_bucket_sample_counts_by_reg: Dict[Tuple[str, bool], int] = defaultdict(int)
        self._pres_boundary_emas_by_reg: Dict[Tuple[int, bool], EMAScalar] = {}
        self._pres_boundary_sample_counts_by_reg: Dict[Tuple[int, bool], int] = defaultdict(int)
        self._pres_matrix_emas_by_reg: Dict[Tuple[str, str, bool], EMAScalar] = {}
        self._has_preservation_data = False

        # Manifest join (toolkit/training_manifest.ManifestIndex), set by the
        # trainer once dataset configs are known. None -> folder attribution only.
        self.manifest: Any = None
        # Provenance echoed into the snapshot header and the first JSONL record
        self.run_info: Dict[str, Any] = {}
        self._jsonl_wrote_provenance = False
        self.current_epoch: Optional[int] = None

        # Manifest cohort axes: phase_type and level (only joined events carry them)
        self._phase_emas_by_reg: Dict[Tuple[str, bool], EMAScalar] = {}
        self._phase_sample_counts_by_reg: Dict[Tuple[str, bool], int] = defaultdict(int)
        self._level_emas_by_reg: Dict[Tuple[str, bool], EMAScalar] = {}
        self._level_sample_counts_by_reg: Dict[Tuple[str, bool], int] = defaultdict(int)

        # Check 1: realized exposure per semantic group / source take (normal events)
        self._exposure_group: Dict[str, Dict[str, int]] = {}
        self._exposure_take: Dict[str, Dict[str, int]] = {}
        self._exposure_total_draws = 0
        self._exposure_joined_draws = 0

        # Check 2: same-semantic-group adjacency per pooled folder per epoch
        self._interleave: Dict[Tuple[str, int], Dict[str, Any]] = {}

        # Whether any reg data has been seen
        self._has_reg_data = False

        # Cumulative sample totals by is_reg
        self._samples_total_by_reg: Dict[bool, int] = {False: 0, True: 0}

        # Per-video tracking (bounded), keyed by (source_id, is_reg)
        self._video_stats: Dict[Tuple[str, bool], VideoStats] = {}

        # Collision-safe name registry: original → sanitized key
        self._sanitized_names: Dict[str, str] = {}
        # Reverse lookup: sanitized key → original (first registrant)
        self._sanitized_reverse: Dict[str, str] = {}

        # Cached group stats (rebuilt at snapshot time)
        self._cached_group_stats: Dict[Tuple[str, bool], Dict[str, Any]] = {}

        # Step-level buffer
        self._current_step_events: List[LossEvent] = []

        # External state (set by caller before commit_step)
        self.last_grad_norm: float = 0.0
        self.last_lr: float = 0.0

        # Internal timing
        self._last_commit_time: float = time.time()

        # JSONL
        self._jsonl_path = os.path.join(save_root, "loss_events.jsonl") if save_root else ""
        self._jsonl_file: Optional[IO] = None
        self._jsonl_lines_since_flush: int = 0

        # Run identity
        self.run_id = f"{int(time.time())}_{os.getpid()}"

        # Banner
        self._banner_printed = False

    # -----------------------------------------------------------------------
    # Collision-safe group name registry
    # -----------------------------------------------------------------------

    def _get_sanitized_group(self, original_name: str) -> str:
        if original_name in self._sanitized_names:
            return self._sanitized_names[original_name]
        sanitized = _sanitize_metric_name(original_name)
        # Check for collision
        if sanitized in self._sanitized_reverse and self._sanitized_reverse[sanitized] != original_name:
            suffix = hashlib.md5(original_name.encode()).hexdigest()[:4]
            sanitized = f"{sanitized}_{suffix}"
        self._sanitized_names[original_name] = sanitized
        self._sanitized_reverse[sanitized] = original_name
        return sanitized

    # -----------------------------------------------------------------------
    # Manifest attribution
    # -----------------------------------------------------------------------

    def attribution_for(self, file_item: Any, folder_group: str) -> Dict[str, Any]:
        """LossEvent attribution kwargs for a file item.

        ``dataset_group`` becomes the manifest's semantic group when the file
        joins (every existing group-keyed family turns semantic with no other
        change); it stays the folder-derived name otherwise. ``folder_group``
        always keeps the folder so folder-scoped checks still work.
        """
        fields: Dict[str, Any] = {
            "dataset_group": folder_group,
            "folder_group": folder_group,
            "window_start": getattr(file_item, "window_start", None),
            "window_interval": getattr(file_item, "window_interval", None),
        }
        manifest = self.manifest
        if manifest is None:
            return fields
        try:
            a = manifest.lookup(getattr(file_item, "path", None))
        except Exception:
            a = None
        if a is None:
            return fields
        fields.update({
            "dataset_group": a.semantic_group or folder_group,
            "semantic_group": a.semantic_group or None,
            "phase_type": a.phase_type or None,
            "phase_id": a.phase_id or None,
            "level": a.level or None,
            "source_take": a.source_take or None,
            "segment_id": a.segment_id or None,
            "duplicate_of": a.duplicate_of or None,
            "clip_key": a.base_segment_id or a.base_file or None,
            "clip_name": a.clip_name or None,
            "is_manifest_joined": True,
        })
        return fields

    # -----------------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------------

    def record_sample_loss(self, event: LossEvent) -> None:
        self._current_step_events.append(event)

    # -----------------------------------------------------------------------
    # Per-step commit
    # -----------------------------------------------------------------------

    def commit_step(
        self,
        step: int,
        optimizer_stepped: bool = True,
        epoch: Optional[int] = None,
    ) -> Dict[str, float]:
        if epoch is not None:
            self.current_epoch = epoch
        # Always update timing so step_time_ms stays meaningful
        now = time.time()
        step_time_ms = (now - self._last_commit_time) * 1000.0
        self._last_commit_time = now

        if not self._current_step_events:
            return {}

        # Print banner once
        if not self._banner_printed:
            self._print_banner()
            self._banner_printed = True

        # Snapshot and clear
        all_events = self._current_step_events
        self._current_step_events = []

        # Preservation events feed their own metric family; everything below
        # that reads `events` (EMAs, video stats, loss_mean) is normal-loss only
        events = [e for e in all_events if not e.is_preservation]
        pres_events = [e for e in all_events if e.is_preservation]

        if not events:
            # preservation-only step (shouldn't happen in practice): still
            # aggregate preservation and write nothing else
            metrics = {}
            if pres_events:
                self._aggregate_preservation(pres_events, metrics)
            return metrics

        # Aggregate loss
        mean_loss_raw = sum(e.loss_raw for e in events) / len(events)
        mean_loss_final = sum(e.loss_final for e in events) / len(events)

        self._ema_50.update(mean_loss_final)
        self._ema_200.update(mean_loss_final)
        self._ema_1000.update(mean_loss_final)

        metrics: Dict[str, float] = {
            "loss_tracker/ema_50": self._ema_50.value,
            "loss_tracker/ema_200": self._ema_200.value,
            "loss_tracker/ema_1000": self._ema_1000.value,
            "loss_tracker/grad_norm": self.last_grad_norm,
            "loss_tracker/step_time_ms": step_time_ms,
        }

        # ---- Partition events by is_reg ----
        events_by_reg: Dict[bool, List[LossEvent]] = defaultdict(list)
        for e in events:
            events_by_reg[e.is_reg].append(e)
        if True in events_by_reg:
            self._has_reg_data = True

        for is_reg, split_events in events_by_reg.items():
            reg_tag = "reg" if is_reg else "concept"
            split_mean = sum(e.loss_final for e in split_events) / len(split_events)

            # Cumulative totals
            self._samples_total_by_reg[is_reg] += len(split_events)

            # Overall reg/concept EMA
            self._reg_ema_50[is_reg].update(split_mean)
            self._reg_ema_200[is_reg].update(split_mean)
            self._reg_ema_1000[is_reg].update(split_mean)
            metrics[f"loss_by_reg_ema/{reg_tag}"] = self._reg_ema_200[is_reg].value
            metrics[f"samples_by_reg/{reg_tag}"] = float(len(split_events))

            # Per-group breakdown (concept = default keys, reg = _reg_ keys)
            group_events: Dict[str, List[LossEvent]] = defaultdict(list)
            for e in split_events:
                group_events[e.dataset_group].append(e)
            for group, evs in group_events.items():
                losses = [e.loss_final for e in evs]
                g_key = self._get_sanitized_group(group)
                key = (g_key, is_reg)
                if key not in self._group_emas_by_reg:
                    self._group_emas_by_reg[key] = EMAScalar(span=200)
                    self._group_raw_emas_by_reg[key] = EMAScalar(span=200)
                self._group_emas_by_reg[key].update(sum(losses) / len(losses))
                self._group_raw_emas_by_reg[key].update(
                    sum(e.loss_raw for e in evs) / len(evs)
                )
                self._group_multiplier_by_reg[key] = evs[-1].loss_multiplier
                self._group_sample_counts_by_reg[key] += len(losses)
                if is_reg:
                    metrics[f"loss_by_group_reg_ema/{g_key}"] = self._group_emas_by_reg[key].value
                    metrics[f"samples_by_group_reg/{g_key}"] = float(len(losses))
                else:
                    metrics[f"loss_by_group_ema/{g_key}"] = self._group_emas_by_reg[key].value
                    metrics[f"samples_by_group/{g_key}"] = float(len(losses))

            # Per-phase / per-level breakdown (manifest axes; only joined events carry them)
            for attr, emas, counts, key_prefix in (
                ("phase_type", self._phase_emas_by_reg, self._phase_sample_counts_by_reg, "loss_by_phase"),
                ("level", self._level_emas_by_reg, self._level_sample_counts_by_reg, "loss_by_level"),
            ):
                axis_losses: Dict[str, List[float]] = defaultdict(list)
                for e in split_events:
                    val = getattr(e, attr)
                    if val:
                        axis_losses[val].append(e.loss_final)
                for name, losses in axis_losses.items():
                    a_key = self._get_sanitized_group(name)
                    key = (a_key, is_reg)
                    if key not in emas:
                        emas[key] = EMAScalar(span=200)
                    emas[key].update(sum(losses) / len(losses))
                    counts[key] += len(losses)
                    metric_key = f"{key_prefix}_reg_ema/{a_key}" if is_reg else f"{key_prefix}_ema/{a_key}"
                    metrics[metric_key] = emas[key].value

            # Per-noise-bucket breakdown
            bucket_losses: Dict[str, List[float]] = defaultdict(list)
            for e in split_events:
                bucket_losses[e.timestep_bucket].append(e.loss_final)
            for bucket, losses in bucket_losses.items():
                key = (bucket, is_reg)
                if key not in self._bucket_emas_by_reg:
                    self._bucket_emas_by_reg[key] = EMAScalar(span=200)
                self._bucket_emas_by_reg[key].update(sum(losses) / len(losses))
                self._bucket_sample_counts_by_reg[key] += len(losses)
                if is_reg:
                    metrics[f"loss_by_noise_reg_ema/{bucket}"] = self._bucket_emas_by_reg[key].value
                    metrics[f"samples_by_noise_reg/{bucket}"] = float(len(losses))
                else:
                    metrics[f"loss_by_noise_ema/{bucket}"] = self._bucket_emas_by_reg[key].value
                    metrics[f"samples_by_noise/{bucket}"] = float(len(losses))

            # Per-boundary breakdown (multistage only)
            boundary_losses: Dict[int, List[float]] = defaultdict(list)
            for e in split_events:
                if e.boundary_index is not None:
                    boundary_losses[e.boundary_index].append(e.loss_final)
            for bidx, losses in boundary_losses.items():
                key = (bidx, is_reg)
                if key not in self._boundary_emas_by_reg:
                    self._boundary_emas_by_reg[key] = EMAScalar(span=200)
                self._boundary_emas_by_reg[key].update(sum(losses) / len(losses))
                self._boundary_sample_counts_by_reg[key] += len(losses)
                if is_reg:
                    metrics[f"loss_by_boundary_reg/{bidx}"] = self._boundary_emas_by_reg[key].value
                    metrics[f"samples_by_boundary_reg/{bidx}"] = float(len(losses))
                else:
                    metrics[f"loss_by_boundary/{bidx}"] = self._boundary_emas_by_reg[key].value
                    metrics[f"samples_by_boundary/{bidx}"] = float(len(losses))

            # Group × bucket matrix EMAs
            for e in split_events:
                g_key = self._get_sanitized_group(e.dataset_group)
                key = (g_key, e.timestep_bucket, is_reg)
                if key not in self._matrix_emas_by_reg:
                    self._matrix_emas_by_reg[key] = EMAScalar(span=200)
                self._matrix_emas_by_reg[key].update(e.loss_final)

        # Re-emit every initialized split EMA so chart lines stay continuous when
        # reg/concept events alternate across steps. EMA values persist between
        # updates, so writing them on steps without new samples is meaningful.
        for is_reg in (False, True):
            if self._reg_ema_200[is_reg]._initialized:
                reg_tag = "reg" if is_reg else "concept"
                metrics[f"loss_by_reg_ema/{reg_tag}"] = self._reg_ema_200[is_reg].value
        for (g_key, is_reg), ema in self._group_emas_by_reg.items():
            if not ema._initialized:
                continue
            metric_key = (
                f"loss_by_group_reg_ema/{g_key}" if is_reg else f"loss_by_group_ema/{g_key}"
            )
            metrics.setdefault(metric_key, ema.value)
        for (bucket, is_reg), ema in self._bucket_emas_by_reg.items():
            if not ema._initialized:
                continue
            metric_key = (
                f"loss_by_noise_reg_ema/{bucket}" if is_reg else f"loss_by_noise_ema/{bucket}"
            )
            metrics.setdefault(metric_key, ema.value)
        for (bidx, is_reg), ema in self._boundary_emas_by_reg.items():
            if not ema._initialized:
                continue
            metric_key = (
                f"loss_by_boundary_reg/{bidx}" if is_reg else f"loss_by_boundary/{bidx}"
            )
            metrics.setdefault(metric_key, ema.value)
        for emas, key_prefix in (
            (self._phase_emas_by_reg, "loss_by_phase"),
            (self._level_emas_by_reg, "loss_by_level"),
        ):
            for (a_key, is_reg), ema in emas.items():
                if not ema._initialized:
                    continue
                metric_key = f"{key_prefix}_reg_ema/{a_key}" if is_reg else f"{key_prefix}_ema/{a_key}"
                metrics.setdefault(metric_key, ema.value)

        # Ratio metrics (only when both sides have data this step)
        concept_count = len(events_by_reg.get(False, []))
        reg_count = len(events_by_reg.get(True, []))
        if concept_count > 0 and reg_count > 0:
            metrics["loss_ratio/reg_to_concept"] = (
                self._reg_ema_200[True].value
                / max(self._reg_ema_200[False].value, 1e-10)
            )
            metrics["samples_ratio/reg_to_concept"] = float(reg_count) / float(concept_count)

        # Manifest checks 1 and 2 (normal-loss draws, in draw order)
        self._update_exposure_and_interleaving(events, self.current_epoch)

        # Preservation-loss aggregation (separate metric family)
        if pres_events:
            self._aggregate_preservation(pres_events, metrics)

        # Per-video tracking
        for e in events:
            self._update_video_stats(e, step)

        # JSONL (both normal and preservation samples; the latter flagged)
        if self.config.write_jsonl:
            self._write_jsonl(all_events, step, mean_loss_final, step_time_ms, optimizer_stepped)

        return metrics

    def _aggregate_preservation(self, pres_events: List[LossEvent], metrics: Dict[str, float]) -> None:
        """Update preservation EMAs and emit `preservation_*` metrics.

        Mirrors the normal-loss family structure (reg split, per-group,
        per-noise-bucket, per-boundary, group×bucket matrix) using
        ``loss_final`` (i.e. after the preservation multiplier).
        Initialized EMAs are re-emitted every call so chart lines stay
        continuous, matching the normal-loss behavior.
        """
        self._has_preservation_data = True

        by_reg: Dict[bool, List[LossEvent]] = defaultdict(list)
        for e in pres_events:
            by_reg[e.is_reg].append(e)

        for is_reg, split_events in by_reg.items():
            reg_suffix = "_reg" if is_reg else ""
            reg_tag = "reg" if is_reg else "concept"
            split_mean = sum(e.loss_final for e in split_events) / len(split_events)

            self._pres_ema_by_reg[is_reg].update(split_mean)
            metrics[f"preservation_ema/{reg_tag}"] = self._pres_ema_by_reg[is_reg].value

            group_losses: Dict[str, List[float]] = defaultdict(list)
            for e in split_events:
                group_losses[e.dataset_group].append(e.loss_final)
            for group, losses in group_losses.items():
                g_key = self._get_sanitized_group(group)
                key = (g_key, is_reg)
                if key not in self._pres_group_emas_by_reg:
                    self._pres_group_emas_by_reg[key] = EMAScalar(span=200)
                self._pres_group_emas_by_reg[key].update(sum(losses) / len(losses))
                self._pres_group_sample_counts_by_reg[key] += len(losses)
                metrics[f"preservation_by_group{reg_suffix}_ema/{g_key}"] = self._pres_group_emas_by_reg[key].value

            bucket_losses: Dict[str, List[float]] = defaultdict(list)
            for e in split_events:
                bucket_losses[e.timestep_bucket].append(e.loss_final)
            for bucket, losses in bucket_losses.items():
                key = (bucket, is_reg)
                if key not in self._pres_bucket_emas_by_reg:
                    self._pres_bucket_emas_by_reg[key] = EMAScalar(span=200)
                self._pres_bucket_emas_by_reg[key].update(sum(losses) / len(losses))
                self._pres_bucket_sample_counts_by_reg[key] += len(losses)
                metrics[f"preservation_by_noise{reg_suffix}_ema/{bucket}"] = self._pres_bucket_emas_by_reg[key].value

            boundary_losses: Dict[int, List[float]] = defaultdict(list)
            for e in split_events:
                if e.boundary_index is not None:
                    boundary_losses[e.boundary_index].append(e.loss_final)
            for bidx, losses in boundary_losses.items():
                key = (bidx, is_reg)
                if key not in self._pres_boundary_emas_by_reg:
                    self._pres_boundary_emas_by_reg[key] = EMAScalar(span=200)
                self._pres_boundary_emas_by_reg[key].update(sum(losses) / len(losses))
                self._pres_boundary_sample_counts_by_reg[key] += len(losses)
                metrics[f"preservation_by_boundary{reg_suffix}/{bidx}"] = self._pres_boundary_emas_by_reg[key].value

            for e in split_events:
                g_key = self._get_sanitized_group(e.dataset_group)
                key = (g_key, e.timestep_bucket, is_reg)
                if key not in self._pres_matrix_emas_by_reg:
                    self._pres_matrix_emas_by_reg[key] = EMAScalar(span=200)
                self._pres_matrix_emas_by_reg[key].update(e.loss_final)

        # Re-emit initialized EMAs for line continuity across alternating steps
        for is_reg in (False, True):
            if self._pres_ema_by_reg[is_reg]._initialized:
                reg_tag = "reg" if is_reg else "concept"
                metrics.setdefault(f"preservation_ema/{reg_tag}", self._pres_ema_by_reg[is_reg].value)
        for (g_key, is_reg), ema in self._pres_group_emas_by_reg.items():
            if ema._initialized:
                reg_suffix = "_reg" if is_reg else ""
                metrics.setdefault(f"preservation_by_group{reg_suffix}_ema/{g_key}", ema.value)
        for (bucket, is_reg), ema in self._pres_bucket_emas_by_reg.items():
            if ema._initialized:
                reg_suffix = "_reg" if is_reg else ""
                metrics.setdefault(f"preservation_by_noise{reg_suffix}_ema/{bucket}", ema.value)
        for (bidx, is_reg), ema in self._pres_boundary_emas_by_reg.items():
            if ema._initialized:
                reg_suffix = "_reg" if is_reg else ""
                metrics.setdefault(f"preservation_by_boundary{reg_suffix}/{bidx}", ema.value)

    # -----------------------------------------------------------------------
    # Video stats
    # -----------------------------------------------------------------------

    def _ensure_and_update(
        self, source_id: str, event: LossEvent, step: int,
        display: Optional[str] = None, segment_id: Optional[str] = None,
    ) -> None:
        """Create or update a VideoStats entry keyed by *source_id*.

        *display* is the human-readable id shown in worst-lists (defaults to
        the key); *segment_id* is the manifest's stable identity when known.
        """
        vid_key = (source_id, event.is_reg)
        if vid_key not in self._video_stats:
            if len(self._video_stats) >= self.config.max_tracked_videos:
                evict_key = min(
                    self._video_stats,
                    key=lambda k: (
                        self._video_stats[k].total_count,
                        self._video_stats[k].last_seen_step,
                    ),
                )
                del self._video_stats[evict_key]
            g_key = self._get_sanitized_group(event.dataset_group)
            self._video_stats[vid_key] = VideoStats(
                display or source_id, event.dataset_group, g_key, event.is_reg, window=50
            )
            self._video_stats[vid_key].segment_id = segment_id
        vs = self._video_stats[vid_key]
        vs.add(event.loss_final, event.loss_raw, step)
        if vs.caption is None and event.caption:
            vs.caption = " ".join(event.caption.split())
        if vs.source_path is None and event.source_path:
            vs.source_path = event.source_path

    def _update_video_stats(self, event: LossEvent, step: int) -> None:
        if event.is_manifest_joined and event.clip_key:
            # manifest identity: share copies collapse onto their base segment
            # (via duplicate_of), and the per-take aggregate replaces the
            # _windowN parent. Unjoined files below keep the suffix-regex path.
            self._ensure_and_update(
                event.clip_key, event, step,
                display=event.clip_name or event.source_id,
                segment_id=event.clip_key,
            )
            if event.source_take:
                take_key = f"take:{event.source_take}"
                self._ensure_and_update(take_key, event, step, display=take_key)
            return

        norm_id = _normalize_source_id(event.source_id)

        # Per-clip entry (dups merge, windows stay separate)
        self._ensure_and_update(norm_id, event, step)

        # Parent aggregate for windowed clips
        parent_id = _parent_source_id(event.source_id)
        if parent_id is not None and parent_id != norm_id:
            self._ensure_and_update(parent_id, event, step)

    def get_worst_videos(
        self, top_n: int = 20, is_reg: Optional[bool] = False,
    ) -> List[Tuple[str, str, float, float, float, int]]:
        """Return worst videos by rolling mean loss_raw.

        Sorted on the pre-multiplier loss so groups with a loss_multiplier
        (e.g. 0.5-damped datasets) compete on the same scale instead of being
        silently pushed out of the list.

        Args:
            top_n: Maximum number of videos to return.
            is_reg: Filter by reg status.  False = concept only (default),
                    True = reg only, None = all videos.

        Returns list of (source_id, dataset_group, mean_final, mean_raw,
        p90_final, total_count).  Filtered to total_count >= worst_min_count.
        """
        eligible = [
            v
            for v in self._video_stats.values()
            if v.total_count >= self.config.worst_min_count
            and (is_reg is None or v.is_reg == is_reg)
        ]
        eligible.sort(key=lambda v: v.mean_raw, reverse=True)
        result = []
        for v in eligible[:top_n]:
            result.append((
                v.source_id,
                v.dataset_group,
                v.mean,
                v.mean_raw,
                v.p90,
                v.total_count,
            ))
        return result

    # -----------------------------------------------------------------------
    # Group stats & extended diagnostics
    # -----------------------------------------------------------------------

    def _cache_group_stats(self) -> None:
        """Compute per-group stats from current clip rolling means.

        Stored as self._cached_group_stats: {(g_key, is_reg): {mean, std, min, max, clip_count}}.
        Called at the start of each snapshot cycle.
        """
        from collections import defaultdict as _dd
        clips_by_group: Dict[Tuple[str, bool], List[float]] = _dd(list)
        for v in self._video_stats.values():
            if v.total_count >= self.config.worst_min_count:
                clips_by_group[(v.dataset_group_key, v.is_reg)].append(v.mean)

        stats: Dict[Tuple[str, bool], Dict[str, float]] = {}
        for key, means in clips_by_group.items():
            n = len(means)
            avg = sum(means) / n
            if n >= 2:
                var = sum((m - avg) ** 2 for m in means) / (n - 1)
                std = math.sqrt(var)
            else:
                std = 0.0
            stats[key] = {
                "mean": avg,
                "std": std,
                "effective_std": max(std, avg * 0.05, 0.001),
                "min": min(means),
                "max": max(means),
                "clip_count": n,
            }
        self._cached_group_stats = stats

    def _record_all_trends(self, step: int) -> None:
        """Record trend snapshot for all eligible clips."""
        for vs in self._video_stats.values():
            if vs.total_count >= self.config.worst_min_count:
                vs.record_trend(step)

    def _compute_clip_diagnostics(self, vs: VideoStats) -> Dict[str, Any]:
        """Compute z-score, outlier status, trend, and loss_ratio for a single clip."""
        g_stats = self._cached_group_stats.get((vs.dataset_group_key, vs.is_reg))
        if g_stats:
            g_mean = g_stats["mean"]
            eff_std = g_stats["effective_std"]
            z = (vs.mean - g_mean) / eff_std
            loss_ratio = vs.mean / max(g_mean, 1e-6)
        else:
            z = 0.0
            g_mean = 0.0
            loss_ratio = 0.0

        # Outlier detection with hysteresis
        sigma = self.config.loss_outlier_sigma
        if not vs.was_outlier:
            is_outlier = z > sigma
        else:
            is_outlier = z >= sigma * 0.8
        is_new_outlier = is_outlier and not vs.was_outlier
        vs.was_outlier = is_outlier

        trend = vs.trend_direction(self.config.worst_min_count)

        return {
            "source_id": vs.source_id,
            "dataset_group": vs.dataset_group,
            "mean_loss_final": round(vs.mean, 6),
            "mean_loss_raw": round(vs.mean_raw, 6),
            "p90_loss_final": round(vs.p90, 6),
            "count": vs.total_count,
            "z_score": round(z, 2),
            "is_outlier": is_outlier,
            "is_new_outlier": is_new_outlier,
            "trend": trend,
            "trend_slope": round(vs.trend_slope, 6),
            "trend_delta": round(vs.trend_delta, 6),
            "loss_ratio": round(loss_ratio, 3),
            "caption": vs.caption,
            "source_path": vs.source_path,
            "segment_id": vs.segment_id,
        }

    def get_worst_videos_extended(
        self, top_n: int = 20, is_reg: Optional[bool] = False,
    ) -> List[Dict[str, Any]]:
        """Like get_worst_videos but returns dicts with z-score, trend, caption.

        Sorted by mean loss_raw (pre-multiplier) so multiplier-damped groups
        are not scale-excluded. Requires _cache_group_stats() first.
        """
        eligible = [
            v for v in self._video_stats.values()
            if v.total_count >= self.config.worst_min_count
            and (is_reg is None or v.is_reg == is_reg)
        ]
        eligible.sort(key=lambda v: v.mean_raw, reverse=True)
        return [self._compute_clip_diagnostics(v) for v in eligible[:top_n]]

    def get_worst_by_group(
        self, top_n: Optional[int] = None, is_reg: Optional[bool] = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return worst videos per group, sorted by z_score desc.

        Requires _cache_group_stats() to have been called first.
        """
        if top_n is None:
            top_n = self.config.worst_per_group_n

        eligible = [
            v for v in self._video_stats.values()
            if v.total_count >= self.config.worst_min_count
            and (is_reg is None or v.is_reg == is_reg)
        ]

        by_group: Dict[str, List[VideoStats]] = defaultdict(list)
        for v in eligible:
            by_group[v.dataset_group].append(v)

        result: Dict[str, List[Dict[str, Any]]] = {}
        for group, clips in sorted(by_group.items()):
            # Compute diagnostics for all clips in this group
            diags = [self._compute_clip_diagnostics(v) for v in clips]
            # Sort by z_score desc, then trend_slope desc, then mean desc
            diags.sort(key=lambda d: (-d["z_score"], -d.get("trend_slope", 0.0), -d["mean_loss_final"]))
            result[group] = diags[:top_n]

        return result

    # -----------------------------------------------------------------------
    # Periodic tables
    # -----------------------------------------------------------------------

    def _print_worst_videos_extended(self, step: int, label: str, worst: List[Dict[str, Any]]) -> None:
        """Print a worst-videos table with z-score, trend, ratio, caption."""
        if not worst:
            return
        trend_chars = {"up": "\u2191", "down": "\u2193", "flat": "\u2192", "~": "~"}
        print(f"\n[LossTracker] Worst {label} videos at step {step}:")
        print(f"  {'source_id':<10} {'group':<16} {'mean':>8} {'p90':>8} {'z':>8} {'r':>5} {'t':>2} {'cnt':>5}  caption")
        outlier_count = 0
        new_outlier_count = 0
        for w in worst:
            flag = ""
            z_suffix = ""
            if w.get("is_new_outlier"):
                flag = "  [NEW OUTLIER]"
                z_suffix = "!!"
                new_outlier_count += 1
                outlier_count += 1
            elif w.get("is_outlier"):
                flag = "  [OUTLIER]"
                z_suffix = "!"
                outlier_count += 1
            trend = trend_chars.get(w.get("trend", "~"), "~")
            cap = (w.get("caption") or "")[:40]
            if w.get("z_score") is not None:
                z_str = f"{w['z_score']:.1f}{z_suffix}"
                z_str = f"{z_str:>8}"
            else:
                z_str = "       -"
            r_str = f"{w['loss_ratio']:>5.2f}" if w.get("loss_ratio") else "    -"
            print(
                f"  {w['source_id']:<10} {w['dataset_group']:<16} "
                f"{w['mean_loss_final']:>8.5f} {w['p90_loss_final']:>8.5f} "
                f"{z_str} {r_str} {trend:>2} {w['count']:>5}  {cap}{flag}"
            )
        if outlier_count > 0:
            print(f"  [{outlier_count} outlier(s) (z > {self.config.loss_outlier_sigma}), {new_outlier_count} new]")

    def log_worst_videos_table(self, step: int, logger: Any) -> None:
        # Ensure group stats are cached for z-score computation
        self._record_all_trends(step)
        self._cache_group_stats()

        concept_ext = self.get_worst_videos_extended(top_n=20, is_reg=False)
        reg_ext = self.get_worst_videos_extended(top_n=20, is_reg=True) if self._has_reg_data else []
        if not concept_ext and not reg_ext:
            return

        self._print_worst_videos_extended(step, "concept", concept_ext)
        if reg_ext:
            self._print_worst_videos_extended(step, "reg", reg_ext)
        print()

        # W&B table (concept only, tuple format for compatibility)
        concept_tuples = self.get_worst_videos(top_n=20, is_reg=False)
        if concept_tuples and hasattr(logger, "_log"):
            try:
                import wandb

                table = wandb.Table(
                    columns=[
                        "source_id",
                        "dataset_group",
                        "mean_loss_final",
                        "mean_loss_raw",
                        "p90_loss_final",
                        "count",
                    ],
                    data=[
                        [sid, grp, mean_f, mean_r, p90, cnt]
                        for sid, grp, mean_f, mean_r, p90, cnt in concept_tuples
                    ],
                )
                logger._log({"worst_videos": table}, commit=False)
            except ImportError:
                pass

    def log_matrix_table(self, step: int, logger: Any) -> None:
        """Log group × noise bucket cross-tabulation as a W&B Table."""
        if not self._matrix_emas_by_reg:
            return

        # Build concept rows
        concept_rows = []
        for (g_key, b_key, is_reg), ema in sorted(self._matrix_emas_by_reg.items()):
            orig_name = self._sanitized_reverse.get(g_key, g_key)
            if not is_reg:
                concept_rows.append([orig_name, b_key, ema.value])

        # Console
        if concept_rows:
            print(f"\n[LossTracker] Loss matrix (concept) at step {step}:")
            print(f"  {'group':<20} {'bucket':<8} {'mean_loss':>10}")
            for g, b, v in concept_rows:
                print(f"  {g:<20} {b:<8} {v:>10.6f}")

        if self._has_reg_data:
            reg_rows = []
            for (g_key, b_key, is_reg), ema in sorted(self._matrix_emas_by_reg.items()):
                orig_name = self._sanitized_reverse.get(g_key, g_key)
                if is_reg:
                    reg_rows.append([orig_name, b_key, ema.value])
            if reg_rows:
                print(f"\n[LossTracker] Loss matrix (reg) at step {step}:")
                print(f"  {'group':<20} {'bucket':<8} {'mean_loss':>10}")
                for g, b, v in reg_rows:
                    print(f"  {g:<20} {b:<8} {v:>10.6f}")

        print()

        # W&B table (concept only)
        if concept_rows and hasattr(logger, "_log"):
            try:
                import wandb

                table = wandb.Table(
                    columns=["group", "noise_bucket", "mean_loss_final"],
                    data=concept_rows,
                )
                logger._log({"loss_matrix": table}, commit=False)
            except ImportError:
                pass

    def _build_summary_rows(self, is_reg: bool) -> list:
        """Build summary rows for a given reg status."""
        rows = []
        for (g_key, reg), g_ema in sorted(self._group_emas_by_reg.items()):
            if reg != is_reg:
                continue
            orig_name = self._sanitized_reverse.get(g_key, g_key)
            rows.append(["group", orig_name, g_ema.value, self._group_sample_counts_by_reg.get((g_key, is_reg), 0)])
        for (b_key, reg), b_ema in sorted(self._bucket_emas_by_reg.items()):
            if reg != is_reg:
                continue
            rows.append(["noise", b_key, b_ema.value, self._bucket_sample_counts_by_reg.get((b_key, is_reg), 0)])
        for (bidx, reg), b_ema in sorted(self._boundary_emas_by_reg.items()):
            if reg != is_reg:
                continue
            rows.append(["boundary", str(bidx), b_ema.value, self._boundary_sample_counts_by_reg.get((bidx, is_reg), 0)])
        for row_type, emas, counts in (
            ("phase", self._phase_emas_by_reg, self._phase_sample_counts_by_reg),
            ("level", self._level_emas_by_reg, self._level_sample_counts_by_reg),
        ):
            for (a_key, reg), ema in sorted(emas.items()):
                if reg != is_reg:
                    continue
                rows.append([row_type, self._sanitized_reverse.get(a_key, a_key), ema.value, counts.get((a_key, is_reg), 0)])
        return rows

    def log_summary_table(self, step: int, logger: Any) -> None:
        """At-a-glance health report: per-group and per-bucket stats with sample counts."""
        concept_rows = self._build_summary_rows(is_reg=False)
        reg_rows = self._build_summary_rows(is_reg=True) if self._has_reg_data else []

        if not concept_rows and not reg_rows:
            return

        if concept_rows:
            print(f"\n[LossTracker] Training summary (concept) at step {step}:")
            print(f"  {'type':<10} {'name':<20} {'ema_200':>10} {'samples':>8}")
            for typ, name, val, cnt in concept_rows:
                print(f"  {typ:<10} {name:<20} {val:>10.6f} {cnt:>8}")

        if reg_rows:
            print(f"\n[LossTracker] Training summary (reg) at step {step}:")
            print(f"  {'type':<10} {'name':<20} {'ema_200':>10} {'samples':>8}")
            for typ, name, val, cnt in reg_rows:
                print(f"  {typ:<10} {name:<20} {val:>10.6f} {cnt:>8}")

        if self._has_preservation_data:
            for is_reg, label in ((False, "concept"), (True, "reg")):
                rows = self._preservation_summary_snapshot(is_reg=is_reg)
                if not rows:
                    continue
                print(f"\n[LossTracker] Preservation summary ({label}) at step {step}:")
                print(f"  {'type':<10} {'name':<20} {'ema_200':>10} {'samples':>8}")
                for row in rows:
                    print(f"  {row['type']:<10} {row['name']:<20} {row['ema_200']:>10.6f} {row['sample_count']:>8}")

        print()

        # W&B table (concept only for default dashboard)
        if concept_rows and hasattr(logger, "_log"):
            try:
                import wandb

                table = wandb.Table(
                    columns=["type", "name", "ema_200", "sample_count"],
                    data=concept_rows,
                )
                logger._log({"training_summary": table}, commit=False)
            except ImportError:
                pass

    # -----------------------------------------------------------------------
    # Manifest checks: exposure (1), interleaving (2), provenance
    # -----------------------------------------------------------------------

    def _update_exposure_and_interleaving(self, events: List[LossEvent], epoch: Optional[int]) -> None:
        """Bookkeeping for checks 1 and 2 over this step's normal-loss events."""
        ep = epoch if epoch is not None else -1
        for e in events:
            self._exposure_total_draws += 1
            if not e.is_manifest_joined:
                continue
            self._exposure_joined_draws += 1
            if e.semantic_group:
                g = self._exposure_group.setdefault(e.semantic_group, {"draws": 0, "reg_draws": 0})
                g["draws"] += 1
                if e.is_reg:
                    g["reg_draws"] += 1
            if e.source_take:
                t = self._exposure_take.setdefault(e.source_take, {"draws": 0, "reg_draws": 0})
                t["draws"] += 1
                if e.is_reg:
                    t["reg_draws"] += 1
            # interleaving: same-semantic-group adjacency in a pooled folder's draw sequence
            if e.semantic_group and e.folder_group:
                key = (e.folder_group, ep)
                st = self._interleave.get(key)
                if st is None:
                    st = {"last": None, "pairs": 0, "same": 0, "counts": defaultdict(int)}
                    self._interleave[key] = st
                    self._prune_interleave(e.folder_group)
                if st["last"] is not None:
                    st["pairs"] += 1
                    if st["last"] == e.semantic_group:
                        st["same"] += 1
                st["last"] = e.semantic_group
                st["counts"][e.semantic_group] += 1

    def _prune_interleave(self, folder: str, keep: int = 20) -> None:
        epochs = sorted(ep for (f, ep) in self._interleave if f == folder)
        for ep in epochs[:-keep]:
            self._interleave.pop((folder, ep), None)

    @staticmethod
    def _r4(x: Optional[float]) -> Optional[float]:
        return None if x is None else round(x, 4)

    def _exposure_snapshot(self) -> Dict[str, Any]:
        """Check 1: realized draws per semantic group / source take vs design.

        Expected share is the manifest's file share (duplicates count as
        themselves: duplication IS the share system). The per-take view
        collapses duplicate_of on the design side (distinct base files) but
        counts every copy's draws, so a duplicated take shows its amplified
        realized share. Groups from reg datasets carry reg_draws so their
        lower draw rate (reg_every_n) can be read for what it is.
        """
        manifest = self.manifest
        joined = self._exposure_joined_draws
        total_files = getattr(manifest, "total_files", 0) if manifest is not None else 0
        design_g = getattr(manifest, "design_by_group", {}) if manifest is not None else {}
        design_t = getattr(manifest, "design_by_take", {}) if manifest is not None else {}
        epoch = self.current_epoch

        def ratio(realized: Optional[float], expected: Optional[float]) -> Optional[float]:
            if realized is None or not expected:
                return None
            return round(realized / expected, 3)

        by_group = []
        for name in sorted(set(design_g) | set(self._exposure_group)):
            files = design_g.get(name, {}).get("files", 0)
            counts = self._exposure_group.get(name, {"draws": 0, "reg_draws": 0})
            expected_share = files / total_files if total_files else None
            realized_share = counts["draws"] / joined if joined else None
            by_group.append({
                "name": name,
                "files": files,
                "draws": counts["draws"],
                "reg_draws": counts["reg_draws"],
                "expected_share": self._r4(expected_share),
                "realized_share": self._r4(realized_share),
                "ratio": ratio(realized_share, expected_share),
                "expected_draws": files * epoch if (epoch is not None and epoch > 0) else None,
            })

        total_copies = sum(d.get("copies", 0) for d in design_t.values())
        by_take = []
        for name in sorted(set(design_t) | set(self._exposure_take)):
            design = design_t.get(name, {"files": 0, "copies": 0})
            counts = self._exposure_take.get(name, {"draws": 0, "reg_draws": 0})
            expected_share = design["copies"] / total_copies if total_copies else None
            realized_share = counts["draws"] / joined if joined else None
            by_take.append({
                "name": name,
                "files": design["files"],
                "copies": design["copies"],
                "draws": counts["draws"],
                "reg_draws": counts["reg_draws"],
                "expected_share": self._r4(expected_share),
                "realized_share": self._r4(realized_share),
                "ratio": ratio(realized_share, expected_share),
            })

        return {
            "epoch": epoch,
            "total_draws": self._exposure_total_draws,
            "joined_draws": joined,
            "unjoined_draws": self._exposure_total_draws - joined,
            "manifest_files": total_files,
            "by_semantic_group": by_group,
            "by_source_take": by_take,
        }

    def _interleaving_snapshot(self) -> Dict[str, Any]:
        """Check 2: same-semantic-group adjacency per pooled folder per epoch.

        A shuffled baseline for the adjacency rate is sum(p_g^2) over the
        folder's group proportions. A sequential folder walk over grouped
        files reads as adjacency near 1.0 -- semantically clustered batches.
        """
        rows = []
        for (folder, ep), st in sorted(self._interleave.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            n = sum(st["counts"].values())
            pairs = st["pairs"]
            adjacency = st["same"] / pairs if pairs else None
            baseline = sum((c / n) ** 2 for c in st["counts"].values()) if n else None
            rt = (adjacency / baseline) if (adjacency is not None and baseline) else None
            flagged = bool(
                pairs >= 20
                and adjacency is not None and baseline is not None
                and adjacency - baseline > 0.1
                and rt is not None and rt > 1.5
            )
            rows.append({
                "folder": folder,
                "epoch": ep if ep >= 0 else None,
                "draws": n,
                "groups": len(st["counts"]),
                "adjacency_rate": self._r4(adjacency),
                "baseline_rate": self._r4(baseline),
                "ratio": round(rt, 3) if rt is not None else None,
                "flagged": flagged,
            })
        return {
            "epoch": self.current_epoch,
            "by_folder": rows,
            "any_flagged": any(r["flagged"] for r in rows),
        }

    def _provenance_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(self.run_info)
        out["noise_bucket_edges"] = self.config.noise_bucket_edges
        try:
            out["manifests"] = self.manifest.provenance() if self.manifest is not None else []
        except Exception:
            out["manifests"] = []
        return out

    # -----------------------------------------------------------------------
    # JSONL
    # -----------------------------------------------------------------------

    def _write_jsonl(
        self,
        events: List[LossEvent],
        step: int,
        loss_mean: float,
        step_time_ms: float,
        optimizer_stepped: bool,
    ) -> None:
        if not self._jsonl_path:
            return

        if self._jsonl_file is None:
            parent = os.path.dirname(os.path.abspath(self._jsonl_path))
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            self._jsonl_file = open(self._jsonl_path, "a", encoding="utf-8")

        groups = set()
        samples = []
        for e in events:
            groups.add(e.dataset_group)
            sample: Dict[str, Any] = {
                "source_id": e.source_id,
                "dataset_group": e.dataset_group,
                "loss_raw": round(e.loss_raw, 8),
                "loss_final": round(e.loss_final, 8),
                "timestep": e.timestep,
                "timestep_bucket": e.timestep_bucket,
                "boundary_index": e.boundary_index,
                "is_reg": e.is_reg,
                "is_caption_dropped": e.is_caption_dropped,
                "loss_multiplier": e.loss_multiplier,
            }
            if e.is_preservation:
                sample["is_preservation"] = True
            if e.caption is not None:
                sample["caption"] = e.caption
            # events now always carry the path (for the UI's clip player);
            # keep the JSONL lean unless debug payloads are requested
            if self.config.debug and e.source_path:
                sample["source_path"] = e.source_path
            if e.is_manifest_joined:
                for field in ("semantic_group", "phase_type", "level", "source_take", "segment_id"):
                    val = getattr(e, field)
                    if val:
                        sample[field] = val
                if e.folder_group and e.folder_group != e.dataset_group:
                    sample["folder_group"] = e.folder_group
            if e.window_start is not None:
                sample["window_start"] = e.window_start
                if e.window_interval is not None:
                    sample["window_interval"] = e.window_interval
            samples.append(sample)

        record = {
            "run_id": self.run_id,
            "step": step,
            "step_type": "train",
            "epoch": self.current_epoch,
            "optimizer_step": optimizer_stepped,
            "wall_time": time.time(),
            "lr": self.last_lr,
            "grad_norm": round(self.last_grad_norm, 6),
            "step_time_ms": round(step_time_ms, 2),
            "loss_mean": round(loss_mean, 8),
            "ema_50": round(self._ema_50.value, 8),
            "ema_200": round(self._ema_200.value, 8),
            "ema_1000": round(self._ema_1000.value, 8),
            # samples_n counts normal-loss samples only, so it keeps its meaning
            # for pre-preservation tooling; preservation samples are counted apart
            "samples_n": sum(1 for e in events if not e.is_preservation),
            "preservation_samples_n": sum(1 for e in events if e.is_preservation),
            "groups_n": len(groups),
            "samples": samples,
        }

        if not self._jsonl_wrote_provenance:
            # first record of the run echoes manifest hashes and the run's
            # expert-boundary / timestep knobs for file-level ledger comparisons
            record["provenance"] = self._provenance_dict()
            self._jsonl_wrote_provenance = True

        line = json.dumps(record, ensure_ascii=False)
        self._jsonl_file.write(line + "\n")
        self._jsonl_lines_since_flush += 1

        if self._jsonl_lines_since_flush >= 50:
            self._jsonl_file.flush()
            self._jsonl_lines_since_flush = 0

    # -----------------------------------------------------------------------
    # Startup banner
    # -----------------------------------------------------------------------

    def _print_banner(self) -> None:
        edges = self.config.noise_bucket_edges
        if edges is None or len(edges) != 2 or edges[0] >= edges[1]:
            edge_str = "auto (thirds)"
        else:
            edge_str = f"{edges[0]}/{edges[1]}"

        print(f"\n[LossTracker] Structured loss logging enabled")
        print(f"  Run ID: {self.run_id}")
        if self.config.write_jsonl:
            print(f"  JSONL: {self._jsonl_path}")
        print(
            f"  Worst videos table: every {self.config.worst_every} steps "
            f"(min_count={self.config.worst_min_count}, top 20)"
        )
        print(f"  Noise buckets: low/mid/high (edges: {edge_str})")
        print(f"  Max tracked videos: {self.config.max_tracked_videos}")
        print(f"  Outlier sigma: {self.config.loss_outlier_sigma}")
        print(f"  Worst per group: {self.config.worst_per_group_n}")
        print(f"  Caption max len: {self.config.caption_max_len}")
        print(f"  Debug payloads: {'on' if self.config.debug else 'off'}")
        if self.manifest is not None and getattr(self.manifest, "loaded", False):
            print(
                f"  Manifest: {self.manifest.total_files} rows from "
                f"{len(self.manifest.provenance())} manifest(s); "
                f"{len(self.manifest.design_by_group)} semantic groups"
            )
        print()

    # -----------------------------------------------------------------------
    # Snapshot for web UI
    # -----------------------------------------------------------------------

    def _worst_videos_snapshot(self, is_reg: Optional[bool]) -> list:
        """Format worst videos for JSON snapshot (extended format)."""
        return self.get_worst_videos_extended(top_n=20, is_reg=is_reg)

    def _summary_snapshot(self, is_reg: Optional[bool]) -> list:
        """Format summary rows for JSON snapshot.

        Args:
            is_reg: False = concept, True = reg, None = combined (all).
        """
        # Compute global EMA for relative difficulty
        global_ema = self._ema_200.value if self._ema_200.value > 0 else 1.0

        rows = []
        for (g_key, reg), g_ema in sorted(self._group_emas_by_reg.items()):
            if is_reg is not None and reg != is_reg:
                continue
            orig_name = self._sanitized_reverse.get(g_key, g_key)
            sample_count = self._group_sample_counts_by_reg.get((g_key, reg), 0)
            row: Dict[str, Any] = {
                "type": "group",
                "name": orig_name,
                "ema_200": round(g_ema.value, 6),
                "sample_count": sample_count,
            }
            raw_ema = self._group_raw_emas_by_reg.get((g_key, reg))
            if raw_ema is not None and raw_ema._initialized:
                row["ema_200_raw"] = round(raw_ema.value, 6)
            multiplier = self._group_multiplier_by_reg.get((g_key, reg))
            if multiplier is not None:
                row["loss_multiplier"] = multiplier
            # Enrich with cached group stats if available
            g_stats = getattr(self, "_cached_group_stats", {}).get((g_key, reg))
            if g_stats:
                row["std_loss"] = round(g_stats["std"], 6)
                row["min_loss"] = round(g_stats["min"], 6)
                row["max_loss"] = round(g_stats["max"], 6)
                row["clip_count"] = g_stats["clip_count"]
                row["relative_difficulty"] = round(g_ema.value / global_ema, 4)
            rows.append(row)
        for (b_key, reg), b_ema in sorted(self._bucket_emas_by_reg.items()):
            if is_reg is not None and reg != is_reg:
                continue
            rows.append({
                "type": "noise",
                "name": b_key,
                "ema_200": round(b_ema.value, 6),
                "sample_count": self._bucket_sample_counts_by_reg.get((b_key, reg), 0),
            })
        for (bidx, reg), b_ema in sorted(self._boundary_emas_by_reg.items()):
            if is_reg is not None and reg != is_reg:
                continue
            rows.append({
                "type": "boundary",
                "name": str(bidx),
                "ema_200": round(b_ema.value, 6),
                "sample_count": self._boundary_sample_counts_by_reg.get((bidx, reg), 0),
            })
        for row_type, emas, counts in (
            ("phase", self._phase_emas_by_reg, self._phase_sample_counts_by_reg),
            ("level", self._level_emas_by_reg, self._level_sample_counts_by_reg),
        ):
            for (a_key, reg), ema in sorted(emas.items()):
                if is_reg is not None and reg != is_reg:
                    continue
                rows.append({
                    "type": row_type,
                    "name": self._sanitized_reverse.get(a_key, a_key),
                    "ema_200": round(ema.value, 6),
                    "sample_count": counts.get((a_key, reg), 0),
                })
        return rows

    def _matrix_snapshot(self, is_reg: Optional[bool]) -> list:
        """Format matrix rows for JSON snapshot."""
        rows = []
        for (g_key, b_key, reg), ema in sorted(self._matrix_emas_by_reg.items()):
            if is_reg is not None and reg != is_reg:
                continue
            orig_name = self._sanitized_reverse.get(g_key, g_key)
            rows.append({
                "group": orig_name,
                "noise_bucket": b_key,
                "mean_loss_final": round(ema.value, 6),
            })
        return rows

    def _preservation_summary_snapshot(self, is_reg: Optional[bool]) -> list:
        """Preservation-loss summary rows (group / noise / boundary EMAs)."""
        rows = []
        for (g_key, reg), ema in sorted(self._pres_group_emas_by_reg.items()):
            if is_reg is not None and reg != is_reg:
                continue
            rows.append({
                "type": "group",
                "name": self._sanitized_reverse.get(g_key, g_key),
                "ema_200": round(ema.value, 6),
                "sample_count": self._pres_group_sample_counts_by_reg.get((g_key, reg), 0),
            })
        for (b_key, reg), ema in sorted(self._pres_bucket_emas_by_reg.items()):
            if is_reg is not None and reg != is_reg:
                continue
            rows.append({
                "type": "noise",
                "name": b_key,
                "ema_200": round(ema.value, 6),
                "sample_count": self._pres_bucket_sample_counts_by_reg.get((b_key, reg), 0),
            })
        for (bidx, reg), ema in sorted(self._pres_boundary_emas_by_reg.items()):
            if is_reg is not None and reg != is_reg:
                continue
            rows.append({
                "type": "boundary",
                "name": str(bidx),
                "ema_200": round(ema.value, 6),
                "sample_count": self._pres_boundary_sample_counts_by_reg.get((bidx, reg), 0),
            })
        return rows

    def _preservation_matrix_snapshot(self, is_reg: Optional[bool]) -> list:
        """Preservation group × noise bucket rows for JSON snapshot."""
        rows = []
        for (g_key, b_key, reg), ema in sorted(self._pres_matrix_emas_by_reg.items()):
            if is_reg is not None and reg != is_reg:
                continue
            rows.append({
                "group": self._sanitized_reverse.get(g_key, g_key),
                "noise_bucket": b_key,
                "mean_loss_final": round(ema.value, 6),
            })
        return rows

    def _ema_snapshot(self, is_reg: bool) -> dict:
        """Format EMA values for JSON snapshot."""
        return {
            "ema_50": round(self._reg_ema_50[is_reg].value, 6),
            "ema_200": round(self._reg_ema_200[is_reg].value, 6),
            "ema_1000": round(self._reg_ema_1000[is_reg].value, 6),
        }

    def _group_stats_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Format group stats for JSON snapshot (external analysis scripts)."""
        result = {}
        for (g_key, is_reg), stats in getattr(self, "_cached_group_stats", {}).items():
            if is_reg:
                continue  # concept-only for the top-level group_stats
            orig_name = self._sanitized_reverse.get(g_key, g_key)
            result[orig_name] = {
                "mean": round(stats["mean"], 6),
                "std": round(stats["std"], 6),
                "min": round(stats["min"], 6),
                "max": round(stats["max"], 6),
                "clip_count": stats["clip_count"],
            }
        return result

    def _write_snapshot(self, step: int) -> None:
        """Write a JSON snapshot of current tables for the web UI."""
        # Record trends and cache group stats before building snapshot
        self._record_all_trends(step)
        self._cache_group_stats()

        samples_total = self._samples_total_by_reg[False] + self._samples_total_by_reg[True]

        snapshot: Dict[str, Any] = {
            "step": step,
            "wall_time": time.time(),
            "run_id": self.run_id,

            # Combined (backwards compat)
            "worst_videos": self._worst_videos_snapshot(is_reg=None),
            "summary": self._summary_snapshot(is_reg=None),
            "matrix": self._matrix_snapshot(is_reg=None),
            "ema": {
                "ema_50": round(self._ema_50.value, 6),
                "ema_200": round(self._ema_200.value, 6),
                "ema_1000": round(self._ema_1000.value, 6),
            },

            # Split fields (always present, empty array/0.0 if no data)
            "worst_videos_concept": self._worst_videos_snapshot(is_reg=False),
            "worst_videos_reg": self._worst_videos_snapshot(is_reg=True),
            "summary_concept": self._summary_snapshot(is_reg=False),
            "summary_reg": self._summary_snapshot(is_reg=True),
            "matrix_concept": self._matrix_snapshot(is_reg=False),
            "matrix_reg": self._matrix_snapshot(is_reg=True),
            "ema_concept": self._ema_snapshot(is_reg=False),
            "ema_reg": self._ema_snapshot(is_reg=True),
            "has_reg_data": self._has_reg_data,

            # Per-group worst clips
            "worst_by_group_concept": self.get_worst_by_group(is_reg=False),
            "worst_by_group_reg": self.get_worst_by_group(is_reg=True) if self._has_reg_data else {},

            # Preservation-loss (DOP) breakdowns
            "has_preservation_data": self._has_preservation_data,
            "preservation_summary_concept": self._preservation_summary_snapshot(is_reg=False),
            "preservation_summary_reg": self._preservation_summary_snapshot(is_reg=True),
            "preservation_matrix_concept": self._preservation_matrix_snapshot(is_reg=False),
            "preservation_matrix_reg": self._preservation_matrix_snapshot(is_reg=True),

            # Group-level stats for external analysis
            "group_stats": self._group_stats_snapshot(),

            # Manifest-driven diagnostics (P1): exposure vs design, draw
            # interleaving, and run provenance
            "exposure": self._exposure_snapshot(),
            "interleaving": self._interleaving_snapshot(),
            "provenance": self._provenance_dict(),

            # Sample totals
            "samples_total": samples_total,
            "samples_concept_total": self._samples_total_by_reg[False],
            "samples_reg_total": self._samples_total_by_reg[True],
        }

        # Atomic write: write to tmp, then rename
        snapshot_path = os.path.join(
            os.path.dirname(self._jsonl_path) if self._jsonl_path else ".",
            "loss_analysis.json",
        )
        tmp_path = snapshot_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False)
        os.replace(tmp_path, snapshot_path)

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def close(self) -> None:
        if self._jsonl_file is not None:
            self._jsonl_file.flush()
            self._jsonl_file.close()
            self._jsonl_file = None
