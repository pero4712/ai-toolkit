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
    name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9_\-]", "", name)
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
        )


# ---------------------------------------------------------------------------
# Per-sample event
# ---------------------------------------------------------------------------

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
        "losses",
        "idx",
        "count",
        "sum",
        "total_count",
        "last_seen_step",
    )

    def __init__(
        self,
        source_id: str,
        dataset_group: str,
        window: int = 50,
    ) -> None:
        self.source_id = source_id
        self.dataset_group = dataset_group  # original unsanitized
        self.losses: List[float] = [0.0] * window
        self.idx = 0
        self.count = 0
        self.sum = 0.0
        self.total_count = 0
        self.last_seen_step = 0

    def add(self, loss: float, step: int) -> None:
        if self.count >= len(self.losses):
            self.sum -= self.losses[self.idx]
        self.losses[self.idx] = loss
        self.sum += loss
        self.idx = (self.idx + 1) % len(self.losses)
        self.count = min(self.count + 1, len(self.losses))
        self.total_count += 1
        self.last_seen_step = step

    @property
    def mean(self) -> float:
        return self.sum / max(self.count, 1)

    @property
    def p90(self) -> float:
        if self.count == 0:
            return 0.0
        vals = sorted(self.losses[: self.count])
        k = max(1, math.ceil(0.9 * self.count))
        return vals[k - 1]


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

        # Per-group EMAs  (sanitized key → EMA)
        self._group_emas: Dict[str, EMAScalar] = {}
        # Per-noise-bucket EMAs
        self._bucket_emas: Dict[str, EMAScalar] = {}
        # Per-boundary EMAs (WAN 14B multistage)
        self._boundary_emas: Dict[int, EMAScalar] = {}

        # Per-video tracking (bounded)
        self._video_stats: Dict[str, VideoStats] = {}

        # Collision-safe name registry: original → sanitized key
        self._sanitized_names: Dict[str, str] = {}
        # Reverse lookup: sanitized key → original (first registrant)
        self._sanitized_reverse: Dict[str, str] = {}

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
    ) -> Dict[str, float]:
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
        events = self._current_step_events
        self._current_step_events = []

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

        # Per-group breakdown
        group_losses: Dict[str, List[float]] = defaultdict(list)
        for e in events:
            group_losses[e.dataset_group].append(e.loss_final)

        for group, losses in group_losses.items():
            g_key = self._get_sanitized_group(group)
            g_mean = sum(losses) / len(losses)
            if g_key not in self._group_emas:
                self._group_emas[g_key] = EMAScalar(span=200)
            self._group_emas[g_key].update(g_mean)
            metrics[f"loss_by_group_ema/{g_key}"] = self._group_emas[g_key].value
            metrics[f"samples_by_group/{g_key}"] = float(len(losses))

        # Per-noise-bucket breakdown
        bucket_losses: Dict[str, List[float]] = defaultdict(list)
        for e in events:
            bucket_losses[e.timestep_bucket].append(e.loss_final)

        for bucket, losses in bucket_losses.items():
            b_mean = sum(losses) / len(losses)
            if bucket not in self._bucket_emas:
                self._bucket_emas[bucket] = EMAScalar(span=200)
            self._bucket_emas[bucket].update(b_mean)
            metrics[f"loss_by_noise_ema/{bucket}"] = self._bucket_emas[bucket].value
            metrics[f"samples_by_noise/{bucket}"] = float(len(losses))

        # Per-boundary breakdown (multistage only)
        boundary_losses: Dict[int, List[float]] = defaultdict(list)
        for e in events:
            if e.boundary_index is not None:
                boundary_losses[e.boundary_index].append(e.loss_final)

        for bidx, losses in boundary_losses.items():
            b_mean = sum(losses) / len(losses)
            if bidx not in self._boundary_emas:
                self._boundary_emas[bidx] = EMAScalar(span=200)
            self._boundary_emas[bidx].update(b_mean)
            metrics[f"loss_by_boundary/{bidx}"] = self._boundary_emas[bidx].value

        # Per-video tracking
        for e in events:
            self._update_video_stats(e, step)

        # JSONL
        if self.config.write_jsonl:
            self._write_jsonl(events, step, mean_loss_final, step_time_ms, optimizer_stepped)

        return metrics

    # -----------------------------------------------------------------------
    # Video stats
    # -----------------------------------------------------------------------

    def _update_video_stats(self, event: LossEvent, step: int) -> None:
        sid = event.source_id
        if sid not in self._video_stats:
            if len(self._video_stats) >= self.config.max_tracked_videos:
                # Evict: lowest total_count, tiebreaker oldest last_seen_step
                evict_key = min(
                    self._video_stats,
                    key=lambda k: (
                        self._video_stats[k].total_count,
                        self._video_stats[k].last_seen_step,
                    ),
                )
                del self._video_stats[evict_key]
            self._video_stats[sid] = VideoStats(
                sid, event.dataset_group, window=50
            )
        self._video_stats[sid].add(event.loss_final, step)

    def get_worst_videos(
        self, top_n: int = 20
    ) -> List[Tuple[str, str, float, float, float, int]]:
        """Return worst videos by rolling mean loss_final.

        Returns list of (source_id, dataset_group, mean_final, mean_raw,
        p90_final, total_count).  Filtered to total_count >= worst_min_count.
        """
        eligible = [
            v
            for v in self._video_stats.values()
            if v.total_count >= self.config.worst_min_count
        ]
        eligible.sort(key=lambda v: v.mean, reverse=True)
        result = []
        for v in eligible[:top_n]:
            result.append((
                v.source_id,
                v.dataset_group,
                v.mean,
                v.mean,  # raw mean not separately tracked per-video; same as final
                v.p90,
                v.total_count,
            ))
        return result

    # -----------------------------------------------------------------------
    # Periodic tables
    # -----------------------------------------------------------------------

    def log_worst_videos_table(self, step: int, logger: Any) -> None:
        worst = self.get_worst_videos(top_n=20)
        if not worst:
            return

        # Console output
        print(f"\n[LossTracker] Worst videos at step {step}:")
        print(f"  {'source_id':<10} {'group':<20} {'mean':>8} {'p90':>8} {'count':>6}")
        for sid, grp, mean_f, _mean_r, p90, cnt in worst:
            print(f"  {sid:<10} {grp:<20} {mean_f:>8.5f} {p90:>8.5f} {cnt:>6}")
        print()

        # W&B table
        if hasattr(logger, "_log"):
            try:
                import wandb

                table = wandb.Table(
                    columns=[
                        "source_id",
                        "dataset_group",
                        "mean_loss_final",
                        "p90_loss_final",
                        "count",
                    ],
                    data=[
                        [sid, grp, mean_f, p90, cnt]
                        for sid, grp, mean_f, _mean_r, p90, cnt in worst
                    ],
                )
                logger._log({"worst_videos": table}, commit=False)
            except ImportError:
                pass

    def log_matrix_table(self, step: int, logger: Any) -> None:
        """Log group × noise bucket cross-tabulation as a W&B Table."""
        if not hasattr(logger, "_log"):
            return

        # Build matrix from current group/bucket EMA values
        rows = []
        for g_key, g_ema in self._group_emas.items():
            for b_key, b_ema in self._bucket_emas.items():
                rows.append([g_key, b_key, g_ema.value, b_ema.value])

        if not rows:
            return

        try:
            import wandb

            table = wandb.Table(
                columns=["group", "noise_bucket", "group_ema", "bucket_ema"],
                data=rows,
            )
            logger._log({"loss_matrix": table}, commit=False)
        except ImportError:
            pass

    def log_summary_table(self, step: int, logger: Any) -> None:
        """At-a-glance health report: per-group and per-bucket stats."""
        rows = []

        # Group rows
        for g_key, g_ema in sorted(self._group_emas.items()):
            rows.append(["group", g_key, g_ema.value])

        # Bucket rows
        for b_key, b_ema in sorted(self._bucket_emas.items()):
            rows.append(["noise", b_key, b_ema.value])

        # Boundary rows
        for bidx, b_ema in sorted(self._boundary_emas.items()):
            rows.append(["boundary", str(bidx), b_ema.value])

        if not rows:
            return

        # Console
        print(f"\n[LossTracker] Training summary at step {step}:")
        print(f"  {'type':<10} {'name':<20} {'ema_200':>10}")
        for typ, name, val in rows:
            print(f"  {typ:<10} {name:<20} {val:>10.6f}")
        print()

        # W&B table
        if hasattr(logger, "_log"):
            try:
                import wandb

                table = wandb.Table(
                    columns=["type", "name", "ema_200"],
                    data=rows,
                )
                logger._log({"training_summary": table}, commit=False)
            except ImportError:
                pass

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
            if e.caption is not None:
                sample["caption"] = e.caption
            if e.source_path:
                sample["source_path"] = e.source_path
            samples.append(sample)

        record = {
            "run_id": self.run_id,
            "step": step,
            "step_type": "train",
            "optimizer_step": optimizer_stepped,
            "wall_time": time.time(),
            "lr": self.last_lr,
            "grad_norm": round(self.last_grad_norm, 6),
            "step_time_ms": round(step_time_ms, 2),
            "loss_mean": round(loss_mean, 8),
            "ema_50": round(self._ema_50.value, 8),
            "ema_200": round(self._ema_200.value, 8),
            "ema_1000": round(self._ema_1000.value, 8),
            "samples_n": len(events),
            "groups_n": len(groups),
            "samples": samples,
        }

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
        print(f"  Debug payloads: {'on' if self.config.debug else 'off'}")
        print()

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def close(self) -> None:
        if self._jsonl_file is not None:
            self._jsonl_file.flush()
            self._jsonl_file.close()
            self._jsonl_file = None
