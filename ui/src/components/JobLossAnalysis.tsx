'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { Job } from '@prisma/client';
import { Play, X } from 'lucide-react';
import useJobLossAnalysis, { MatrixCell, SummaryRow, WorstVideo } from '@/hooks/useJobLossAnalysis';
import UniversalTable from './UniversalTable';
import { encodeFilePathForUrl, exportFileName } from '@/utils/basic';

const VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.wmv', '.m4v', '.flv'];

function isVideoPath(p: string) {
  const lower = p.toLowerCase();
  return VIDEO_EXTS.some(ext => lower.endsWith(ext));
}

function formatNum(v: number) {
  if (!Number.isFinite(v)) return '';
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(3);
  if (Math.abs(v) >= 1) return v.toFixed(4);
  return v.toPrecision(4);
}

function timeAgo(wallTime: number): string {
  const seconds = Math.floor(Date.now() / 1000 - wallTime);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

const TREND_ARROWS: Record<string, string> = { up: '\u2191', down: '\u2193', flat: '\u2192', '~': '~' };
const TREND_COLORS: Record<string, string> = {
  up: 'text-red-400',
  down: 'text-green-400',
  flat: 'text-gray-400',
  '~': 'text-gray-600',
};

function ZScoreCell({ row }: { row: WorstVideo }) {
  const z = row.z_score;
  if (z == null) return <span className="text-gray-600">-</span>;
  let cls = 'font-mono';
  if (z > 3) cls += ' text-red-400 font-bold';
  else if (z > 2) cls += ' text-orange-400 font-bold';
  const suffix = row.is_new_outlier ? '!!' : row.is_outlier ? '!' : '';
  return <span className={cls}>{z.toFixed(1)}{suffix}</span>;
}

function TrendCell({ row }: { row: WorstVideo }) {
  const t = row.trend ?? '~';
  return <span className={TREND_COLORS[t] ?? 'text-gray-600'}>{TREND_ARROWS[t] ?? '~'}</span>;
}

// --- cell helpers for the manifest diagnostics tables ---
const pctCell = (key: string) => (row: any) =>
  row[key] != null ? `${(row[key] * 100).toFixed(1)}%` : '—';
const numCell = (key: string) => (row: any) => (row[key] != null ? formatNum(row[key]) : '—');
const ratioCell = (row: any) => {
  if (row.ratio == null) return '—';
  const off = Math.abs(row.ratio - 1) > 0.5;
  return <span className={off ? 'text-amber-400' : ''}>{row.ratio.toFixed(2)}×</span>;
};
const flagCell = (row: any) =>
  row.flagged ? <span className="text-red-400 font-medium">flag</span> : null;
function FlagTag() {
  return <span className="ml-2 text-red-400 font-medium">flagged</span>;
}

/** Pivot flat matrix cells into { group → { bucket → value } } */
function pivotMatrix(cells: MatrixCell[]) {
  const groups = new Set<string>();
  const buckets = new Set<string>();
  const map: Record<string, Record<string, number>> = {};

  for (const c of cells) {
    groups.add(c.group);
    buckets.add(c.noise_bucket);
    if (!map[c.group]) map[c.group] = {};
    map[c.group][c.noise_bucket] = c.mean_loss_final;
  }

  // Sort buckets in canonical order
  const bucketOrder = ['low', 'mid', 'high'];
  const sortedBuckets = [...buckets].sort(
    (a, b) => (bucketOrder.indexOf(a) ?? 99) - (bucketOrder.indexOf(b) ?? 99),
  );

  return { groups: [...groups].sort(), buckets: sortedBuckets, map };
}

/** Shared columns for worst-video tables */
function worstVideoColumns(compact = false, onPlay?: (row: WorstVideo) => void) {
  const cols: any[] = [
    {
      title: '',
      key: 'play',
      render: (row: WorstVideo) =>
        row.source_path && onPlay ? (
          <button
            type="button"
            onClick={() => onPlay(row)}
            title={`Play ${row.source_path}`}
            className="text-purple-400 hover:text-purple-200 transition-colors"
          >
            <Play className="w-3.5 h-3.5" />
          </button>
        ) : null,
      className: 'w-6',
    },
    { title: 'Source ID', key: 'source_id', className: 'font-mono' },
  ];
  if (!compact) {
    cols.push({ title: 'Group', key: 'dataset_group' });
  }
  cols.push(
    {
      title: 'Mean Loss',
      key: 'mean_loss_final',
      render: (row: WorstVideo) => formatNum(row.mean_loss_final),
      className: 'text-right font-mono',
    },
    {
      // pre-multiplier loss: the comparable number across groups with
      // different loss_multiplier settings
      title: 'Raw',
      key: 'mean_loss_raw',
      render: (row: WorstVideo) =>
        row.mean_loss_raw != null ? formatNum(row.mean_loss_raw) : '',
      className: 'text-right font-mono text-gray-400',
    },
    {
      title: 'Z',
      key: 'z_score',
      render: (row: WorstVideo) => <ZScoreCell row={row} />,
      className: 'text-right',
    },
    {
      title: 'Ratio',
      key: 'loss_ratio',
      render: (row: WorstVideo) =>
        row.loss_ratio != null ? `${row.loss_ratio.toFixed(2)}x` : '',
      className: 'text-right font-mono',
    },
    {
      title: 'Trend',
      key: 'trend',
      render: (row: WorstVideo) => <TrendCell row={row} />,
      className: 'text-center',
    },
  );
  if (!compact) {
    cols.push(
      {
        title: 'P90',
        key: 'p90_loss_final',
        render: (row: WorstVideo) => formatNum(row.p90_loss_final),
        className: 'text-right font-mono',
      },
      {
        title: 'Count',
        key: 'count',
        render: (row: WorstVideo) => row.count.toLocaleString(),
        className: 'text-right',
      },
    );
  }
  cols.push({
    title: 'Caption',
    key: 'caption',
    render: (row: WorstVideo) => (
      <span
        className="text-gray-500 text-xs truncate max-w-[200px] inline-block align-bottom"
        title={row.caption ?? ''}
      >
        {row.caption ?? ''}
      </span>
    ),
  });
  return cols;
}

export default function JobLossAnalysis({ job }: { job: Job }) {
  const { data, status, refresh } = useJobLossAnalysis(job.id, 10000);
  const [viewReg, setViewReg] = useState(false);
  const [playerClip, setPlayerClip] = useState<WorstVideo | null>(null);

  // Effective view: force concept when no reg data exists
  const showReg = viewReg && !!data.has_reg_data;

  // Pick data based on selected view, falling back to combined fields
  const activeSummary = useMemo(() => {
    if (showReg) return data.summary_reg ?? data.summary ?? [];
    return data.summary_concept ?? data.summary ?? [];
  }, [data, showReg]);

  const activeMatrix = useMemo(() => {
    if (showReg) return data.matrix_reg ?? data.matrix ?? [];
    return data.matrix_concept ?? data.matrix ?? [];
  }, [data, showReg]);

  const activeWorstVideos = useMemo(() => {
    if (showReg) return data.worst_videos_reg ?? data.worst_videos ?? [];
    return data.worst_videos_concept ?? data.worst_videos ?? [];
  }, [data, showReg]);

  const activeEma = useMemo(() => {
    if (showReg) return data.ema_reg ?? data.ema;
    return data.ema_concept ?? data.ema;
  }, [data, showReg]);

  const activeWorstByGroup = useMemo(() => {
    if (showReg) return data.worst_by_group_reg ?? {};
    return data.worst_by_group_concept ?? {};
  }, [data, showReg]);

  const groupSummary = useMemo(
    () => activeSummary.filter(r => r.type === 'group'),
    [activeSummary],
  );
  const noiseSummary = useMemo(
    () => activeSummary.filter(r => r.type === 'noise' || r.type === 'boundary'),
    [activeSummary],
  );
  const cohortSummary = useMemo(
    () => activeSummary.filter(r => r.type === 'phase' || r.type === 'level'),
    [activeSummary],
  );

  const matrix = useMemo(() => pivotMatrix(activeMatrix), [activeMatrix]);

  // Preservation (DOP) data — present when the tracker records preservation events
  const activePresSummary = useMemo(() => {
    if (showReg) return data.preservation_summary_reg ?? [];
    return data.preservation_summary_concept ?? [];
  }, [data, showReg]);

  const activePresMatrix = useMemo(() => {
    if (showReg) return data.preservation_matrix_reg ?? [];
    return data.preservation_matrix_concept ?? [];
  }, [data, showReg]);

  const presGroupRows = useMemo(
    () =>
      activePresSummary
        .filter(r => r.type === 'group')
        .map(r => {
          // Preservation loss is never scaled by the group's loss_multiplier,
          // but the normal-loss EMA is — dividing by the post-multiplier value
          // would overstate drift-per-unit-learning on damped groups (2x at
          // x0.5). Prefer the raw EMA as denominator; fall back to un-scaling
          // via the multiplier, then to the naive ratio for older snapshots.
          const normal = activeSummary.find(g => g.type === 'group' && g.name === r.name);
          let denom: number | null = null;
          let denomIsRaw = true;
          if (normal) {
            if (normal.ema_200_raw != null) {
              denom = normal.ema_200_raw;
            } else if (normal.loss_multiplier != null && normal.loss_multiplier > 0) {
              denom = normal.ema_200 / normal.loss_multiplier;
            } else {
              denom = normal.ema_200;
              denomIsRaw = false;
            }
          }
          const dop_share = denom != null && denom > 0 ? r.ema_200 / denom : null;
          return { ...r, dop_share, dop_share_is_raw: denomIsRaw };
        }),
    [activePresSummary, activeSummary],
  );

  const presNoiseRows = useMemo(
    () => activePresSummary.filter(r => r.type === 'noise' || r.type === 'boundary' || r.type === 'phase'),
    [activePresSummary],
  );

  const presMatrix = useMemo(() => pivotMatrix(activePresMatrix), [activePresMatrix]);
  const presValues = useMemo(() => activePresMatrix.map(c => c.mean_loss_final), [activePresMatrix]);
  const presMin = presValues.length ? Math.min(...presValues) : 0;
  const presMax = presValues.length ? Math.max(...presValues) : 1;
  const presRange = presMax - presMin || 1;

  // Empirical reg share of samples so far
  const regShare = useMemo(() => {
    const c = data.samples_concept_total ?? 0;
    const r = data.samples_reg_total ?? 0;
    return c + r > 0 ? r / (c + r) : null;
  }, [data]);

  // Heatmap range
  const allValues = useMemo(() => activeMatrix.map(c => c.mean_loss_final), [activeMatrix]);
  const minVal = allValues.length ? Math.min(...allValues) : 0;
  const maxVal = allValues.length ? Math.max(...allValues) : 1;
  const range = maxVal - minVal || 1;

  // Export
  const [exportOpen, setExportOpen] = useState(false);
  const exportRef = useRef<HTMLDivElement>(null);

  const buildExportJson = useCallback(() => {
    // boundaries as the trainer sees them: [1] + expert_boundaries (fractions of
    // the 0..1000 timestep scale); expert i covers (b[i]*1000 .. b[i+1]*1000]
    const expertRange = (i: number): [number, number] | null => {
      const eb = data.provenance?.expert_boundaries;
      if (!Array.isArray(eb) || !Number.isFinite(i) || i < 0 || i >= eb.length) return null;
      const b = [1, ...eb.map(Number)];
      return [Math.round(b[i] * 1000), Math.round(b[i + 1] * 1000)];
    };

    const buildSection = (
      summary: SummaryRow[],
      matrix: MatrixCell[],
      worstVideos: WorstVideo[],
      worstByGroup: Record<string, WorstVideo[]>,
      ema: { ema_50: number; ema_200: number; ema_1000: number } | undefined,
      samplesTotal: number | null,
    ) => {
      const section: Record<string, any> = {};
      if (samplesTotal != null) section.samples_total = samplesTotal;
      if (ema) section.ema = ema;

      const groups = summary.filter(r => r.type === 'group');
      const noises = summary.filter(r => r.type === 'noise');
      const experts = summary.filter(r => r.type === 'boundary');
      const cohorts = summary.filter(r => r.type === 'phase' || r.type === 'level');

      if (groups.length > 0) {
        section.group_summary = groups.map(r => ({
          name: r.name,
          ema_200: r.ema_200,
          ema_200_raw: r.ema_200_raw ?? null,
          loss_multiplier: r.loss_multiplier ?? null,
          std_loss: r.std_loss ?? null,
          relative_difficulty: r.relative_difficulty ?? null,
          clip_count: r.clip_count ?? null,
          sample_count: r.sample_count,
        }));
      }

      if (noises.length > 0) {
        section.noise_summary = noises.map(r => ({
          type: r.type,
          name: r.name,
          ema_200: r.ema_200,
          sample_count: r.sample_count,
        }));
      }
      // expert split (multistage boundary index); previously mixed into
      // noise_summary as type: "boundary" rows
      if (experts.length > 0) {
        section.expert_summary = experts.map(r => ({
          expert_index: Number(r.name),
          timestep_range: expertRange(Number(r.name)),
          ema_200: r.ema_200,
          sample_count: r.sample_count,
          dropped_ema_200: r.dropped_ema_200 ?? null,
        }));
      }
      // manifest cohort axes (phase_type / level)
      if (cohorts.length > 0) {
        section.cohort_summary = cohorts.map(r => ({
          axis: r.type,
          name: r.name,
          ema_200: r.ema_200,
          sample_count: r.sample_count,
          dropped_ema_200: r.dropped_ema_200 ?? null,
          dropped_samples: r.dropped_samples ?? null,
        }));
      }

      if (matrix.length > 0) {
        section.group_noise_matrix = {};
        for (const c of matrix) {
          if (!section.group_noise_matrix[c.group]) section.group_noise_matrix[c.group] = {};
          section.group_noise_matrix[c.group][c.noise_bucket] = c.mean_loss_final;
        }
      }

      if (worstVideos.length > 0) {
        section.worst_videos = worstVideos.map(v => ({
          source_id: v.source_id,
          group: v.dataset_group,
          mean_loss: v.mean_loss_final,
          mean_loss_raw: v.mean_loss_raw ?? null,
          z_score: v.z_score ?? null,
          loss_ratio: v.loss_ratio ?? null,
          trend: v.trend ?? null,
          count: v.count,
          caption: v.caption ?? null,
        }));
      }

      if (Object.keys(worstByGroup).length > 0) {
        section.worst_by_group = {};
        for (const [group, clips] of Object.entries(worstByGroup)) {
          section.worst_by_group[group] = clips.map(v => ({
            source_id: v.source_id,
            mean_loss: v.mean_loss_final,
            z_score: v.z_score ?? null,
            loss_ratio: v.loss_ratio ?? null,
            trend: v.trend ?? null,
            count: v.count,
            caption: v.caption ?? null,
          }));
        }
      }

      return section;
    };

    const buildPresSection = (summary: SummaryRow[], matrix: MatrixCell[]) => {
      const section: Record<string, any> = {};
      const groups = summary.filter(r => r.type === 'group');
      const phases = summary.filter(r => r.type === 'phase');
      const noises = summary.filter(r => r.type === 'noise');
      const experts = summary.filter(r => r.type === 'boundary');
      const rowOf = (r: SummaryRow) => ({ name: r.name, ema_200: r.ema_200, sample_count: r.sample_count });
      if (groups.length > 0) section.by_group = groups.map(rowOf);
      if (phases.length > 0) section.by_phase = phases.map(rowOf);
      if (noises.length > 0) section.by_noise = noises.map(r => ({ type: r.type, ...rowOf(r) }));
      // expert split (multistage boundary index); previously mixed into
      // by_noise as type: "boundary" rows
      if (experts.length > 0) {
        section.by_expert = experts.map(r => ({
          expert_index: Number(r.name),
          timestep_range: expertRange(Number(r.name)),
          ema_200: r.ema_200,
          sample_count: r.sample_count,
        }));
      }
      if (matrix.length > 0) {
        section.group_noise_matrix = {};
        for (const c of matrix) {
          if (!section.group_noise_matrix[c.group]) section.group_noise_matrix[c.group] = {};
          section.group_noise_matrix[c.group][c.noise_bucket] = c.mean_loss_final;
        }
      }
      return Object.keys(section).length > 0 ? section : null;
    };

    const obj: Record<string, any> = {
      step: data.step ?? null,
      wall_time: data.wall_time ? new Date(data.wall_time * 1000).toISOString() : null,
    };

    if (data.has_reg_data) {
      obj.samples = {
        concept: data.samples_concept_total ?? null,
        reg: data.samples_reg_total ?? null,
      };

      obj.concept = buildSection(
        data.summary_concept ?? data.summary ?? [],
        data.matrix_concept ?? data.matrix ?? [],
        data.worst_videos_concept ?? data.worst_videos ?? [],
        data.worst_by_group_concept ?? {},
        data.ema_concept ?? data.ema,
        data.samples_concept_total ?? null,
      );

      obj.reg = buildSection(
        data.summary_reg ?? [],
        data.matrix_reg ?? [],
        data.worst_videos_reg ?? [],
        data.worst_by_group_reg ?? {},
        data.ema_reg,
        data.samples_reg_total ?? null,
      );

      if (data.has_preservation_data) {
        const presConcept = buildPresSection(
          data.preservation_summary_concept ?? [],
          data.preservation_matrix_concept ?? [],
        );
        const presReg = buildPresSection(
          data.preservation_summary_reg ?? [],
          data.preservation_matrix_reg ?? [],
        );
        if (presConcept) obj.concept.preservation = presConcept;
        if (presReg) obj.reg.preservation = presReg;
      }
    } else {
      // No reg data — emit a single concept section at the top level for compactness.
      Object.assign(
        obj,
        buildSection(
          data.summary_concept ?? data.summary ?? [],
          data.matrix_concept ?? data.matrix ?? [],
          data.worst_videos_concept ?? data.worst_videos ?? [],
          data.worst_by_group_concept ?? {},
          data.ema_concept ?? data.ema,
          data.samples_total ?? null,
        ),
      );
      if (data.has_preservation_data) {
        const pres = buildPresSection(
          data.preservation_summary_concept ?? [],
          data.preservation_matrix_concept ?? [],
        );
        if (pres) obj.preservation = pres;
      }
    }

    const diagnostics: Record<string, any> = {};
    for (const key of ['exposure', 'interleaving', 'window_coverage', 'dropout', 'duplicates', 'provenance'] as const) {
      if ((data as any)[key] != null) diagnostics[key] = (data as any)[key];
    }
    if (Object.keys(diagnostics).length > 0) obj.diagnostics = diagnostics;

    return JSON.stringify(obj, null, 2);
  }, [data]);

  const handleDownload = useCallback(() => {
    const json = buildExportJson();
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = exportFileName(job.name, 'loss-analysis', data.step);
    a.click();
    URL.revokeObjectURL(url);
    setExportOpen(false);
  }, [buildExportJson, data.step, job.name]);

  const handleCopy = useCallback(async () => {
    const json = buildExportJson();
    await navigator.clipboard.writeText(json);
    setExportOpen(false);
  }, [buildExportJson]);

  if (!data.available) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-gray-400 space-y-2">
        {data.enabled ? (
          <>
            <p className="text-sm">Structured loss logging is enabled.</p>
            <p className="text-xs">Waiting for the first periodic snapshot...</p>
          </>
        ) : (
          <>
            <p className="text-sm">Structured loss logging is not enabled.</p>
            <p className="text-xs">
              Add <code className="bg-gray-800 px-1 rounded">structured_loss: true</code> under
              the <code className="bg-gray-800 px-1 rounded">logging</code> section in your process config.
            </p>
          </>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-6 px-4 pb-8">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <h2 className="text-lg font-medium text-gray-100">Loss Analysis</h2>
        {data.step != null && (
          <span className="text-gray-400">
            Step {data.step.toLocaleString()}
          </span>
        )}
        {data.wall_time != null && (
          <span className="text-gray-500">
            &middot; Updated {timeAgo(data.wall_time)}
          </span>
        )}

        {/* Concept / Reg toggle — only when reg data exists */}
        {data.has_reg_data && (
          <div className="flex gap-1 ml-2">
            <button
              onClick={() => setViewReg(false)}
              className={`px-2 py-0.5 rounded text-xs font-medium transition-colors ${
                !viewReg
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-gray-200'
              }`}
            >
              Concept
            </button>
            <button
              onClick={() => setViewReg(true)}
              className={`px-2 py-0.5 rounded text-xs font-medium transition-colors ${
                viewReg
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-gray-200'
              }`}
            >
              Reg
            </button>
          </div>
        )}

        {/* Sample counts */}
        {data.has_reg_data && data.samples_concept_total != null && data.samples_reg_total != null && (
          <span className="text-xs text-gray-500">
            {data.samples_concept_total.toLocaleString()} concept / {data.samples_reg_total.toLocaleString()} reg samples
            {regShare != null && (
              <span className="text-gray-400"> ({(regShare * 100).toFixed(1)}% reg)</span>
            )}
          </span>
        )}

        <div className="flex-1" />
        {activeEma && (
          <div className="flex gap-3 text-xs">
            <span className="bg-gray-800 px-2 py-0.5 rounded text-blue-400">
              EMA-50: {formatNum(activeEma.ema_50)}
            </span>
            <span className="bg-gray-800 px-2 py-0.5 rounded text-emerald-400">
              EMA-200: {formatNum(activeEma.ema_200)}
            </span>
            <span className="bg-gray-800 px-2 py-0.5 rounded text-purple-400">
              EMA-1000: {formatNum(activeEma.ema_1000)}
            </span>
          </div>
        )}

        {/* Export dropdown */}
        <div className="relative" ref={exportRef}>
          <button
            onClick={() => setExportOpen(v => !v)}
            className="px-2 py-0.5 rounded text-xs font-medium bg-gray-800 text-gray-400 hover:text-gray-200 transition-colors"
          >
            Export JSON
          </button>
          {exportOpen && (
            <div className="absolute right-0 top-full mt-1 z-20 bg-gray-800 border border-gray-700 rounded shadow-lg overflow-hidden">
              <button
                onClick={handleDownload}
                className="block w-full text-left px-4 py-2 text-xs text-gray-300 hover:bg-gray-700 transition-colors whitespace-nowrap"
              >
                Download file
              </button>
              <button
                onClick={handleCopy}
                className="block w-full text-left px-4 py-2 text-xs text-gray-300 hover:bg-gray-700 transition-colors whitespace-nowrap"
              >
                Copy to clipboard
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Training Summary */}
      {(groupSummary.length > 0 || noiseSummary.length > 0) && (
        <div>
          <h3 className="text-sm font-medium text-gray-300 mb-2">Training Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {groupSummary.length > 0 && (
              <div>
                <p className="text-xs text-gray-500 mb-1">By Dataset Group</p>
                <UniversalTable
                  isLoading={status === 'loading'}
                  onRefresh={refresh}
                  columns={[
                    { title: 'Group', key: 'name' },
                    {
                      title: 'EMA-200',
                      key: 'ema_200',
                      render: (row: any) => formatNum(row.ema_200),
                      className: 'text-right font-mono',
                    },
                    {
                      title: 'Raw',
                      key: 'ema_200_raw',
                      render: (row: any) =>
                        row.ema_200_raw != null ? formatNum(row.ema_200_raw) : '',
                      className: 'text-right font-mono text-gray-400',
                    },
                    {
                      title: 'Dropped',
                      key: 'dropped_ema_200',
                      render: (row: any) =>
                        row.dropped_ema_200 != null ? (
                          <span title={`${row.dropped_samples ?? 0} dropped-caption draws`}>
                            {formatNum(row.dropped_ema_200)}
                          </span>
                        ) : (
                          ''
                        ),
                      className: 'text-right font-mono text-gray-400',
                    },
                    {
                      title: 'Mult',
                      key: 'loss_multiplier',
                      render: (row: any) =>
                        row.loss_multiplier != null && row.loss_multiplier !== 1
                          ? `×${row.loss_multiplier}`
                          : '',
                      className: 'text-right text-gray-500',
                    },
                    {
                      title: 'StdDev',
                      key: 'std_loss',
                      render: (row: any) => row.std_loss != null ? formatNum(row.std_loss) : '',
                      className: 'text-right font-mono',
                    },
                    {
                      title: 'Difficulty',
                      key: 'relative_difficulty',
                      render: (row: any) =>
                        row.relative_difficulty != null
                          ? `${Math.round(row.relative_difficulty * 100)}%`
                          : '',
                      className: 'text-right',
                    },
                    {
                      title: 'Clips',
                      key: 'clip_count',
                      render: (row: any) => row.clip_count ?? '',
                      className: 'text-right',
                    },
                    {
                      title: 'Samples',
                      key: 'sample_count',
                      render: (row: any) => row.sample_count.toLocaleString(),
                      className: 'text-right',
                    },
                  ]}
                  rows={groupSummary}
                />
              </div>
            )}
            {noiseSummary.length > 0 && (
              <div>
                <p className="text-xs text-gray-500 mb-1">By Noise Bucket</p>
                <UniversalTable
                  isLoading={status === 'loading'}
                  onRefresh={refresh}
                  columns={[
                    { title: 'Bucket', key: 'name' },
                    {
                      title: 'EMA-200',
                      key: 'ema_200',
                      render: (row: any) => formatNum(row.ema_200),
                      className: 'text-right font-mono',
                    },
                    {
                      title: 'Samples',
                      key: 'sample_count',
                      render: (row: any) => row.sample_count.toLocaleString(),
                      className: 'text-right',
                    },
                  ]}
                  rows={noiseSummary}
                />
              </div>
            )}
            {cohortSummary.length > 0 && (
              <div>
                <p className="text-xs text-gray-500 mb-1">By Phase / Level (manifest)</p>
                <UniversalTable
                  isLoading={status === 'loading'}
                  onRefresh={refresh}
                  columns={[
                    { title: 'Axis', key: 'type' },
                    { title: 'Name', key: 'name' },
                    {
                      title: 'EMA-200',
                      key: 'ema_200',
                      render: (row: any) => formatNum(row.ema_200),
                      className: 'text-right font-mono',
                    },
                    {
                      title: 'Dropped',
                      key: 'dropped_ema_200',
                      render: (row: any) => (row.dropped_ema_200 != null ? formatNum(row.dropped_ema_200) : ''),
                      className: 'text-right font-mono text-gray-400',
                    },
                    {
                      title: 'Samples',
                      key: 'sample_count',
                      render: (row: any) => row.sample_count.toLocaleString(),
                      className: 'text-right',
                    },
                  ]}
                  rows={cohortSummary}
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Dataset diagnostics (manifest-driven) */}
      {(data.exposure || data.interleaving || data.window_coverage || data.dropout || data.duplicates) && (
        <div>
          <h3 className="text-sm font-medium text-gray-300 mb-2">
            Dataset Diagnostics{' '}
            <span className="text-gray-500 font-normal">
              (manifest join: {data.exposure?.joined_draws?.toLocaleString() ?? 0} of{' '}
              {data.exposure?.total_draws?.toLocaleString() ?? 0} draws attributed)
            </span>
          </h3>
          <div className="space-y-4">
            {data.exposure && data.exposure.by_semantic_group.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-gray-500 mb-1">Exposure by semantic group (realized vs design share)</p>
                  <UniversalTable
                    isLoading={status === 'loading'}
                    onRefresh={refresh}
                    columns={[
                      { title: 'Group', key: 'name' },
                      { title: 'Files', key: 'files', className: 'text-right' },
                      { title: 'Draws', key: 'draws', className: 'text-right' },
                      { title: 'Expected', key: 'expected_share', render: pctCell('expected_share'), className: 'text-right font-mono' },
                      { title: 'Realized', key: 'realized_share', render: pctCell('realized_share'), className: 'text-right font-mono' },
                      { title: 'Ratio', key: 'ratio', render: ratioCell, className: 'text-right font-mono' },
                    ]}
                    rows={data.exposure.by_semantic_group}
                  />
                </div>
                {data.exposure.by_source_take.length > 0 && (
                  <div>
                    <p className="text-xs text-gray-500 mb-1">Exposure by source take (copies collapse duplicates)</p>
                    <UniversalTable
                      isLoading={status === 'loading'}
                      onRefresh={refresh}
                      columns={[
                        { title: 'Take', key: 'name' },
                        { title: 'Files', key: 'files', className: 'text-right' },
                        { title: 'Copies', key: 'copies', className: 'text-right' },
                        { title: 'Draws', key: 'draws', className: 'text-right' },
                        { title: 'Realized', key: 'realized_share', render: pctCell('realized_share'), className: 'text-right font-mono' },
                        { title: 'Ratio', key: 'ratio', render: ratioCell, className: 'text-right font-mono' },
                      ]}
                      rows={data.exposure.by_source_take}
                    />
                  </div>
                )}
              </div>
            )}
            {data.interleaving && data.interleaving.by_folder.length > 0 && (
              <div>
                <p className="text-xs text-gray-500 mb-1">
                  Draw interleaving per pooled folder (same-group adjacency vs shuffled baseline)
                  {data.interleaving.any_flagged && <FlagTag />}
                </p>
                <UniversalTable
                  isLoading={status === 'loading'}
                  onRefresh={refresh}
                  columns={[
                    { title: 'Folder', key: 'folder' },
                    { title: 'Epoch', key: 'epoch', render: (r: any) => r.epoch ?? '—', className: 'text-right' },
                    { title: 'Draws', key: 'draws', className: 'text-right' },
                    { title: 'Groups', key: 'groups', className: 'text-right' },
                    { title: 'Adjacency', key: 'adjacency_rate', render: pctCell('adjacency_rate'), className: 'text-right font-mono' },
                    { title: 'Baseline', key: 'baseline_rate', render: pctCell('baseline_rate'), className: 'text-right font-mono' },
                    { title: 'Ratio', key: 'ratio', render: ratioCell, className: 'text-right font-mono' },
                    { title: '', key: 'flagged', render: flagCell },
                  ]}
                  rows={data.interleaving.by_folder}
                />
              </div>
            )}
            {data.window_coverage && data.window_coverage.by_folder.length > 0 && (
              <div>
                <p className="text-xs text-gray-500 mb-1">
                  Sliding-window coverage (uniformity of window starts over each clip's legal range)
                  {data.window_coverage.any_flagged && <FlagTag />}
                </p>
                <UniversalTable
                  isLoading={status === 'loading'}
                  onRefresh={refresh}
                  columns={[
                    { title: 'Folder', key: 'folder' },
                    { title: 'Files', key: 'files', className: 'text-right' },
                    { title: 'Scored', key: 'files_scored', className: 'text-right' },
                    { title: 'Draws', key: 'draws', className: 'text-right' },
                    { title: 'Uniformity', key: 'mean_uniformity', render: numCell('mean_uniformity'), className: 'text-right font-mono' },
                    { title: 'Single-bin', key: 'files_single_bin', className: 'text-right' },
                    { title: '', key: 'flagged', render: flagCell },
                  ]}
                  rows={data.window_coverage.by_folder}
                />
              </div>
            )}
            {data.dropout && data.dropout.by_folder.length > 0 && (
              <div>
                <p className="text-xs text-gray-500 mb-1">
                  Caption dropout realized vs configured
                  {data.dropout.any_flagged && <FlagTag />}
                </p>
                <UniversalTable
                  isLoading={status === 'loading'}
                  onRefresh={refresh}
                  columns={[
                    { title: 'Folder', key: 'folder' },
                    { title: 'Draws', key: 'draws', className: 'text-right' },
                    { title: 'Dropped', key: 'caption_dropped', className: 'text-right' },
                    { title: 'Realized', key: 'realized_rate', render: pctCell('realized_rate'), className: 'text-right font-mono' },
                    { title: 'Configured', key: 'configured_rate', render: pctCell('configured_rate'), className: 'text-right font-mono' },
                    { title: 'Ratio', key: 'ratio', render: ratioCell, className: 'text-right font-mono' },
                    { title: '', key: 'flagged', render: flagCell },
                  ]}
                  rows={data.dropout.by_folder}
                />
              </div>
            )}
            {data.duplicates && data.duplicates.pairs.length > 0 && (
              <div>
                <p className="text-xs text-gray-500 mb-1">
                  Duplicate integrity (share copies vs their base)
                  {data.duplicates.any_flagged && <FlagTag />}
                </p>
                <UniversalTable
                  isLoading={status === 'loading'}
                  onRefresh={refresh}
                  columns={[
                    { title: 'Base', key: 'base', className: 'font-mono' },
                    { title: 'Base draws', key: 'base_draws', className: 'text-right' },
                    {
                      title: 'Copies',
                      key: 'copies',
                      render: (r: any) =>
                        Object.entries(r.copies as Record<string, number>)
                          .map(([k, v]) => `${k}: ${v}`)
                          .join(', '),
                      className: 'font-mono text-xs',
                    },
                    { title: 'Min ratio', key: 'min_ratio', render: numCell('min_ratio'), className: 'text-right font-mono' },
                    { title: 'Max ratio', key: 'max_ratio', render: numCell('max_ratio'), className: 'text-right font-mono' },
                    { title: '', key: 'flagged', render: flagCell },
                  ]}
                  rows={data.duplicates.pairs}
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Group × Noise Matrix */}
      {matrix.groups.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-300 mb-2">Group × Noise Matrix</h3>
          <div className="overflow-x-auto">
            <table className="text-sm text-left text-gray-300">
              <thead className="text-xs uppercase bg-gray-800 text-gray-400">
                <tr>
                  <th className="px-3 py-2">Group</th>
                  {matrix.buckets.map(b => (
                    <th key={b} className="px-3 py-2 text-right">
                      {b}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.groups.map((g, gi) => (
                  <tr
                    key={g}
                    className={`${gi % 2 === 0 ? 'bg-gray-900' : 'bg-gray-800'} border-b border-gray-700`}
                  >
                    <td className="px-3 py-2">{g}</td>
                    {matrix.buckets.map(b => {
                      const val = matrix.map[g]?.[b];
                      const norm = val != null ? (val - minVal) / range : 0;
                      // opacity 10-50% based on normalized value
                      const opacity = Math.round(10 + norm * 40);
                      return (
                        <td
                          key={b}
                          className="px-3 py-2 text-right font-mono"
                          style={{
                            backgroundColor: `rgba(239, 68, 68, ${opacity / 100})`,
                          }}
                        >
                          {val != null ? formatNum(val) : '\u2014'}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Preservation (DOP) drift pressure */}
      {data.has_preservation_data && (presGroupRows.length > 0 || presNoiseRows.length > 0) && (
        <div>
          <h3 className="text-sm font-medium text-gray-300 mb-2">
            Preservation Loss (DOP){' '}
            <span className="text-gray-500 font-normal">
              (class-prompt drift pressure; values are post-multiplier)
            </span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {presGroupRows.length > 0 && (
              <div>
                <p className="text-xs text-gray-500 mb-1">By Dataset Group</p>
                <UniversalTable
                  isLoading={status === 'loading'}
                  onRefresh={refresh}
                  columns={[
                    { title: 'Group', key: 'name' },
                    {
                      title: 'Pres EMA-200',
                      key: 'ema_200',
                      render: (row: any) => formatNum(row.ema_200),
                      className: 'text-right font-mono',
                    },
                    {
                      title: 'DOP Share',
                      key: 'dop_share',
                      render: (row: any) =>
                        row.dop_share != null ? (
                          <span
                            title={
                              row.dop_share_is_raw
                                ? 'preservation EMA / pre-multiplier normal-loss EMA (drift per unit of learning)'
                                : 'naive ratio vs post-multiplier loss: on multiplier-damped groups, true drift-per-unit-learning is share x multiplier (older snapshot without raw EMA)'
                            }
                          >
                            {(row.dop_share * 100).toPrecision(3)}%
                            {!row.dop_share_is_raw && <span className="text-amber-500">*</span>}
                          </span>
                        ) : (
                          ''
                        ),
                      className: 'text-right font-mono',
                    },
                    {
                      title: 'Samples',
                      key: 'sample_count',
                      render: (row: any) => row.sample_count.toLocaleString(),
                      className: 'text-right',
                    },
                  ]}
                  rows={presGroupRows}
                />
              </div>
            )}
            {presNoiseRows.length > 0 && (
              <div>
                <p className="text-xs text-gray-500 mb-1">By Noise Bucket / Expert / Phase</p>
                <UniversalTable
                  isLoading={status === 'loading'}
                  onRefresh={refresh}
                  columns={[
                    {
                      title: 'Bucket',
                      key: 'name',
                      render: (row: any) =>
                        row.type === 'boundary' ? `expert ${row.name}` : row.type === 'phase' ? `phase: ${row.name}` : row.name,
                    },
                    {
                      title: 'Pres EMA-200',
                      key: 'ema_200',
                      render: (row: any) => formatNum(row.ema_200),
                      className: 'text-right font-mono',
                    },
                    {
                      title: 'Samples',
                      key: 'sample_count',
                      render: (row: any) => row.sample_count.toLocaleString(),
                      className: 'text-right',
                    },
                  ]}
                  rows={presNoiseRows}
                />
              </div>
            )}
          </div>

          {presMatrix.groups.length > 0 && (
            <div className="mt-3">
              <p className="text-xs text-gray-500 mb-1">Preservation Group × Noise Matrix</p>
              <div className="overflow-x-auto">
                <table className="text-sm text-left text-gray-300">
                  <thead className="text-xs uppercase bg-gray-800 text-gray-400">
                    <tr>
                      <th className="px-3 py-2">Group</th>
                      {presMatrix.buckets.map(b => (
                        <th key={b} className="px-3 py-2 text-right">
                          {b}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {presMatrix.groups.map((g, gi) => (
                      <tr
                        key={g}
                        className={`${gi % 2 === 0 ? 'bg-gray-900' : 'bg-gray-800'} border-b border-gray-700`}
                      >
                        <td className="px-3 py-2">{g}</td>
                        {presMatrix.buckets.map(b => {
                          const val = presMatrix.map[g]?.[b];
                          const norm = val != null ? (val - presMin) / presRange : 0;
                          const opacity = Math.round(10 + norm * 40);
                          return (
                            <td
                              key={b}
                              className="px-3 py-2 text-right font-mono"
                              style={{
                                backgroundColor: `rgba(59, 130, 246, ${opacity / 100})`,
                              }}
                            >
                              {val != null ? formatNum(val) : '—'}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Worst Videos (global) */}
      {activeWorstVideos.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-300 mb-2">
            Worst {showReg ? 'Reg' : 'Concept'} Videos{' '}
            <span className="text-gray-500 font-normal">(top 20 by raw loss, pre-multiplier)</span>
          </h3>
          <UniversalTable
            isLoading={status === 'loading'}
            onRefresh={refresh}
            columns={worstVideoColumns(false, setPlayerClip)}
            rows={activeWorstVideos}
          />
        </div>
      )}

      {/* Clip player overlay */}
      {playerClip && playerClip.source_path && (
        <div
          className="fixed inset-0 z-50 bg-black/85 flex items-center justify-center p-6"
          onClick={() => setPlayerClip(null)}
        >
          <div
            className="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden max-w-4xl w-full"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-2 bg-gray-800">
              <div className="min-w-0">
                <p className="text-sm text-gray-200 font-mono truncate">{playerClip.source_id}</p>
                <p className="text-xs text-gray-500 truncate" title={playerClip.source_path}>
                  {playerClip.dataset_group} · mean {formatNum(playerClip.mean_loss_final)}
                  {playerClip.mean_loss_raw != null && ` · raw ${formatNum(playerClip.mean_loss_raw)}`}
                  {playerClip.z_score != null && ` · z ${playerClip.z_score.toFixed(1)}`}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setPlayerClip(null)}
                className="text-gray-400 hover:text-gray-100 ml-3 flex-shrink-0"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="bg-black flex items-center justify-center" style={{ maxHeight: '70vh' }}>
              {isVideoPath(playerClip.source_path) ? (
                <video
                  key={playerClip.source_path}
                  src={`/api/files/${encodeFilePathForUrl(playerClip.source_path)}`}
                  controls
                  autoPlay
                  loop
                  className="max-h-[70vh] w-auto max-w-full"
                />
              ) : (
                <img
                  src={`/api/files/${encodeFilePathForUrl(playerClip.source_path)}`}
                  alt={playerClip.source_id}
                  className="max-h-[70vh] w-auto max-w-full object-contain"
                />
              )}
            </div>
            {playerClip.caption && (
              <p className="px-4 py-2 text-xs text-gray-400 border-t border-gray-800">{playerClip.caption}</p>
            )}
          </div>
        </div>
      )}

      {/* Worst Clips by Group */}
      {Object.keys(activeWorstByGroup).length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-300 mb-2">
            Worst Clips by Group{' '}
            <span className="text-gray-500 font-normal">(sorted by z-score)</span>
          </h3>
          <div className="space-y-3">
            {Object.entries(activeWorstByGroup).map(([group, clips]) => (
              <div key={group}>
                <p className="text-xs text-gray-400 mb-1 font-medium">{group}</p>
                <UniversalTable
                  isLoading={status === 'loading'}
                  onRefresh={refresh}
                  columns={worstVideoColumns(true, setPlayerClip)}
                  rows={clips}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
