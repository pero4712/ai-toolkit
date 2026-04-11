'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { Job } from '@prisma/client';
import useJobLossAnalysis, { MatrixCell, SummaryRow, WorstVideo } from '@/hooks/useJobLossAnalysis';
import UniversalTable from './UniversalTable';

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
function worstVideoColumns(compact = false) {
  const cols: any[] = [
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

  const matrix = useMemo(() => pivotMatrix(activeMatrix), [activeMatrix]);

  // Heatmap range
  const allValues = useMemo(() => activeMatrix.map(c => c.mean_loss_final), [activeMatrix]);
  const minVal = allValues.length ? Math.min(...allValues) : 0;
  const maxVal = allValues.length ? Math.max(...allValues) : 1;
  const range = maxVal - minVal || 1;

  // Export
  const [exportOpen, setExportOpen] = useState(false);
  const exportRef = useRef<HTMLDivElement>(null);

  const buildExportJson = useCallback(() => {
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
      const noises = summary.filter(r => r.type === 'noise' || r.type === 'boundary');

      if (groups.length > 0) {
        section.group_summary = groups.map(r => ({
          name: r.name,
          ema_200: r.ema_200,
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
    }

    return JSON.stringify(obj, null, 2);
  }, [data]);

  const handleDownload = useCallback(() => {
    const json = buildExportJson();
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `loss-analysis-step${data.step ?? 0}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setExportOpen(false);
  }, [buildExportJson, data.step]);

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

      {/* Worst Videos (global) */}
      {activeWorstVideos.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-300 mb-2">
            Worst {showReg ? 'Reg' : 'Concept'} Videos{' '}
            <span className="text-gray-500 font-normal">(top 20 by mean loss)</span>
          </h3>
          <UniversalTable
            isLoading={status === 'loading'}
            onRefresh={refresh}
            columns={worstVideoColumns(false)}
            rows={activeWorstVideos}
          />
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
                  columns={worstVideoColumns(true)}
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
