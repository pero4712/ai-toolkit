'use client';

import { useMemo } from 'react';
import { Job } from '@prisma/client';
import useJobLossAnalysis, { MatrixCell } from '@/hooks/useJobLossAnalysis';
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

export default function JobLossAnalysis({ job }: { job: Job }) {
  const { data, status, refresh } = useJobLossAnalysis(job.id, 10000);

  const groupSummary = useMemo(
    () => (data.summary ?? []).filter(r => r.type === 'group'),
    [data.summary],
  );
  const noiseSummary = useMemo(
    () => (data.summary ?? []).filter(r => r.type === 'noise' || r.type === 'boundary'),
    [data.summary],
  );

  const matrix = useMemo(() => pivotMatrix(data.matrix ?? []), [data.matrix]);

  // Heatmap range
  const allValues = useMemo(() => (data.matrix ?? []).map(c => c.mean_loss_final), [data.matrix]);
  const minVal = allValues.length ? Math.min(...allValues) : 0;
  const maxVal = allValues.length ? Math.max(...allValues) : 1;
  const range = maxVal - minVal || 1;

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
        <div className="flex-1" />
        {data.ema && (
          <div className="flex gap-3 text-xs">
            <span className="bg-gray-800 px-2 py-0.5 rounded text-blue-400">
              EMA-50: {formatNum(data.ema.ema_50)}
            </span>
            <span className="bg-gray-800 px-2 py-0.5 rounded text-emerald-400">
              EMA-200: {formatNum(data.ema.ema_200)}
            </span>
            <span className="bg-gray-800 px-2 py-0.5 rounded text-purple-400">
              EMA-1000: {formatNum(data.ema.ema_1000)}
            </span>
          </div>
        )}
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

      {/* Worst Videos */}
      {(data.worst_videos ?? []).length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-300 mb-2">
            Worst Videos{' '}
            <span className="text-gray-500 font-normal">(top 20 by mean loss)</span>
          </h3>
          <UniversalTable
            isLoading={status === 'loading'}
            onRefresh={refresh}
            columns={[
              { title: 'Source ID', key: 'source_id', className: 'font-mono' },
              { title: 'Group', key: 'dataset_group' },
              {
                title: 'Mean Loss',
                key: 'mean_loss_final',
                render: (row: any) => formatNum(row.mean_loss_final),
                className: 'text-right font-mono',
              },
              {
                title: 'Mean Raw',
                key: 'mean_loss_raw',
                render: (row: any) => formatNum(row.mean_loss_raw),
                className: 'text-right font-mono',
              },
              {
                title: 'P90',
                key: 'p90_loss_final',
                render: (row: any) => formatNum(row.p90_loss_final),
                className: 'text-right font-mono',
              },
              {
                title: 'Count',
                key: 'count',
                render: (row: any) => row.count.toLocaleString(),
                className: 'text-right',
              },
            ]}
            rows={data.worst_videos ?? []}
          />
        </div>
      )}
    </div>
  );
}
