'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { Job } from '@prisma/client';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ReferenceLine,
} from 'recharts';
import useJobTimingLog, { TIMING_SERIES_KEYS } from '@/hooks/useJobTimingLog';
import useMonitorStream from '@/hooks/useMonitorStream';
import { downsampleSeries } from '@/components/JobLossGraph';

// stacked step-phase series, bottom-up render order; palette matches JobLossGraph
const PHASES: { key: string; name: string; color: string }[] = [
  { key: 'timing/data_wait_ms', name: 'data wait', color: 'rgba(248,113,113,1)' }, // red-400
  { key: 'timing/vae_encode_ms', name: 'vae encode', color: 'rgba(251,191,36,1)' }, // amber-400
  { key: 'timing/i2v_cond_encode_ms', name: 'i2v cond encode', color: 'rgba(253,186,116,1)' }, // orange-300
  { key: 'timing/te_encode_ms', name: 'text encode', color: 'rgba(244,114,182,1)' }, // pink-400
  { key: 'timing/prior_forward_ms', name: 'prior forward', color: 'rgba(129,140,248,1)' }, // indigo-400
  { key: 'timing/forward_ms', name: 'forward', color: 'rgba(96,165,250,1)' }, // blue-400
  { key: 'timing/loss_ms', name: 'loss', color: 'rgba(34,211,238,1)' }, // cyan-400
  { key: 'timing/backward_ms', name: 'backward', color: 'rgba(52,211,153,1)' }, // emerald-400
  { key: 'timing/optimizer_ms', name: 'optimizer', color: 'rgba(167,139,250,1)' }, // purple-400
  { key: 'timing/ema_ms', name: 'ema', color: 'rgba(192,132,252,1)' }, // purple-300
  { key: 'timing/expert_swap_ms', name: 'expert swap', color: 'rgba(250,204,21,1)' }, // yellow-400
];

const OTHER_COLOR = 'rgba(156,163,175,0.7)'; // gray-400

function fmtMs(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return '—';
  if (ms >= 10000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${ms.toFixed(0)} ms`;
}

function latest(points: { step: number; value: number | null }[] | undefined): number | null {
  if (!points || points.length === 0) return null;
  const v = points[points.length - 1].value;
  return v == null ? null : v;
}

function StatCard({ label, value, accent, warn }: { label: string; value: string; accent?: string; warn?: boolean }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg px-4 py-3">
      <p className="text-xs text-gray-500">{label}</p>
      <p className={`text-lg font-mono ${warn ? 'text-red-400' : accent ?? 'text-gray-100'}`}>{value}</p>
    </div>
  );
}

function WarningBadge({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs rounded-lg px-3 py-2">
      {children}
    </div>
  );
}

/** GPU utilization sparkline for one GPU from the monitor's rolling history */
function GpuSparkline({
  label,
  points,
}: {
  label: string;
  points: { t: number; load: number }[];
}) {
  const troughs = points.filter(p => p.load < 10).length;
  const avg = points.length ? points.reduce((s, p) => s + p.load, 0) / points.length : 0;
  return (
    <div>
      <div className="flex items-baseline gap-2 mb-1 px-1">
        <span className="text-xs font-medium text-gray-300">{label}</span>
        <span className="text-xs text-gray-500">
          avg {avg.toFixed(0)}% · {troughs > 0 ? `${((troughs / Math.max(points.length, 1)) * 100).toFixed(0)}% idle samples` : 'no idle troughs'}
        </span>
      </div>
      <div className="bg-gray-950 rounded-lg border border-gray-800" style={{ height: 80 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={points} margin={{ top: 6, right: 8, bottom: 2, left: 8 }}>
            <YAxis domain={[0, 100]} hide />
            <XAxis dataKey="t" hide />
            <ReferenceLine y={10} stroke="rgba(248,113,113,0.35)" strokeDasharray="3 3" />
            <Area
              type="monotone"
              dataKey="load"
              stroke="rgba(96,165,250,1)"
              fill="rgba(96,165,250,0.20)"
              strokeWidth={1.5}
              isAnimationActive={false}
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function JobEfficiency({ job }: { job: Job }) {
  const { series, status } = useJobTimingLog(job.id, 5000);
  const monitor = useMonitorStream();

  // ---- headline stats (latest values) ----
  const stepMs = latest(series['timing/step_ms']);
  const dataWaitPct = latest(series['efficiency/data_wait_pct']);
  const vramPeak = latest(series['efficiency/vram_peak_gb']);
  const swapsPerStep = latest(series['efficiency/expert_swaps_per_step']);
  const i2vCondMs = latest(series['timing/i2v_cond_encode_ms']);

  // ---- GPU history for this job's assigned GPUs ----
  const gpuSparklines = useMemo(() => {
    const history = monitor.history ?? [];
    const gpus = monitor.gpu?.gpus ?? [];
    if (history.length === 0 || gpus.length === 0) return [];
    const assigned = String(job.gpu_ids ?? '')
      .split(',')
      .map(s => s.trim())
      .filter(s => /^\d+$/.test(s))
      .map(Number);
    // history rows are ordered like gpus (sorted by index)
    const positions =
      assigned.length > 0
        ? gpus.map((g, i) => ({ g, i })).filter(({ g }) => assigned.includes(g.index))
        : gpus.map((g, i) => ({ g, i }));
    return positions.map(({ g, i }) => ({
      label: `GPU ${g.index} — ${g.name}`,
      points: history
        .filter(h => h.gpus[i] != null)
        .map(h => ({ t: h.t, load: h.gpus[i].load })),
    }));
  }, [monitor.history, monitor.gpu, job.gpu_ids]);

  const gpuBusyAvg = useMemo(() => {
    const all = gpuSparklines.flatMap(s => s.points.map(p => p.load));
    return all.length ? all.reduce((a, b) => a + b, 0) / all.length : null;
  }, [gpuSparklines]);

  // ---- stacked breakdown chart data ----
  const chartData = useMemo(() => {
    const stepPoints = series['timing/step_ms'] ?? [];
    if (stepPoints.length === 0) return [];
    const byStep = new Map<number, Record<string, number>>();
    for (const p of stepPoints) {
      if (p.value != null) byStep.set(p.step, { step: p.step, step_ms: p.value });
    }
    for (const phase of PHASES) {
      for (const p of series[phase.key] ?? []) {
        const row = byStep.get(p.step);
        if (row && p.value != null) row[phase.name] = p.value;
      }
    }
    const rows = [...byStep.values()].sort((a, b) => a.step - b.step);
    for (const row of rows) {
      const accounted = PHASES.reduce((s, ph) => s + (row[ph.name] ?? 0), 0);
      row['other'] = Math.max(0, row.step_ms - accounted);
    }
    return rows;
  }, [series]);

  const activePhases = useMemo(
    () => PHASES.filter(ph => (series[ph.key] ?? []).some(p => p.value != null && p.value > 0.5)),
    [series],
  );

  // ---- warnings ----
  const warnings: string[] = [];
  if (dataWaitPct != null && dataWaitPct > 15) {
    warnings.push(
      `Input pipeline is starving the GPU (${dataWaitPct.toFixed(0)}% of step time spent waiting for data). Raise num_workers / prefetch_factor on the first dataset block, or cache latents to disk.`,
    );
  }
  if (swapsPerStep != null && swapsPerStep >= 1) {
    warnings.push(
      `Expert transformers are swapping CPU↔GPU ~${swapsPerStep.toFixed(1)}× per step (low_vram + frequent boundary switching). Raise switch_boundary_every or disable low_vram if VRAM allows.`,
    );
  }
  if (i2vCondMs != null && stepMs != null && i2vCondMs > stepMs * 0.1) {
    warnings.push(
      `First-frame conditioning re-encodes a full-length video through the VAE every step (${fmtMs(i2vCondMs)}/step) — a known wan22_14b_i2v architecture cost.`,
    );
  }

  const hasTimingData = chartData.length > 0;

  // ---- export (same pattern as Loss Analysis: download or clipboard) ----
  const [exportOpen, setExportOpen] = useState(false);
  const exportRef = useRef<HTMLDivElement>(null);

  const buildExportJson = useCallback(() => {
    const lastStep = chartData.length ? chartData[chartData.length - 1].step : null;

    const seriesOut: Record<string, { steps: number[]; values: number[] }> = {};
    for (const key of TIMING_SERIES_KEYS) {
      const pts = (series[key] ?? [])
        .filter(p => p.value != null && Number.isFinite(p.value))
        .map(p => ({ step: p.step, value: p.value as number }));
      if (pts.length === 0) continue;
      const ds = downsampleSeries(pts, 500);
      seriesOut[key] = {
        steps: ds.map(p => p.step),
        values: ds.map(p => Math.round(p.value * 1000) / 1000),
      };
    }

    const obj: Record<string, any> = {
      job: job.name,
      step: lastStep,
      generated_at: new Date().toISOString(),
      headline: {
        step_time_ms: stepMs,
        data_wait_pct: dataWaitPct,
        gpu_busy_avg_pct: gpuBusyAvg != null ? Math.round(gpuBusyAvg * 10) / 10 : null,
        vram_peak_gb: vramPeak,
        expert_swaps_per_step: swapsPerStep,
        i2v_cond_encode_ms: i2vCondMs,
      },
      warnings,
      // live snapshot only: the monitor keeps a 2-minute rolling window
      gpu_utilization_window: gpuSparklines.map(s => ({
        gpu: s.label,
        window_seconds: s.points.length
          ? Math.round((s.points[s.points.length - 1].t - s.points[0].t) / 1000)
          : 0,
        avg_pct:
          s.points.length > 0
            ? Math.round((s.points.reduce((a, p) => a + p.load, 0) / s.points.length) * 10) / 10
            : null,
        idle_sample_pct:
          s.points.length > 0
            ? Math.round((s.points.filter(p => p.load < 10).length / s.points.length) * 1000) / 10
            : null,
      })),
      series: seriesOut,
    };
    return JSON.stringify(obj, null, 2);
  }, [chartData, series, job.name, stepMs, dataWaitPct, gpuBusyAvg, vramPeak, swapsPerStep, i2vCondMs, warnings, gpuSparklines]);

  const handleDownload = useCallback(() => {
    const json = buildExportJson();
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const lastStep = chartData.length ? chartData[chartData.length - 1].step : 0;
    a.download = `efficiency-step${lastStep}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setExportOpen(false);
  }, [buildExportJson, chartData]);

  const handleCopy = useCallback(async () => {
    const json = buildExportJson();
    await navigator.clipboard.writeText(json);
    setExportOpen(false);
  }, [buildExportJson]);

  return (
    <div className="space-y-6 px-4 pb-8">
      <div className="flex items-center gap-3">
        <h2 className="text-lg font-medium text-gray-100">Efficiency</h2>
        {!hasTimingData && status === 'success' && (
          <span className="text-xs text-gray-500">
            No timing data yet — the trainer emits it every ~10 steps once training is running.
          </span>
        )}
        <div className="flex-1" />
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

      {/* Headline stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <StatCard label="Step time" value={fmtMs(stepMs)} />
        <StatCard
          label="Data wait"
          value={dataWaitPct != null ? `${dataWaitPct.toFixed(1)}%` : '—'}
          warn={dataWaitPct != null && dataWaitPct > 15}
        />
        <StatCard
          label="GPU busy (2 min)"
          value={gpuBusyAvg != null ? `${gpuBusyAvg.toFixed(0)}%` : '—'}
          warn={gpuBusyAvg != null && gpuBusyAvg < 70}
        />
        <StatCard label="VRAM peak" value={vramPeak != null ? `${vramPeak.toFixed(1)} GB` : '—'} />
        <StatCard
          label="Expert swaps / step"
          value={swapsPerStep != null ? swapsPerStep.toFixed(1) : '—'}
          warn={swapsPerStep != null && swapsPerStep >= 1}
        />
      </div>

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="space-y-2">
          {warnings.map((w, i) => (
            <WarningBadge key={i}>{w}</WarningBadge>
          ))}
        </div>
      )}

      {/* Live GPU utilization */}
      {gpuSparklines.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-300 mb-2">
            GPU Utilization <span className="text-gray-500 font-normal">(live, last 2 minutes)</span>
          </h3>
          <div className="space-y-3">
            {gpuSparklines.map(s => (
              <GpuSparkline key={s.label} label={s.label} points={s.points} />
            ))}
          </div>
        </div>
      )}

      {/* Step time breakdown over the run */}
      {hasTimingData && (
        <div>
          <h3 className="text-sm font-medium text-gray-300 mb-2">
            Step Time Breakdown{' '}
            <span className="text-gray-500 font-normal">
              (wall-clock per phase; async GPU work attributes to its sync point)
            </span>
          </h3>
          <div className="bg-gray-950 rounded-lg border border-gray-800" style={{ height: 340 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="step"
                  tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }}
                  tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  minTickGap={40}
                />
                <YAxis
                  tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }}
                  tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  width={72}
                  tickFormatter={(v: number) => fmtMs(v)}
                />
                <Tooltip
                  cursor={{ stroke: 'rgba(59,130,246,0.25)', strokeWidth: 1 }}
                  content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null;
                    return (
                      <div
                        style={{
                          background: 'rgba(17,24,39,0.96)',
                          border: '1px solid rgba(31,41,55,1)',
                          borderRadius: 10,
                          padding: '8px 12px',
                          fontSize: 12,
                        }}
                      >
                        <div style={{ color: 'rgba(255,255,255,0.75)', marginBottom: 4 }}>step {label}</div>
                        {[...payload].reverse().map((entry, i) => (
                          <div
                            key={i}
                            style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'rgba(255,255,255,0.9)' }}
                          >
                            <span style={{ color: String(entry.color) }}>●</span>
                            <span>
                              {entry.name}: {fmtMs(Number(entry.value))}
                            </span>
                          </div>
                        ))}
                      </div>
                    );
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: 4, color: 'rgba(255,255,255,0.7)', fontSize: 11 }} />
                {activePhases.map(ph => (
                  <Area
                    key={ph.key}
                    type="monotone"
                    dataKey={ph.name}
                    stackId="phases"
                    stroke={ph.color}
                    fill={ph.color.replace('1)', '0.35)')}
                    strokeWidth={1}
                    isAnimationActive={false}
                  />
                ))}
                <Area
                  type="monotone"
                  dataKey="other"
                  stackId="phases"
                  stroke={OTHER_COLOR}
                  fill="rgba(156,163,175,0.15)"
                  strokeWidth={1}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
