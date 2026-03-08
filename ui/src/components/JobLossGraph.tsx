'use client';

import { Job } from '@prisma/client';
import useJobLossLog, { LossPoint } from '@/hooks/useJobLossLog';
import { useMemo, useState, useCallback } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';

// ---------------------------------------------------------------------------
// Utilities (unchanged)
// ---------------------------------------------------------------------------

function formatNum(v: number) {
  if (!Number.isFinite(v)) return '';
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(3);
  if (Math.abs(v) >= 1) return v.toFixed(4);
  return v.toPrecision(4);
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function emaSmoothPoints(points: { step: number; value: number }[], alpha: number) {
  if (points.length === 0) return [];
  const a = clamp01(alpha);
  const out: { step: number; value: number }[] = new Array(points.length);
  let prev = points[0].value;
  out[0] = { step: points[0].step, value: prev };
  for (let i = 1; i < points.length; i++) {
    const x = points[i].value;
    prev = a * x + (1 - a) * prev;
    out[i] = { step: points[i].step, value: prev };
  }
  return out;
}

function hashToIndex(str: string, mod: number) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h) % mod;
}

const PALETTE = [
  'rgba(96,165,250,1)',   // blue-400
  'rgba(52,211,153,1)',   // emerald-400
  'rgba(167,139,250,1)',  // purple-400
  'rgba(251,191,36,1)',   // amber-400
  'rgba(244,114,182,1)',  // pink-400
  'rgba(248,113,113,1)',  // red-400
  'rgba(34,211,238,1)',   // cyan-400
  'rgba(129,140,248,1)',  // indigo-400
];

function strokeForKey(key: string) {
  return PALETTE[hashToIndex(key, PALETTE.length)];
}

// ---------------------------------------------------------------------------
// Key categorization
// ---------------------------------------------------------------------------

type Section = 'overall' | 'noise' | 'boundary' | 'group' | 'other' | 'skip';

function categorizeKey(key: string): Section {
  if (key === 'loss') return 'overall';
  if (key.startsWith('loss_by_reg_ema/')) return 'overall';
  if (key.startsWith('loss_by_noise_ema/')) return 'noise';
  if (key.startsWith('loss_by_noise_reg_ema/')) return 'skip';
  if (key.startsWith('loss_by_boundary/')) return 'boundary';
  if (key.startsWith('loss_by_boundary_reg/')) return 'skip';
  if (key.startsWith('loss_by_group_ema/')) return 'group';
  if (key.startsWith('loss_by_group_reg_ema/')) return 'skip';
  if (key.includes('samples')) return 'skip';
  if (key === 'step_time_ms') return 'skip';
  return 'other';
}

function groupNameFromKey(key: string): string {
  const raw = key.replace('loss_by_group_ema/', '');
  return raw.charAt(0).toUpperCase() + raw.slice(1);
}

/** Friendly display name for a loss key */
function displayName(key: string): string {
  if (key === 'loss') return 'loss';
  // strip common prefixes for readability
  for (const prefix of ['loss_by_reg_ema/', 'loss_by_noise_ema/', 'loss_by_boundary/', 'loss_by_group_ema/']) {
    if (key.startsWith(prefix)) return key.slice(prefix.length);
  }
  return key;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type PerSeriesMap = Record<string, { raw: { step: number; value: number }[]; smooth: { step: number; value: number }[] }>;

// ---------------------------------------------------------------------------
// Sub-chart component
// ---------------------------------------------------------------------------

interface LossSubChartProps {
  title: string;
  subtitle?: string;
  keys: string[];
  perSeries: PerSeriesMap;
  showRaw: boolean;
  showSmoothed: boolean;
  useLogScale: boolean;
  clipOutliers: boolean;
  height?: number;
}

function LossSubChart({ title, subtitle, keys, perSeries, showRaw, showSmoothed, useLogScale, clipOutliers, height = 240 }: LossSubChartProps) {
  const chartData = useMemo(() => {
    const map = new Map<number, any>();
    for (const key of keys) {
      const s = perSeries[key];
      if (!s) continue;
      for (const p of s.raw) {
        const row = map.get(p.step) ?? { step: p.step };
        row[`${key}__raw`] = p.value;
        map.set(p.step, row);
      }
      for (const p of s.smooth) {
        const row = map.get(p.step) ?? { step: p.step };
        row[`${key}__smooth`] = p.value;
        map.set(p.step, row);
      }
    }
    const arr = Array.from(map.values());
    arr.sort((a, b) => a.step - b.step);
    return arr;
  }, [keys, perSeries]);

  const yDomain = useMemo((): [number | 'auto', number | 'auto'] => {
    if (!clipOutliers || chartData.length < 10) return ['auto', 'auto'];
    const vals: number[] = [];
    for (const row of chartData) {
      for (const key of keys) {
        const k = showSmoothed ? `${key}__smooth` : `${key}__raw`;
        const v = row[k];
        if (typeof v === 'number' && Number.isFinite(v)) vals.push(v);
      }
    }
    if (vals.length < 10) return ['auto', 'auto'];
    vals.sort((a, b) => a - b);
    const lo = vals[Math.floor(vals.length * 0.02)];
    const hi = vals[Math.ceil(vals.length * 0.98) - 1];
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return ['auto', 'auto'];
    return [lo, hi];
  }, [clipOutliers, chartData, keys, showSmoothed]);

  if (chartData.length < 2) return null;

  const showLegend = keys.length > 1;

  return (
    <div>
      <div className="flex items-baseline gap-2 mb-1 px-1">
        <span className="text-xs font-medium text-gray-300">{title}</span>
        {subtitle && <span className="text-xs text-gray-500">{subtitle}</span>}
      </div>
      <div className="bg-gray-950 rounded-lg border border-gray-800 relative" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis
              dataKey="step"
              tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }}
              tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
              minTickGap={40}
            />
            <YAxis
              scale={useLogScale ? 'log' : 'linear'}
              tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }}
              tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
              axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
              width={72}
              tickFormatter={formatNum}
              domain={yDomain}
              allowDataOverflow={clipOutliers}
            />
            <Tooltip
              cursor={{ stroke: 'rgba(59,130,246,0.25)', strokeWidth: 1 }}
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null;
                const sorted = [...payload].sort((a, b) => Number(b.value) - Number(a.value));
                return (
                  <div style={{
                    background: 'rgba(17,24,39,0.96)',
                    border: '1px solid rgba(31,41,55,1)',
                    borderRadius: 10,
                    padding: '8px 12px',
                    fontSize: 12,
                  }}>
                    <div style={{ color: 'rgba(255,255,255,0.75)', marginBottom: 4 }}>step {label}</div>
                    {sorted.map((entry, i) => (
                      <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'rgba(255,255,255,0.9)' }}>
                        <span style={{ color: String(entry.color) }}>●</span>
                        <span>{entry.name}: {formatNum(Number(entry.value))}</span>
                      </div>
                    ))}
                  </div>
                );
              }}
            />
            {showLegend && (
              <Legend wrapperStyle={{ paddingTop: 4, color: 'rgba(255,255,255,0.7)', fontSize: 11 }} />
            )}
            {keys.map(k => {
              const color = strokeForKey(k);
              const name = displayName(k);
              return (
                <g key={k}>
                  {showRaw && (
                    <Line
                      type="monotone"
                      dataKey={`${k}__raw`}
                      name={`${name} (raw)`}
                      stroke={color.replace('1)', '0.40)')}
                      strokeWidth={1.25}
                      dot={false}
                      isAnimationActive={false}
                    />
                  )}
                  {showSmoothed && (
                    <Line
                      type="monotone"
                      dataKey={`${k}__smooth`}
                      name={name}
                      stroke={color}
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  )}
                </g>
              );
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const GROUP_CAP = 8;

export default function JobLossGraph({ job }: { job: Job }) {
  const { series, lossKeys, status, refreshLoss } = useJobLossLog(job.id, 2000);

  // Shared controls
  const [useLogScale, setUseLogScale] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [showSmoothed, setShowSmoothed] = useState(true);
  const [smoothing, setSmoothing] = useState(90);
  const [plotStride, setPlotStride] = useState(1);
  const [windowSize, setWindowSize] = useState<number>(4000);
  const [clipOutliers, setClipOutliers] = useState(false);

  // Group filter & cap
  const [groupFilter, setGroupFilter] = useState('');
  const [showAllGroups, setShowAllGroups] = useState(false);

  // Compare mode
  const [compareOpen, setCompareOpen] = useState(false);
  const [compareSelected, setCompareSelected] = useState<Record<string, boolean>>({});

  // Bucket keys into sections
  const sections = useMemo(() => {
    const overall: string[] = [];
    const noise: string[] = [];
    const boundary: string[] = [];
    const group: string[] = [];
    const other: string[] = [];
    const allNonSkipped: string[] = [];

    for (const k of lossKeys) {
      const sec = categorizeKey(k);
      if (sec === 'skip') continue;
      allNonSkipped.push(k);
      switch (sec) {
        case 'overall': overall.push(k); break;
        case 'noise': noise.push(k); break;
        case 'boundary': boundary.push(k); break;
        case 'group': group.push(k); break;
        case 'other': other.push(k); break;
      }
    }
    return { overall, noise, boundary, group, other, allNonSkipped };
  }, [lossKeys]);

  // Process all non-skipped series (shared across all sub-charts)
  const perSeries = useMemo(() => {
    const stride = Math.max(1, plotStride | 0);
    const t = clamp01(smoothing / 100);
    const alpha = 1.0 - t * 0.98;

    const out: PerSeriesMap = {};
    for (const key of sections.allNonSkipped) {
      const pts: LossPoint[] = series[key] ?? [];
      let raw = pts
        .filter(p => p.value !== null && Number.isFinite(p.value as number))
        .map(p => ({ step: p.step, value: p.value as number }))
        .filter(p => (useLogScale ? p.value > 0 : true))
        .filter((_, idx) => idx % stride === 0);
      if (windowSize > 0 && raw.length > windowSize) {
        raw = raw.slice(raw.length - windowSize);
      }
      const smooth = emaSmoothPoints(raw, alpha);
      out[key] = { raw, smooth };
    }
    return out;
  }, [series, sections.allNonSkipped, smoothing, plotStride, windowSize, useLogScale]);

  /** Get latest value for a key (smoothed preferred, fallback raw). Used for sorting. */
  const latestValue = useCallback((key: string): number => {
    const s = perSeries[key];
    if (!s) return -Infinity;
    if (s.smooth.length) return s.smooth[s.smooth.length - 1].value;
    if (s.raw.length) return s.raw[s.raw.length - 1].value;
    return -Infinity;
  }, [perSeries]);

  // Sort group keys by latest value descending (hardest first)
  const sortedGroupKeys = useMemo(() => {
    return [...sections.group].sort((a, b) => latestValue(b) - latestValue(a));
  }, [sections.group, latestValue]);

  // Filter and cap groups
  const visibleGroupKeys = useMemo(() => {
    let keys = sortedGroupKeys;
    if (groupFilter) {
      const lower = groupFilter.toLowerCase();
      keys = keys.filter(k => groupNameFromKey(k).toLowerCase().includes(lower));
    }
    // Cap only when no filter text and not showing all
    if (!groupFilter && !showAllGroups && keys.length > GROUP_CAP) {
      return keys.slice(0, GROUP_CAP);
    }
    return keys;
  }, [sortedGroupKeys, groupFilter, showAllGroups]);

  const totalGroupCount = sortedGroupKeys.length;
  const isGroupCapped = !groupFilter && !showAllGroups && totalGroupCount > GROUP_CAP;

  // Compare: selected keys
  const compareKeys = useMemo(
    () => sections.allNonSkipped.filter(k => compareSelected[k]),
    [sections.allNonSkipped, compareSelected],
  );

  // Compare: preset helpers
  const applyPreset = useCallback((keys: string[]) => {
    const next: Record<string, boolean> = {};
    for (const k of keys) next[k] = true;
    setCompareSelected(next);
  }, []);

  const presetConceptVsReg = useMemo(
    () => sections.overall.filter(k => k.startsWith('loss_by_reg_ema/')),
    [sections.overall],
  );
  const presetTop3 = useMemo(
    () => sortedGroupKeys.slice(0, 3),
    [sortedGroupKeys],
  );

  // Overall status (latest loss value)
  const hasAnyData = useMemo(
    () => sections.allNonSkipped.some(k => {
      const s = perSeries[k];
      return s && (s.raw.length > 1 || s.smooth.length > 1);
    }),
    [sections.allNonSkipped, perSeries],
  );

  const latestLoss = useMemo(() => {
    const s = perSeries['loss'];
    if (!s) return null;
    const last = s.smooth.length ? s.smooth[s.smooth.length - 1] : (s.raw.length ? s.raw[s.raw.length - 1] : null);
    return last;
  }, [perSeries]);

  const sharedProps = { perSeries, showRaw, showSmoothed, useLogScale, clipOutliers };

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col">
      {/* Header */}
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-blue-400" />
          <h2 className="text-gray-100 text-sm font-medium">Loss graph</h2>
          <span className="text-xs text-gray-400">
            {status === 'loading' && 'Loading...'}
            {status === 'refreshing' && 'Refreshing...'}
            {status === 'error' && 'Error'}
            {status === 'success' && !hasAnyData && 'No data yet'}
            {status === 'success' && hasAnyData && latestLoss && (
              <>step {latestLoss.step.toLocaleString()} &middot; loss {formatNum(latestLoss.value)}</>
            )}
          </span>
        </div>
        <button
          type="button"
          onClick={refreshLoss}
          className="px-3 py-1 rounded-md text-xs bg-gray-700/60 hover:bg-gray-700 text-gray-200 border border-gray-700"
        >
          Refresh
        </button>
      </div>

      {/* Charts */}
      <div className="px-4 pt-4 pb-4 space-y-4">
        {!hasAnyData ? (
          <div className="bg-gray-950 rounded-lg border border-gray-800 h-96 flex items-center justify-center text-sm text-gray-400">
            {status === 'error' ? 'Failed to load loss logs.' : 'Waiting for loss points...'}
          </div>
        ) : (
          <>
            {/* Compare chart (above everything, only when open + keys selected) */}
            {compareOpen && compareKeys.length > 0 && (
              <LossSubChart
                title="Compare"
                subtitle={`(${compareKeys.length} selected)`}
                keys={compareKeys}
                height={384}
                {...sharedProps}
              />
            )}

            {/* Overall */}
            {sections.overall.length > 0 && (
              <LossSubChart
                title="Overall"
                keys={sections.overall}
                height={384}
                {...sharedProps}
              />
            )}

            {/* Noise Buckets */}
            {sections.noise.length > 0 && (
              <LossSubChart
                title="Noise Buckets"
                subtitle={`(${sections.noise.length})`}
                keys={sections.noise}
                {...sharedProps}
              />
            )}

            {/* Boundaries */}
            {sections.boundary.length > 0 && (
              <LossSubChart
                title="Boundaries"
                subtitle={`(${sections.boundary.length})`}
                keys={sections.boundary}
                {...sharedProps}
              />
            )}

            {/* Other Metrics */}
            {sections.other.length > 0 && (
              <LossSubChart
                title="Other Metrics"
                subtitle={`(${sections.other.length})`}
                keys={sections.other}
                {...sharedProps}
              />
            )}

            {/* Per-group charts */}
            {sortedGroupKeys.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <span className="text-xs font-medium text-gray-300">Groups</span>
                  <span className="text-xs text-gray-500">({totalGroupCount})</span>
                  {totalGroupCount >= 4 && (
                    <input
                      type="text"
                      value={groupFilter}
                      onChange={e => { setGroupFilter(e.target.value); setShowAllGroups(false); }}
                      placeholder="Filter groups..."
                      className="bg-gray-950 border border-gray-800 rounded px-2 py-0.5 text-xs text-gray-300 placeholder-gray-600 w-40"
                    />
                  )}
                </div>

                {visibleGroupKeys.map(k => (
                  <LossSubChart
                    key={k}
                    title={groupNameFromKey(k)}
                    keys={[k]}
                    {...sharedProps}
                  />
                ))}

                {isGroupCapped && (
                  <button
                    type="button"
                    onClick={() => setShowAllGroups(true)}
                    className="w-full py-2 text-xs text-gray-400 hover:text-gray-200 bg-gray-950 border border-gray-800 rounded-lg transition-colors"
                  >
                    Show all {totalGroupCount} groups
                  </button>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* Controls */}
      <div className="px-4 pb-2">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* Display toggles */}
          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <label className="block text-xs text-gray-400 mb-2">Display</label>
            <div className="flex flex-wrap gap-2">
              <ToggleButton checked={showSmoothed} onClick={() => setShowSmoothed(v => !v)} label="Smoothed" />
              <ToggleButton checked={showRaw} onClick={() => setShowRaw(v => !v)} label="Raw" />
              <ToggleButton checked={useLogScale} onClick={() => setUseLogScale(v => !v)} label="Log Y" />
              <ToggleButton checked={clipOutliers} onClick={() => setClipOutliers(v => !v)} label="Clip outliers" />
            </div>
          </div>

          {/* Compare panel */}
          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs text-gray-400">Compare</label>
              <ToggleButton
                checked={compareOpen}
                onClick={() => setCompareOpen(v => !v)}
                label={compareOpen ? 'Close' : 'Open'}
              />
            </div>
            {compareOpen && (
              <div className="space-y-2">
                {/* Quick presets */}
                <div className="flex flex-wrap gap-1.5">
                  {presetConceptVsReg.length >= 2 && (
                    <PresetButton label="Concept vs Reg" onClick={() => applyPreset(presetConceptVsReg)} />
                  )}
                  {sections.noise.length > 0 && (
                    <PresetButton label="Noise buckets" onClick={() => applyPreset(sections.noise)} />
                  )}
                  {sections.boundary.length > 0 && (
                    <PresetButton label="Boundaries" onClick={() => applyPreset(sections.boundary)} />
                  )}
                  {presetTop3.length > 0 && (
                    <PresetButton label="Top 3 hardest" onClick={() => applyPreset(presetTop3)} />
                  )}
                  {compareKeys.length > 0 && (
                    <PresetButton label="Clear" onClick={() => setCompareSelected({})} />
                  )}
                </div>
                {/* Full key toggles */}
                <div className="flex flex-wrap gap-1.5 max-h-40 overflow-y-auto">
                  {sections.allNonSkipped.map(k => (
                    <button
                      key={k}
                      type="button"
                      onClick={() => setCompareSelected(prev => ({ ...prev, [k]: !prev[k] }))}
                      className={[
                        'px-2 py-0.5 rounded text-xs border transition-colors',
                        compareSelected[k]
                          ? 'bg-blue-500/10 text-blue-300 border-blue-500/30'
                          : 'bg-gray-900 text-gray-400 border-gray-800 hover:text-gray-200',
                      ].join(' ')}
                      title={k}
                    >
                      <span className="inline-block h-1.5 w-1.5 rounded-full mr-1.5" style={{ background: strokeForKey(k) }} />
                      {displayName(k)}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Smoothing */}
          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Smoothing</label>
              <span className="text-xs text-gray-300">{smoothing}%</span>
            </div>
            <input
              type="range" min={0} max={100} value={smoothing}
              onChange={e => setSmoothing(Number(e.target.value))}
              className="w-full accent-blue-500"
              disabled={!showSmoothed}
            />
          </div>

          {/* Plot stride */}
          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Plot stride</label>
              <span className="text-xs text-gray-300">every {plotStride} pt</span>
            </div>
            <input
              type="range" min={1} max={20} value={plotStride}
              onChange={e => setPlotStride(Number(e.target.value))}
              className="w-full accent-blue-500"
            />
            <div className="mt-2 text-[11px] text-gray-500">UI downsample for huge runs.</div>
          </div>

          {/* Window */}
          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3 md:col-span-2">
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Window (last N points)</label>
              <span className="text-xs text-gray-300">{windowSize === 0 ? 'all' : windowSize.toLocaleString()}</span>
            </div>
            <input
              type="range" min={0} max={20000} step={250} value={windowSize}
              onChange={e => setWindowSize(Number(e.target.value))}
              className="w-full accent-blue-500"
            />
            <div className="mt-2 text-[11px] text-gray-500">
              Set to 0 to show all (not recommended for very long runs).
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small UI components
// ---------------------------------------------------------------------------

function ToggleButton({ checked, onClick, label }: { checked: boolean; onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'px-3 py-1 rounded-md text-xs border transition-colors',
        checked
          ? 'bg-blue-500/10 text-blue-300 border-blue-500/30 hover:bg-blue-500/15'
          : 'bg-gray-900 text-gray-300 border-gray-800 hover:bg-gray-800/60',
      ].join(' ')}
      aria-pressed={checked}
    >
      {label}
    </button>
  );
}

function PresetButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="px-2 py-0.5 rounded text-xs bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-700 transition-colors"
    >
      {label}
    </button>
  );
}
