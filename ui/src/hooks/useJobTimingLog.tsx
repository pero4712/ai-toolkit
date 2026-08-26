'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';

export interface TimingPoint {
  step: number;
  value: number | null;
}

// Curated keys emitted by BaseSDTrainProcess._log_timing_metrics. Fetching a
// fixed list (intersected with the DB's key list) keeps the poll fan-out small
// compared to useJobLossLog, which fetches every key.
export const TIMING_SERIES_KEYS = [
  'timing/step_ms',
  'timing/data_wait_ms',
  'timing/vae_encode_ms',
  'timing/te_encode_ms',
  'timing/forward_ms',
  'timing/prior_forward_ms',
  'timing/loss_ms',
  'timing/backward_ms',
  'timing/optimizer_ms',
  'timing/ema_ms',
  'timing/expert_swap_ms',
  'timing/i2v_cond_encode_ms',
  'efficiency/data_wait_pct',
  'efficiency/vram_peak_gb',
  'efficiency/expert_swaps_per_step',
];

type SeriesMap = Record<string, TimingPoint[]>;

export default function useJobTimingLog(jobID: string, reloadInterval: null | number = 5000) {
  const [series, setSeries] = useState<SeriesMap>({});
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const inFlightRef = useRef(false);
  const lastStepByKeyRef = useRef<Record<string, number | null>>({});

  const refresh = useCallback(async () => {
    if (!jobID || inFlightRef.current) return;
    inFlightRef.current = true;
    setStatus(prev => (prev === 'success' ? prev : 'loading'));
    try {
      // one cheap call to learn which keys exist in this job's loss_log.db
      const first = await apiClient
        .get(`/api/jobs/${jobID}/loss`, { params: { key: 'loss', limit: 1 } })
        .then(res => res.data as { keys?: string[] });
      const available = new Set(first.keys ?? []);
      const wanted = TIMING_SERIES_KEYS.filter(k => available.has(k));

      const results = await Promise.all(
        wanted.map(k => {
          const params: Record<string, any> = { key: k, limit: 20000 };
          const since = lastStepByKeyRef.current[k];
          if (since != null) params.since_step = since;
          return apiClient
            .get(`/api/jobs/${jobID}/loss`, { params })
            .then(res => res.data as { key: string; points?: TimingPoint[] });
        }),
      );

      setSeries(prev => {
        const next: SeriesMap = { ...prev };
        for (const r of results) {
          const newPoints = (r.points ?? []).filter(p => p.value !== null);
          const existing = next[r.key] ?? [];
          const prevLast = existing.length ? existing[existing.length - 1].step : null;
          const fresh = prevLast == null ? newPoints : newPoints.filter(p => p.step > prevLast);
          next[r.key] = fresh.length ? [...existing, ...fresh] : existing;
          const arr = next[r.key];
          lastStepByKeyRef.current[r.key] = arr.length ? arr[arr.length - 1].step : null;
        }
        return next;
      });
      setStatus('success');
    } catch (err) {
      console.error('Error fetching timing logs:', err);
      setStatus('error');
    } finally {
      inFlightRef.current = false;
    }
  }, [jobID]);

  useEffect(() => {
    lastStepByKeyRef.current = {};
    setSeries({});
    setStatus('idle');
  }, [jobID]);

  usePollLoop(refresh, reloadInterval, [jobID]);

  return { series, status, refresh };
}
