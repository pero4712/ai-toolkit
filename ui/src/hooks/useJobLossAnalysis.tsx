'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { apiClient } from '@/utils/api';

export interface WorstVideo {
  source_id: string;
  dataset_group: string;
  mean_loss_final: number;
  mean_loss_raw: number;
  p90_loss_final: number;
  count: number;
}

export interface MatrixCell {
  group: string;
  noise_bucket: string;
  mean_loss_final: number;
}

export interface SummaryRow {
  type: 'group' | 'noise' | 'boundary';
  name: string;
  ema_200: number;
  sample_count: number;
}

export interface LossAnalysisData {
  available: boolean;
  enabled?: boolean;
  step?: number;
  wall_time?: number;
  run_id?: string;
  worst_videos?: WorstVideo[];
  matrix?: MatrixCell[];
  summary?: SummaryRow[];
  ema?: { ema_50: number; ema_200: number; ema_1000: number };
}

export default function useJobLossAnalysis(
  jobID: string,
  reloadInterval: number | null = 10000,
) {
  const [data, setData] = useState<LossAnalysisData>({ available: false });
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const inFlightRef = useRef(false);

  const refresh = useCallback(async () => {
    if (!jobID || inFlightRef.current) return;
    inFlightRef.current = true;
    setStatus(prev => (prev === 'success' ? prev : 'loading'));

    try {
      const res = await apiClient.get(`/api/jobs/${jobID}/loss-analysis`);
      setData(res.data);
      setStatus('success');
    } catch {
      setStatus('error');
    } finally {
      inFlightRef.current = false;
    }
  }, [jobID]);

  useEffect(() => {
    setData({ available: false });
    setStatus('idle');
    refresh();

    if (reloadInterval) {
      const id = setInterval(refresh, reloadInterval);
      return () => clearInterval(id);
    }
  }, [jobID, reloadInterval, refresh]);

  return { data, status, refresh };
}
