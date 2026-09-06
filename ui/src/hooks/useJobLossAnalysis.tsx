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
  z_score?: number;
  is_outlier?: boolean;
  is_new_outlier?: boolean;
  trend?: 'up' | 'down' | 'flat' | '~';
  trend_delta?: number;
  loss_ratio?: number;
  caption?: string | null;
  // media path on the training machine; present in snapshots from tracker
  // versions that record it — enables the play-clip button
  source_path?: string | null;
}

export interface MatrixCell {
  group: string;
  noise_bucket: string;
  mean_loss_final: number;
}

export interface SummaryRow {
  type: 'group' | 'noise' | 'boundary' | 'phase' | 'level';
  name: string;
  ema_200: number;
  sample_count: number;
  std_loss?: number;
  min_loss?: number;
  max_loss?: number;
  clip_count?: number;
  relative_difficulty?: number;
  // group rows only, tracker >= preservation extension
  ema_200_raw?: number;
  loss_multiplier?: number;
  // dropped-caption companion (unconditional draws), tracker >= manifest P2
  dropped_ema_200?: number;
  dropped_samples?: number;
}

/** Manifest-driven dataset diagnostics (loss_analysis.json sections) */
export interface ExposureRow {
  name: string;
  files: number;
  copies?: number;
  draws: number;
  reg_draws: number;
  expected_share: number | null;
  realized_share: number | null;
  ratio: number | null;
  expected_draws?: number | null;
}
export interface InterleavingRow {
  folder: string;
  epoch: number | null;
  draws: number;
  groups: number;
  adjacency_rate: number | null;
  baseline_rate: number | null;
  ratio: number | null;
  flagged: boolean;
}
export interface WindowCoverageRow {
  folder: string;
  files: number;
  draws: number;
  files_scored: number;
  mean_uniformity: number | null;
  files_single_bin: number;
  flagged: boolean;
}
export interface WindowFileRow {
  source_id: string;
  folder: string;
  draws: number;
  max_start: number;
  bins: number[];
  uniformity: number | null;
}
export interface DropoutRow {
  folder: string;
  draws: number;
  caption_dropped: number;
  realized_rate: number | null;
  configured_rate: number;
  ratio: number | null;
  token_dropout_rate: number;
  flagged: boolean;
}
export interface DuplicateRow {
  base: string;
  base_draws: number;
  copies: Record<string, number>;
  min_ratio: number | null;
  max_ratio: number | null;
  flagged: boolean;
}

export interface LossAnalysisData {
  available: boolean;
  enabled?: boolean;
  step?: number;
  wall_time?: number;
  run_id?: string;
  // Combined (backwards compat)
  worst_videos?: WorstVideo[];
  matrix?: MatrixCell[];
  summary?: SummaryRow[];
  ema?: { ema_50: number; ema_200: number; ema_1000: number };
  // Reg/concept split
  worst_videos_concept?: WorstVideo[];
  worst_videos_reg?: WorstVideo[];
  summary_concept?: SummaryRow[];
  summary_reg?: SummaryRow[];
  matrix_concept?: MatrixCell[];
  matrix_reg?: MatrixCell[];
  ema_concept?: { ema_50: number; ema_200: number; ema_1000: number };
  ema_reg?: { ema_50: number; ema_200: number; ema_1000: number };
  has_reg_data?: boolean;
  samples_total?: number;
  samples_concept_total?: number;
  samples_reg_total?: number;
  // Per-group worst clips
  worst_by_group_concept?: Record<string, WorstVideo[]>;
  worst_by_group_reg?: Record<string, WorstVideo[]>;
  // Preservation (DOP) breakdowns
  has_preservation_data?: boolean;
  preservation_summary_concept?: SummaryRow[];
  preservation_summary_reg?: SummaryRow[];
  preservation_matrix_concept?: MatrixCell[];
  preservation_matrix_reg?: MatrixCell[];
  // Manifest-driven diagnostics
  exposure?: {
    epoch: number | null;
    total_draws: number;
    joined_draws: number;
    unjoined_draws: number;
    manifest_files: number;
    by_semantic_group: ExposureRow[];
    by_source_take: ExposureRow[];
  };
  interleaving?: { epoch: number | null; by_folder: InterleavingRow[]; any_flagged: boolean };
  window_coverage?: { by_folder: WindowCoverageRow[]; least_uniform_files: WindowFileRow[]; any_flagged: boolean };
  dropout?: { by_folder: DropoutRow[]; any_flagged: boolean; note?: string };
  duplicates?: { pairs: DuplicateRow[]; any_flagged: boolean };
  provenance?: Record<string, any>;
  // Group-level stats
  group_stats?: Record<string, { mean: number; std: number; min: number; max: number; clip_count: number }>;
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
