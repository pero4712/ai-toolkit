import React, { useCallback, useEffect, useRef, useState } from 'react';
import useFilesList from '@/hooks/useFilesList';
import { Loader2, AlertCircle, Download, Box, Brain, Trash2, SlidersHorizontal, Info, Package } from 'lucide-react';
import { openMergeLoRAsModal } from './MergeLoRAsModal';
import { openSafetensorsInfoModal } from './SafetensorsInfoModal';
import { getFilename, getFoldername, encodeFilePathForUrl, exportFileName } from '@/utils/basic';
import { openConfirm } from './ConfirmModal';
import { apiClient } from '@/utils/api';

// cron/jobState.ts ExportStatus (subset the widget reads)
type ExportStatus = {
  id?: string;
  state: 'none' | 'requested' | 'zipping' | 'ready' | 'error' | 'cancelled';
  via_trainer?: boolean;
  step?: number;
  missing_inputs?: string[];
  error?: string;
};

const isExportPending = (s: ExportStatus | null) => s?.state === 'requested' || s?.state === 'zipping';

export default function FilesWidget({ jobID, jobName }: { jobID: string; jobName: string }) {
  const { files, status, refreshFiles } = useFilesList(jobID, 5000);
  const [exportStatus, setExportStatus] = useState<ExportStatus | null>(null);
  // last export whose outcome was already shown, so a reload doesn't re-alert
  const announcedRef = useRef<string | null>(null);

  const announce = useCallback(
    (s: ExportStatus) => {
      if (!s.id || announcedRef.current === s.id) return;
      announcedRef.current = s.id;
      if (s.state === 'ready') {
        refreshFiles();
        const missing = s.missing_inputs ?? [];
        if (missing.length > 0) {
          openConfirm({
            title: 'Some training inputs were not bundled',
            message: `These paths from the job config were not found and are not in the zip:\n${missing.join('\n')}`,
            type: 'warning',
            confirmText: 'OK',
            onConfirm: () => {},
          });
        }
      } else if (s.state === 'error') {
        openConfirm({
          title: 'Export failed',
          message: s.error ?? 'Could not build the state zip.',
          type: 'warning',
          confirmText: 'OK',
          onConfirm: () => {},
        });
      }
    },
    [refreshFiles],
  );

  const fetchExportStatus = useCallback(async () => {
    try {
      const res = await apiClient.get(`/api/jobs/${jobID}/export_state`);
      setExportStatus(res.data);
      return res.data as ExportStatus;
    } catch {
      return null;
    }
  }, [jobID]);

  // pick up an export already in flight (e.g. after a page reload) without
  // re-announcing one that finished earlier
  useEffect(() => {
    fetchExportStatus().then(s => {
      if (s && !isExportPending(s)) announcedRef.current = s.id ?? null;
    });
  }, [fetchExportStatus]);

  // poll while pending; the zip appears in the file list once it is ready
  useEffect(() => {
    if (!isExportPending(exportStatus)) return;
    const timer = setInterval(async () => {
      const s = await fetchExportStatus();
      if (s && !isExportPending(s)) announce(s);
    }, 2000);
    return () => clearInterval(timer);
  }, [exportStatus, fetchExportStatus, announce]);

  const handleExportState = async () => {
    if (exportStatus?.state === 'zipping') return;
    if (exportStatus?.state === 'requested') {
      openConfirm({
        title: 'Cancel export?',
        message: exportStatus.via_trainer
          ? 'The export is waiting for the trainer to finish its current step.'
          : 'The export is waiting to snapshot the job.',
        type: 'warning',
        confirmText: 'Cancel export',
        onConfirm: async () => {
          try {
            await apiClient.delete(`/api/jobs/${jobID}/export_state`);
          } catch (error) {
            console.error('Error cancelling export:', error);
          }
          fetchExportStatus();
        },
      });
      return;
    }
    try {
      const res = await apiClient.post(`/api/jobs/${jobID}/export_state`, {});
      setExportStatus(res.data);
    } catch (error: any) {
      console.error('Error exporting job state:', error);
      openConfirm({
        title: 'Export failed',
        message: error?.response?.data?.error ?? 'Could not request the state export.',
        type: 'warning',
        confirmText: 'OK',
        onConfirm: () => {},
      });
    }
  };

  const exportLabel = (() => {
    if (exportStatus?.state === 'requested') return exportStatus.via_trainer ? 'saving at step end…' : 'snapshotting…';
    if (exportStatus?.state === 'zipping') return 'zipping…';
    return 'export state';
  })();

  const isOptimizerFile = (filePath: string) => getFilename(filePath) === 'optimizer.pt';
  const isStateZip = (filePath: string) => /_state_step\d+\.zip$/.test(getFilename(filePath));
  const checkpointFiles = files.filter(file => !isOptimizerFile(file.path) && !isStateZip(file.path));
  const optimizerFile = files.find(file => isOptimizerFile(file.path));
  const stateZipFiles = files.filter(file => isStateZip(file.path));

  // step parsed from the trainer's checkpoint naming (<job>_000000500.safetensors,
  // or <job>_000000500_high_noise.safetensors for multi-expert models)
  const stepFromCheckpoint = (filePath: string): number | null => {
    const m = getFilename(filePath).match(/_(\d+)(?:_(?:high|low)_noise)?\.safetensors$/);
    return m ? parseInt(m[1], 10) : null;
  };

  // aligned download name (<job>_<kind>_step<N>.<ext>) passed to the files
  // route, which uses it for Content-Disposition; on-disk names are untouched
  const downloadUrl = (filePath: string, kind: string, step: number | null, ext: string) => {
    const base = `/api/files/${encodeFilePathForUrl(filePath)}`;
    if (step == null) return base;
    return `${base}?name=${encodeURIComponent(exportFileName(jobName, kind, step, ext))}`;
  };

  // optimizer.pt always corresponds to the most recent save
  const latestStep = checkpointFiles.reduce<number | null>((acc, f) => {
    const s = stepFromCheckpoint(f.path);
    return s != null && (acc == null || s > acc) ? s : acc;
  }, null);

  const cleanSize = (size: number) => {
    if (size < 1024) {
      return `${size} B`;
    } else if (size < 1024 * 1024) {
      return `${(size / 1024).toFixed(1)} KB`;
    } else if (size < 1024 * 1024 * 1024) {
      return `${(size / (1024 * 1024)).toFixed(1)} MB`;
    } else {
      return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    }
  };

  const handleDeleteFile = (filePath: string) => {
    const fileName = getFilename(filePath);
    openConfirm({
      title: 'Delete Checkpoint',
      message: `Are you sure you want to delete "${fileName}"? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .post('/api/files/delete', { filePath })
          .then(() => {
            refreshFiles();
          })
          .catch(error => {
            console.error('Error deleting checkpoint:', error);
          });
      },
    });
  };

  return (
    <div className="col-span-2 bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Brain className="w-5 h-5 text-purple-600 dark:text-purple-400" />
          <h2 className="font-semibold text-gray-100">Checkpoints</h2>
          <span className="px-2 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300">{checkpointFiles.length}</span>
        </div>
        <div className="flex items-center space-x-2">
          {checkpointFiles.length > 0 && (
            <span
              className="px-3 py-1 rounded-full text-sm bg-amber-500/10 text-amber-500 uppercase cursor-pointer hover:bg-amber-500/20 flex items-center gap-1"
              title={
                exportStatus?.state === 'requested'
                  ? 'Click to cancel'
                  : 'Build a zip with a checkpoint at the current step, optimizer state, config, loss history and datasets — importable on another machine to resume training. It appears in the list below when ready.'
              }
              onClick={handleExportState}
            >
              {isExportPending(exportStatus) && <Loader2 className="w-3 h-3 animate-spin" />}
              {exportLabel}
            </span>
          )}
          {checkpointFiles.length > 0 && (
            <span
              className="px-3 py-1 rounded-full text-sm bg-purple-500/10 text-purple-500 uppercase cursor-pointer hover:bg-purple-500/20"
              onClick={() => {
                const outputName = `${jobName}_merged`;
                openMergeLoRAsModal(
                  getFoldername(checkpointFiles[0].path),
                  outputName,
                  checkpointFiles.map(f => ({ path: f.path })),
                  () => {
                    refreshFiles();
                  },
                );
              }}
            >
              merge
            </span>
          )}
        </div>
      </div>

      <div className="p-2">
        {status === 'loading' && (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="w-5 h-5 text-gray-400 animate-spin" />
          </div>
        )}

        {status === 'error' && (
          <div className="flex items-center justify-center py-4 text-rose-400 space-x-2">
            <AlertCircle className="w-4 h-4" />
            <span className="text-sm">Error loading checkpoints</span>
          </div>
        )}

        {['success', 'refreshing'].includes(status) && (
          <div className="space-y-1">
            {checkpointFiles.map((file, index) => {
              const fileName = getFilename(file.path);
              const nameWithoutExt = fileName.replace('.safetensors', '');
              const ckptUrl = downloadUrl(file.path, 'checkpoint', stepFromCheckpoint(file.path), 'safetensors');
              return (
                <div
                  key={index}
                  className="group flex items-center justify-between px-2 py-1.5 rounded-lg hover:bg-gray-800 transition-all duration-200"
                >
                  <a
                    target="_blank"
                    href={ckptUrl}
                    className="flex items-center space-x-2 min-w-0 flex-1"
                  >
                    <Box className="w-4 h-4 text-purple-600 dark:text-purple-400 flex-shrink-0" />
                    <div className="flex flex-col min-w-0">
                      <div className="flex text-sm text-gray-200">
                        <span className="overflow-hidden text-ellipsis direction-rtl whitespace-nowrap">
                          {nameWithoutExt}
                        </span>
                      </div>
                      <span className="text-xs text-gray-500">.safetensors</span>
                    </div>
                  </a>
                  <div className="flex items-center space-x-3 flex-shrink-0">
                    <span className="text-xs text-gray-400">{cleanSize(file.size)}</span>
                    {fileName.endsWith('.safetensors') && (
                      <button
                        type="button"
                        onClick={() => openSafetensorsInfoModal(file.path)}
                        className="bg-sky-500 bg-opacity-0 group-hover:bg-opacity-10 hover:!bg-opacity-30 rounded-full p-1 transition-all"
                        title="Safetensors info"
                      >
                        <Info className="w-3 h-3 text-sky-400" />
                      </button>
                    )}
                    <a
                      target="_blank"
                      href={ckptUrl}
                      className="bg-purple-500 bg-opacity-0 group-hover:bg-opacity-10 rounded-full p-1 transition-all"
                    >
                      <Download className="w-3 h-3 text-purple-600 dark:text-purple-400" />
                    </a>
                    <button
                      type="button"
                      onClick={() => handleDeleteFile(file.path)}
                      className="bg-red-500 bg-opacity-0 group-hover:bg-opacity-10 hover:!bg-opacity-30 rounded-full p-1 transition-all"
                      title="Delete checkpoint"
                    >
                      <Trash2 className="w-3 h-3 text-red-500" />
                    </button>
                  </div>
                </div>
              );
            })}

            {optimizerFile && (
              <div className="group flex items-center justify-between px-2 py-1.5 rounded-lg border-t border-gray-800 mt-1 pt-2 hover:bg-gray-800 transition-all duration-200">
                <a
                  target="_blank"
                  href={downloadUrl(optimizerFile.path, 'optimizer', latestStep, 'pt')}
                  className="flex items-center space-x-2 min-w-0 flex-1"
                >
                  <SlidersHorizontal className="w-4 h-4 text-amber-500 flex-shrink-0" />
                  <div className="flex flex-col min-w-0">
                    <div className="flex text-sm text-amber-200">
                      <span className="overflow-hidden text-ellipsis direction-rtl whitespace-nowrap">optimizer</span>
                    </div>
                    <span className="text-xs text-amber-600/70">.pt · optimizer state</span>
                  </div>
                </a>
                <div className="flex items-center space-x-3 flex-shrink-0">
                  <span className="text-xs text-gray-400">{cleanSize(optimizerFile.size)}</span>
                  <a
                    target="_blank"
                    href={downloadUrl(optimizerFile.path, 'optimizer', latestStep, 'pt')}
                    className="bg-amber-500 bg-opacity-0 group-hover:bg-opacity-10 rounded-full p-1 transition-all"
                  >
                    <Download className="w-3 h-3 text-amber-500" />
                  </a>
                  <button
                    type="button"
                    onClick={() => handleDeleteFile(optimizerFile.path)}
                    className="bg-red-500 bg-opacity-0 group-hover:bg-opacity-10 hover:!bg-opacity-30 rounded-full p-1 transition-all"
                    title="Delete optimizer state"
                  >
                    <Trash2 className="w-3 h-3 text-red-500" />
                  </button>
                </div>
              </div>
            )}

            {stateZipFiles.map(file => {
              const zipUrl = `/api/files/${encodeFilePathForUrl(file.path)}`;
              return (
                <div
                  key={file.path}
                  className="group flex items-center justify-between px-2 py-1.5 rounded-lg border-t border-gray-800 mt-1 pt-2 hover:bg-gray-800 transition-all duration-200"
                >
                  <a target="_blank" href={zipUrl} className="flex items-center space-x-2 min-w-0 flex-1">
                    <Package className="w-4 h-4 text-amber-500 flex-shrink-0" />
                    <div className="flex flex-col min-w-0">
                      <div className="flex text-sm text-amber-200">
                        <span className="overflow-hidden text-ellipsis direction-rtl whitespace-nowrap">
                          {getFilename(file.path).replace(/\.zip$/, '')}
                        </span>
                      </div>
                      <span className="text-xs text-amber-600/70">.zip · job state for import</span>
                    </div>
                  </a>
                  <div className="flex items-center space-x-3 flex-shrink-0">
                    <span className="text-xs text-gray-400">{cleanSize(file.size)}</span>
                    <a
                      target="_blank"
                      href={zipUrl}
                      className="bg-amber-500 bg-opacity-0 group-hover:bg-opacity-10 rounded-full p-1 transition-all"
                    >
                      <Download className="w-3 h-3 text-amber-500" />
                    </a>
                    <button
                      type="button"
                      onClick={() => handleDeleteFile(file.path)}
                      className="bg-red-500 bg-opacity-0 group-hover:bg-opacity-10 hover:!bg-opacity-30 rounded-full p-1 transition-all"
                      title="Delete state zip"
                    >
                      <Trash2 className="w-3 h-3 text-red-500" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {['success', 'refreshing'].includes(status) && files.length === 0 && (
          <div className="text-center py-4 text-gray-400 text-sm">No checkpoints available</div>
        )}
      </div>
    </div>
  );
}
