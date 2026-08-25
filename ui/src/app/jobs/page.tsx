'use client';

import { useRef, useState } from 'react';
import JobsTable from '@/components/JobsTable';
import { TopBar, MainContent } from '@/components/layout';
import Link from 'next/link';
import { Loader2, Upload } from 'lucide-react';
import { apiClient } from '@/utils/api';
import { openConfirm } from '@/components/ConfirmModal';
import { useRouter } from 'next/navigation';

export default function Dashboard() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [importing, setImporting] = useState(false);
  const router = useRouter();

  const uploadState = async (file: File, overwrite: boolean) => {
    const formData = new FormData();
    formData.append('file', file);
    if (overwrite) formData.append('overwrite', 'true');
    return apiClient.post('/api/jobs/import_state', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  };

  const handleImportFile = async (file: File) => {
    setImporting(true);
    try {
      const res = await uploadState(file, false);
      router.push(`/jobs/${res.data.jobID}`);
    } catch (error: any) {
      if (error?.response?.status === 409) {
        openConfirm({
          title: 'Job already exists',
          message: `${error.response.data.error} Overwrite replaces the job's config and copies the zip's files over the existing job folder.`,
          type: 'warning',
          confirmText: 'Overwrite',
          onConfirm: () => {
            setImporting(true);
            uploadState(file, true)
              .then(res => router.push(`/jobs/${res.data.jobID}`))
              .catch(e => {
                console.error('Error importing job state:', e);
              })
              .finally(() => setImporting(false));
          },
        });
      } else {
        console.error('Error importing job state:', error);
        openConfirm({
          title: 'Import failed',
          message: error?.response?.data?.error ?? 'Could not import the job state zip.',
          type: 'warning',
          confirmText: 'OK',
          onConfirm: () => {},
        });
      }
    } finally {
      setImporting(false);
    }
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">Queue</h1>
        </div>
        <div className="flex-1"></div>
        <div className="flex items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept=".zip"
            className="hidden"
            onChange={e => {
              const file = e.target.files?.[0];
              // reset so picking the same file again re-triggers onChange
              e.target.value = '';
              if (file) handleImportFile(file);
            }}
          />
          <button
            type="button"
            disabled={importing}
            onClick={() => fileInputRef.current?.click()}
            className="text-gray-200 bg-gray-700 hover:bg-gray-600 px-2 sm:px-3 py-1 rounded-md text-sm sm:text-base whitespace-nowrap flex items-center gap-1 disabled:opacity-50"
            title="Restore a job from a state zip exported on another machine (checkpoint, optimizer state, config, loss history)"
          >
            {importing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            <span className="hidden sm:inline">{importing ? 'Importing…' : 'Import Job'}</span>
          </button>
          <Link
            href="/jobs/new"
            className="text-white bg-slate-600 px-2 sm:px-3 py-1 rounded-md text-sm sm:text-base whitespace-nowrap"
          >
            <span className="sm:hidden">+ New Job</span>
            <span className="hidden sm:inline">New Training Job</span>
          </Link>
        </div>
      </TopBar>
      <MainContent>
        <JobsTable />
      </MainContent>
    </>
  );
}
