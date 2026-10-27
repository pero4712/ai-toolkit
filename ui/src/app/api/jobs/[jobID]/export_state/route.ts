import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import fsp from 'fs/promises';
import path from 'path';
import crypto from 'crypto';
import { getTrainingFolder } from '@/server/settings';
import {
  ExportStatus,
  REQUEST_FILE,
  TRAINER_MARKER,
  exportDirOf,
  readStatus,
  writeJsonAtomic,
  writeStatus,
} from '../../../../../../cron/jobState';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// the job's folder is written by its trainer in these states, so the trainer
// takes the snapshot (cron/jobState.ts describes the flow)
const RUNNING_STATES = ['running', 'stopping'];

const isPending = (s: ExportStatus | null) => s?.state === 'requested' || s?.state === 'zipping';

async function jobFolderFor(jobID: string) {
  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return null;
  return { job, jobFolder: path.join(await getTrainingFolder(), job.name) };
}

/** Current export status of the job ({state: 'none'} if never exported). */
export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const found = await jobFolderFor(jobID);
  if (!found) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  return NextResponse.json((await readStatus(found.jobFolder)) ?? { state: 'none' });
}

/**
 * Request a state export. Returns at once; the worker builds the zip
 * (via the trainer's snapshot if the job is running, saving at its current
 * step) and the zip then shows up in the job's file list.
 */
export async function POST(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const found = await jobFolderFor(jobID);
  if (!found) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  const { job, jobFolder } = found;

  const current = await readStatus(jobFolder);
  if (isPending(current)) return NextResponse.json(current);

  const viaTrainer = RUNNING_STATES.includes(job.status);
  const exportDir = exportDirOf(jobFolder);
  if (viaTrainer) {
    try {
      await fsp.access(path.join(exportDir, TRAINER_MARKER));
    } catch {
      return NextResponse.json(
        {
          error:
            'This run was started before live state export was supported. Stop the job to export its latest checkpoint, or restart it to export while training.',
        },
        { status: 409 },
      );
    }
  } else {
    let hasCheckpoint = false;
    try {
      hasCheckpoint = (await fsp.readdir(jobFolder)).some(f => f.endsWith('.safetensors'));
    } catch {
      // no folder yet
    }
    if (!hasCheckpoint) return NextResponse.json({ error: 'No checkpoints to export yet' }, { status: 404 });
  }

  const now = Date.now() / 1000;
  const status: ExportStatus = {
    id: crypto.randomUUID(),
    state: 'requested',
    requested_at: now,
    updated_at: now,
    via_trainer: viaTrainer,
  };
  // request first: once status says requested, the worker may look for it
  if (viaTrainer) await writeJsonAtomic(path.join(exportDir, REQUEST_FILE), { id: status.id });
  await writeStatus(jobFolder, status);
  return NextResponse.json(status, { status: 202 });
}

/** Cancel an export still waiting for its snapshot. */
export async function DELETE(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const found = await jobFolderFor(jobID);
  if (!found) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const current = await readStatus(found.jobFolder);
  if (current?.state !== 'requested') {
    return NextResponse.json({ error: 'Only an export still waiting for its snapshot can be cancelled' }, { status: 409 });
  }
  await fsp.rm(path.join(exportDirOf(found.jobFolder), REQUEST_FILE), { force: true });
  await writeStatus(found.jobFolder, { ...current, state: 'cancelled' });
  return NextResponse.json({ ok: true });
}
