import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import fsp from 'fs/promises';
import path from 'path';
import crypto from 'crypto';
import { getTrainingFolder } from '@/server/settings';
import { PAUSE_REQUEST_FILE, TRAINER_MARKER, exportDirOf, writeJsonAtomic } from '../../../../../../cron/jobState';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/**
 * Pause a running job: the trainer saves a checkpoint at the end of the
 * current step and then stops, freeing the GPU (toolkit/job_state_snapshot.py).
 * Start resumes from that step. Unlike /stop, the `stop` flag is not set: on
 * Windows the trainer's stop watcher would kill the run before it can save.
 */
export async function POST(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  if (job.status !== 'running' || job.stop) {
    return NextResponse.json({ error: 'Only a running job can be paused' }, { status: 409 });
  }

  const exportDir = exportDirOf(path.join(await getTrainingFolder(), job.name));
  let marker: { pause?: boolean } | null = null;
  try {
    marker = JSON.parse(await fsp.readFile(path.join(exportDir, TRAINER_MARKER), 'utf-8'));
  } catch {
    // run predates the marker
  }
  if (!marker?.pause) {
    return NextResponse.json(
      {
        error:
          'This run was started before pause was supported, so it would not save first. Stop it instead (progress since the last save is lost), or wait for the next regular save.',
        unsupported: true,
      },
      { status: 409 },
    );
  }

  await writeJsonAtomic(path.join(exportDir, PAUSE_REQUEST_FILE), { id: crypto.randomUUID() });
  await prisma.job.update({
    where: { id: jobID },
    data: { info: 'Pausing: saving at the end of this step...' },
  });
  return NextResponse.json({ ok: true }, { status: 202 });
}
