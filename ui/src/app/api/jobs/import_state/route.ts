import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import unzipper from 'unzipper';
import { getTrainingFolder } from '@/server/settings';
import { isMac } from '@/helpers/basic';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/**
 * Restore a training job from a state zip produced by export_state.
 *
 * Creates (or, with overwrite=true, updates) the Job row from the bundled
 * job_state.json and unpacks the checkpoint / optimizer / instrumentation
 * files into the job's folder. Starting the job afterwards auto-resumes from
 * the extracted checkpoint via the trainer's normal resume path.
 */
export async function POST(request: NextRequest) {
  const formData = await request.formData();
  const file = formData.get('file') as File | null;
  const overwrite = formData.get('overwrite') === 'true';

  if (!file) {
    return NextResponse.json({ error: 'No file provided' }, { status: 400 });
  }

  const trainingFolder = await getTrainingFolder();
  const importDir = path.join(trainingFolder, '.imports');
  await fsp.mkdir(importDir, { recursive: true });
  const tmpZipPath = path.join(importDir, `import_${Date.now()}.zip`);

  try {
    // buffer the upload to disk, then stream-extract from there
    const bytes = await file.arrayBuffer();
    await fsp.writeFile(tmpZipPath, Buffer.from(bytes));

    const directory = await unzipper.Open.file(tmpZipPath);
    const stateEntry = directory.files.find(f => f.path === 'job_state.json');
    if (!stateEntry) {
      return NextResponse.json(
        { error: 'Not a job state zip (job_state.json missing)' },
        { status: 400 },
      );
    }

    let state: any;
    try {
      state = JSON.parse((await stateEntry.buffer()).toString('utf-8'));
    } catch {
      return NextResponse.json({ error: 'job_state.json is not valid JSON' }, { status: 400 });
    }
    if (state.format !== 'aitk_job_state' || !state.name || !state.job_config) {
      return NextResponse.json({ error: 'Unrecognized job state format' }, { status: 400 });
    }

    // the save folder is derived from the config name inside the trainer, so
    // the job keeps its exported name; renaming would break resume pathing
    const name: string = state.name;
    const existing = await prisma.job.findFirst({ where: { name } });
    if (existing && !overwrite) {
      return NextResponse.json(
        { error: `A job named "${name}" already exists. Re-import with overwrite to replace it.`, conflict: true },
        { status: 409 },
      );
    }

    const jobFolder = path.join(trainingFolder, name);
    await fsp.mkdir(jobFolder, { recursive: true });

    const extracted: string[] = [];
    for (const entry of directory.files) {
      if (entry.path === 'job_state.json' || entry.type !== 'File') continue;
      // entries are stored flat; basename also neutralizes any traversal
      const safeName = path.basename(entry.path);
      const destPath = path.join(jobFolder, safeName);
      await new Promise<void>((resolve, reject) => {
        entry
          .stream()
          .pipe(fs.createWriteStream(destPath))
          .on('finish', () => resolve())
          .on('error', reject);
      });
      extracted.push(safeName);
    }

    let gpu_ids: string = state.gpu_ids ?? '0';
    if (isMac()) gpu_ids = 'mps';

    let job;
    if (existing) {
      job = await prisma.job.update({
        where: { id: existing.id },
        data: {
          gpu_ids,
          job_config: JSON.stringify(state.job_config),
          job_type: state.job_type ?? 'train',
          job_ref: state.job_ref ?? null,
          status: 'stopped',
          stop: false,
          info: `Restored from ${file.name}`,
          step: state.step ?? 0,
        },
      });
    } else {
      const highestQueuePosition = await prisma.job.aggregate({ _max: { queue_position: true } });
      job = await prisma.job.create({
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(state.job_config),
          job_type: state.job_type ?? 'train',
          job_ref: state.job_ref ?? null,
          status: 'stopped',
          info: `Restored from ${file.name}`,
          step: state.step ?? 0,
          queue_position: (highestQueuePosition._max.queue_position || 0) + 1000,
        },
      });
    }

    return NextResponse.json({
      ok: true,
      jobID: job.id,
      name,
      extracted,
      step: state.step ?? 0,
    });
  } catch (error: any) {
    console.error('Job state import failed:', error);
    return NextResponse.json({ error: error?.message ?? 'Import failed' }, { status: 500 });
  } finally {
    await fsp.rm(tmpZipPath, { force: true });
  }
}
