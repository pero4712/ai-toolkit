import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import archiver from 'archiver';
import { getTrainingFolder } from '@/server/settings';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// Auxiliary state files carried along with the checkpoint so a run can be
// resumed on another machine with optimizer momentum and instrumentation
// history intact.
const AUX_FILES = ['optimizer.pt', 'config.yaml', 'loss_log.db', 'loss_events.jsonl', 'loss_analysis.json'];

type PostBody = {
  // include every checkpoint instead of only the latest one
  allCheckpoints?: boolean;
};

export async function POST(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  let body: PostBody = {};
  try {
    body = (await request.json()) as PostBody;
  } catch {
    // empty body is fine
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);

  let entries: fs.Dirent[];
  try {
    entries = await fsp.readdir(jobFolder, { withFileTypes: true });
  } catch {
    return NextResponse.json({ error: 'Job folder not found' }, { status: 404 });
  }

  // checkpoints live flat in the job folder; pick by modification time so the
  // exported optimizer.pt (always from the latest save) matches the checkpoint
  const checkpoints: { name: string; mtimeMs: number }[] = [];
  for (const entry of entries) {
    if (entry.isFile() && entry.name.endsWith('.safetensors')) {
      const stat = await fsp.stat(path.join(jobFolder, entry.name));
      checkpoints.push({ name: entry.name, mtimeMs: stat.mtimeMs });
    }
  }
  checkpoints.sort((a, b) => b.mtimeMs - a.mtimeMs);

  if (checkpoints.length === 0) {
    return NextResponse.json({ error: 'No checkpoints to export yet' }, { status: 404 });
  }

  const includedCheckpoints = body.allCheckpoints ? checkpoints : checkpoints.slice(0, 1);

  const includedAux: string[] = [];
  for (const aux of AUX_FILES) {
    try {
      const stat = await fsp.stat(path.join(jobFolder, aux));
      if (stat.isFile()) includedAux.push(aux);
    } catch {
      // not present, skip
    }
  }

  let jobConfig: any = null;
  try {
    jobConfig = JSON.parse(job.job_config);
  } catch {
    return NextResponse.json({ error: 'Job config is not valid JSON' }, { status: 500 });
  }

  const state = {
    format: 'aitk_job_state',
    version: 1,
    exported_at: new Date().toISOString(),
    name: job.name,
    gpu_ids: job.gpu_ids,
    job_type: job.job_type,
    job_ref: job.job_ref,
    step: job.step,
    job_config: jobConfig,
    files: [...includedCheckpoints.map(c => c.name), ...includedAux],
  };

  const exportDir = path.join(jobFolder, '.export');
  await fsp.mkdir(exportDir, { recursive: true });
  const fileName = `${job.name}_state_step${job.step}.zip`;
  const outputPath = path.join(exportDir, fileName);
  await fsp.rm(outputPath, { force: true });

  await new Promise<void>((resolve, reject) => {
    const output = fs.createWriteStream(outputPath);
    // checkpoints are already-compressed tensors; store instead of deflate
    // keeps export fast and barely larger
    const archive = archiver('zip', { zlib: { level: 1 } });

    output.on('close', () => resolve());
    output.on('error', reject);
    archive.on('error', reject);

    archive.pipe(output);
    archive.append(JSON.stringify(state, null, 2), { name: 'job_state.json' });
    for (const c of includedCheckpoints) {
      archive.file(path.join(jobFolder, c.name), { name: c.name });
    }
    for (const aux of includedAux) {
      archive.file(path.join(jobFolder, aux), { name: aux });
    }

    archive.finalize().catch(reject);
  });

  return NextResponse.json({
    ok: true,
    zipPath: outputPath,
    fileName,
    checkpoints: includedCheckpoints.map(c => c.name),
    aux: includedAux,
  });
}
