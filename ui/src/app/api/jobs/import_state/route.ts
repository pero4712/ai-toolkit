import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import unzipper from 'unzipper';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';
import { isMac } from '@/helpers/basic';
import { ARCHIVE_DATASETS_DIR, ARCHIVE_SAMPLE_INPUTS_DIR, PathMapEntry, rewriteConfigPaths } from '../../../../../cron/jobState';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// dropped into a restored dataset folder so a later overwrite import of the
// same job may reuse it, while unrelated folders of the same name are never
// written into
const IMPORT_MARKER = '.aitk_import.json';

/** Join archive path segments under root, refusing anything that escapes it. */
function safeJoin(root: string, archivePath: string): string | null {
  const parts = archivePath.split('/').filter(p => p !== '' && p !== '.');
  if (parts.some(p => p === '..')) return null;
  const dest = path.resolve(root, ...parts);
  const rel = path.relative(root, dest);
  if (rel.startsWith('..') || path.isAbsolute(rel)) return null;
  return dest;
}

/** Pick the folder restored datasets go to: <datasets root>/<job name>[_N]. */
async function datasetTarget(datasetsRoot: string, name: string, overwrite: boolean) {
  for (let i = 1; ; i++) {
    const candidate = path.join(datasetsRoot, i === 1 ? name : `${name}_${i}`);
    let marker: any = null;
    try {
      marker = JSON.parse(await fsp.readFile(path.join(candidate, IMPORT_MARKER), 'utf-8'));
    } catch {
      // no marker (or unreadable): only usable if the folder does not exist
      try {
        await fsp.access(candidate);
        continue;
      } catch {
        return candidate;
      }
    }
    if (overwrite && marker?.job === name) return candidate;
  }
}

/**
 * Restore a training job from a state zip produced by export_state.
 *
 * The zip is the raw request body (streamed to disk, never buffered whole, so
 * multi-GB bundles with datasets work). Query params: overwrite=true replaces
 * an existing job of the same name; filename is the uploaded file's name.
 *
 * Creates (or updates) the Job row from the bundled job_state.json, unpacks
 * checkpoint / optimizer / instrumentation files into the job's folder and
 * bundled datasets under the datasets root, and points the config's dataset
 * and sample-input paths at the restored copies. Starting the job afterwards
 * auto-resumes from the extracted checkpoint via the trainer's resume path.
 */
export async function POST(request: NextRequest) {
  const overwrite = request.nextUrl.searchParams.get('overwrite') === 'true';
  const uploadName = request.nextUrl.searchParams.get('filename') || 'state zip';

  if (!request.body) {
    return NextResponse.json({ error: 'No file provided' }, { status: 400 });
  }

  const trainingFolder = await getTrainingFolder();
  const importDir = path.join(trainingFolder, '.imports');
  await fsp.mkdir(importDir, { recursive: true });
  const tmpZipPath = path.join(importDir, `import_${Date.now()}.zip`);

  try {
    await pipeline(Readable.fromWeb(request.body as any), fs.createWriteStream(tmpZipPath));

    const directory = await unzipper.Open.file(tmpZipPath);
    const stateEntry = directory.files.find(f => f.path === 'job_state.json');
    if (!stateEntry) {
      return NextResponse.json({ error: 'Not a job state zip (job_state.json missing)' }, { status: 400 });
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

    const hasDatasets = directory.files.some(f => f.path.startsWith(`${ARCHIVE_DATASETS_DIR}/`));
    const datasetsRoot = await getDatasetsRoot();
    const datasetDir = hasDatasets ? await datasetTarget(datasetsRoot, name, overwrite) : null;
    if (datasetDir) {
      await fsp.mkdir(datasetDir, { recursive: true });
      await fsp.writeFile(
        path.join(datasetDir, IMPORT_MARKER),
        JSON.stringify({ job: name, source: uploadName, imported_at: new Date().toISOString() }, null, 2),
      );
    }

    // archive path -> restored absolute path
    const restoreLocation = (archivePath: string): string | null => {
      if (archivePath.startsWith(`${ARCHIVE_DATASETS_DIR}/`) || archivePath === ARCHIVE_DATASETS_DIR) {
        if (!datasetDir) return null;
        return safeJoin(datasetDir, archivePath.slice(ARCHIVE_DATASETS_DIR.length));
      }
      if (archivePath.startsWith(`${ARCHIVE_SAMPLE_INPUTS_DIR}/`)) {
        return safeJoin(path.join(jobFolder, ARCHIVE_SAMPLE_INPUTS_DIR), path.basename(archivePath));
      }
      // checkpoints and aux files are stored flat; basename neutralizes traversal
      return path.join(jobFolder, path.basename(archivePath));
    };

    const extracted: string[] = [];
    let inputFiles = 0;
    for (const entry of directory.files) {
      if (entry.path === 'job_state.json' || entry.type !== 'File') continue;
      const destPath = restoreLocation(entry.path);
      if (!destPath) continue;
      await fsp.mkdir(path.dirname(destPath), { recursive: true });
      await pipeline(entry.stream(), fs.createWriteStream(destPath));
      if (path.dirname(destPath) === jobFolder) extracted.push(path.basename(destPath));
      else inputFiles++;
    }

    // point the config at the restored datasets / sample inputs
    const pathMap: PathMapEntry[] = Array.isArray(state.path_map) ? state.path_map : [];
    const mapping = new Map<string, string>();
    for (const { original, archive } of pathMap) {
      const restored = typeof archive === 'string' ? restoreLocation(archive) : null;
      if (typeof original === 'string' && restored) mapping.set(original, restored);
    }
    rewriteConfigPaths(state.job_config, mapping);

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
          info: `Restored from ${uploadName}`,
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
          info: `Restored from ${uploadName}`,
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
      inputFiles,
      datasetDir,
      remappedPaths: mapping.size,
      step: state.step ?? 0,
    });
  } catch (error: any) {
    console.error('Job state import failed:', error);
    return NextResponse.json({ error: error?.message ?? 'Import failed' }, { status: 500 });
  } finally {
    await fsp.rm(tmpZipPath, { force: true });
  }
}
