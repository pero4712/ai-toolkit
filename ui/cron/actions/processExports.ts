import path from 'path';
import fsp from 'fs/promises';
import prisma from '../prisma';
import { getTrainingFolder } from '../paths';
import {
  DONE_FILE,
  ERROR_FILE,
  ExportStatus,
  REQUEST_FILE,
  SNAPSHOT_DIR,
  SnapshotDone,
  buildStateZip,
  exportDirOf,
  readJson,
  readStatus,
  snapshotStoppedJob,
  writeJsonAtomic,
  writeStatus,
} from '../jobState';

// a running trainer answers at its next step boundary; a step that also
// samples can take a long while, so only give up after this
const TRAINER_TIMEOUT_S = 60 * 60;
const RUNNING_STATES = ['running', 'stopping'];

// jobs whose export is being advanced by this process (zips take minutes)
const active = new Set<string>();

/**
 * Advance pending state exports (see cron/jobState.ts for the flow). Called
 * every worker tick; the slow parts run off the loop so the queue keeps moving.
 */
export default async function processExports() {
  const trainingFolder = await getTrainingFolder();
  const jobs = await prisma.job.findMany({ select: { id: true, name: true, status: true, step: true } });
  for (const job of jobs) {
    if (active.has(job.id)) continue;
    const jobFolder = path.join(trainingFolder, job.name);
    const status = await readStatus(jobFolder);
    if (!status || (status.state !== 'requested' && status.state !== 'zipping')) continue;

    active.add(job.id);
    advance(jobFolder, job, status)
      .catch(async error => {
        console.error(`State export for ${job.name} failed:`, error);
        if (await isCurrent(jobFolder, status.id)) {
          await writeStatus(jobFolder, { ...status, state: 'error', error: error?.message ?? String(error) });
        }
      })
      .finally(() => active.delete(job.id));
  }
}

// the route may have cancelled or replaced the request since we read it
async function isCurrent(jobFolder: string, id: string) {
  const now = await readStatus(jobFolder);
  return now?.id === id && (now.state === 'requested' || now.state === 'zipping');
}

async function advance(
  jobFolder: string,
  job: { id: string; name: string; status: string; step: number },
  status: ExportStatus,
) {
  const exportDir = exportDirOf(jobFolder);
  const requestPath = path.join(exportDir, REQUEST_FILE);
  const readDone = async () => {
    const done = await readJson<SnapshotDone>(path.join(exportDir, SNAPSHOT_DIR, DONE_FILE));
    return done?.id === status.id ? done : null;
  };

  let done: SnapshotDone | null = null;
  if (status.state === 'requested') {
    const running = RUNNING_STATES.includes(job.status);
    if (running && !status.via_trainer) {
      // started between request and now: only the trainer may snapshot it
      await writeJsonAtomic(requestPath, { id: status.id });
      await writeStatus(jobFolder, { ...status, via_trainer: true });
      return;
    }
    if (running) {
      const err = await readJson<{ id: string; error: string }>(path.join(exportDir, ERROR_FILE));
      if (err?.id === status.id) throw new Error(`The trainer could not snapshot the job: ${err.error}`);
      done = await readDone();
      if (!done) {
        if (Date.now() / 1000 - status.requested_at > TRAINER_TIMEOUT_S) {
          await fsp.rm(requestPath, { force: true });
          throw new Error('The trainer did not answer the export request. Stop the job to export its latest checkpoint.');
        }
        return; // keep waiting for the next step boundary
      }
    } else {
      // stopped (or the trainer exited before answering): nothing writes the
      // folder now, so snapshot it here unless the trainer got to it first
      await fsp.rm(requestPath, { force: true });
      done = (status.via_trainer ? await readDone() : null) ?? (await snapshotStoppedJob(jobFolder, status.id, job.step));
    }
    if (!(await isCurrent(jobFolder, status.id))) return;
    status = { ...status, state: 'zipping', step: done.step };
    await writeStatus(jobFolder, status);
  } else {
    // 'zipping' left over from a worker restart; the snapshot is still there
    done = await readDone();
    if (!done) throw new Error('The export was interrupted. Request it again.');
  }

  const full = await prisma.job.findUnique({ where: { id: job.id } });
  if (!full) throw new Error('Job no longer exists');
  const zip = await buildStateZip(jobFolder, full, done);
  if (!(await isCurrent(jobFolder, status.id))) return;
  await writeStatus(jobFolder, {
    ...status,
    state: 'ready',
    zip_path: zip.zipPath,
    file_name: zip.fileName,
    size: zip.size,
    missing_inputs: zip.missing,
  });
}
