// Job state export/import: archive layout, config path handling, snapshot of
// a stopped job and the zip builder. Pure (no db access) so both the worker
// (processExports) and the Next routes (import_state, export_state) use it.
//
// Export flow (see toolkit/job_state_snapshot.py for the trainer side):
//   export_state route   -> .export/status.json {state: 'requested'}
//                           (+ .export/request.json when the job is running)
//   trainer, step end    -> saves at that step, .export/snapshot/done.json
//   worker               -> snapshots a stopped job itself, then zips the
//                           snapshot + datasets to .export/<job>_state_step<N>.zip
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import readline from 'readline';
import archiver from 'archiver';

// ---------------------------------------------------------------- layout

export const EXPORT_DIR = '.export';
export const STATUS_FILE = 'status.json';
// shared with toolkit/job_state_snapshot.py
export const REQUEST_FILE = 'request.json';
export const SNAPSHOT_DIR = 'snapshot';
export const DONE_FILE = 'done.json';
export const ERROR_FILE = 'snapshot_error.json';
export const TRAINER_MARKER = 'trainer.json';

// archive folders for bundled training inputs
export const ARCHIVE_DATASETS_DIR = 'datasets';
export const ARCHIVE_SAMPLE_INPUTS_DIR = 'sample_inputs';

// written beside the training manifest by Video LoRA Studio
// (toolkit/training_manifest.py MANIFEST_FILENAME)
export const MANIFEST_FILENAME = 'manifest.json';

// caches and UI bookkeeping inside dataset folders; machine-local and
// regenerated on first use, so they are not exported
export const DATASET_SKIP_NAMES = new Set([
  '_latent_cache',
  '_t_e_cache',
  '_clip_vision_cache',
  '.thumbs',
  '.aitk_size.json',
]);

// dataset config keys holding a folder (or list of folders) on disk
// (toolkit/config_modules.py DatasetConfig)
const DATASET_PATH_KEYS = [
  'folder_path',
  'dataset_path',
  'control_path',
  'inpaint_path',
  'mask_path',
  'unconditional_path',
  'clip_image_path',
];

// Multi-expert models write one file per expert for a single save (Wan 2.2
// 14B: <base>_high_noise / <base>_low_noise). They belong to one checkpoint;
// exporting only one of them makes resume silently restart the other expert.
const EXPERT_SUFFIX = /_(high|low)_noise$/;

// state files a resume needs besides the checkpoint
const AUX_FILES = ['optimizer.pt', 'config.yaml', 'loss_analysis.json'];
const LOSS_DB = 'loss_log.db';
const LOSS_EVENTS = 'loss_events.jsonl';

export type ExportState = 'requested' | 'zipping' | 'ready' | 'error' | 'cancelled';

export type ExportStatus = {
  id: string;
  state: ExportState;
  requested_at: number;
  updated_at: number;
  // the job was running when requested, so the trainer takes the snapshot
  via_trainer?: boolean;
  step?: number;
  zip_path?: string;
  file_name?: string;
  size?: number;
  missing_inputs?: string[];
  error?: string;
};

export type PathMapEntry = {
  // the path string exactly as it appears in the exported job config
  original: string;
  // posix path inside the zip
  archive: string;
};

export type SnapshotDone = {
  id: string | null;
  step: number;
  files: string[];
  created_at: number;
};

export const exportDirOf = (jobFolder: string) => path.join(jobFolder, EXPORT_DIR);

// ---------------------------------------------------------------- json io

export async function readJson<T>(p: string): Promise<T | null> {
  try {
    return JSON.parse(await fsp.readFile(p, 'utf-8')) as T;
  } catch {
    return null;
  }
}

export async function writeJsonAtomic(p: string, data: unknown) {
  await fsp.mkdir(path.dirname(p), { recursive: true });
  const tmp = `${p}.${process.pid}.tmp`;
  await fsp.writeFile(tmp, JSON.stringify(data, null, 2));
  await fsp.rename(tmp, p);
}

export const readStatus = (jobFolder: string) => readJson<ExportStatus>(path.join(exportDirOf(jobFolder), STATUS_FILE));

export async function writeStatus(jobFolder: string, status: ExportStatus) {
  await writeJsonAtomic(path.join(exportDirOf(jobFolder), STATUS_FILE), { ...status, updated_at: Date.now() / 1000 });
}

// ---------------------------------------------------------------- config paths

const asPathList = (value: unknown): string[] => {
  if (typeof value === 'string') return value ? [value] : [];
  if (Array.isArray(value)) return value.filter((v): v is string => typeof v === 'string' && v !== '');
  return [];
};

const processesOf = (jobConfig: any): any[] => {
  const processes = jobConfig?.config?.process;
  return Array.isArray(processes) ? processes : [];
};

/** Every dataset folder and sample control image the job config points at. */
export function collectConfigPaths(jobConfig: any) {
  const datasetPaths = new Set<string>();
  const sampleInputPaths = new Set<string>();
  for (const proc of processesOf(jobConfig)) {
    for (const ds of Array.isArray(proc?.datasets) ? proc.datasets : []) {
      for (const key of DATASET_PATH_KEYS) {
        for (const p of asPathList(ds?.[key])) datasetPaths.add(p);
      }
    }
    const sample = proc?.sample;
    for (const p of asPathList(sample?.ctrl_img)) sampleInputPaths.add(p);
    for (const item of Array.isArray(sample?.samples) ? sample.samples : []) {
      for (const p of asPathList(item?.ctrl_img)) sampleInputPaths.add(p);
    }
  }
  return { datasetPaths: [...datasetPaths], sampleInputPaths: [...sampleInputPaths] };
}

/**
 * Replace the exported path strings with their restored locations, in place.
 * Only the keys collectConfigPaths reads are touched, so unrelated strings
 * that happen to match are left alone.
 */
export function rewriteConfigPaths(jobConfig: any, mapping: Map<string, string>) {
  const swap = (value: unknown): unknown => {
    if (typeof value === 'string') return mapping.get(value) ?? value;
    if (Array.isArray(value)) return value.map(v => (typeof v === 'string' ? mapping.get(v) ?? v : v));
    return value;
  };
  for (const proc of processesOf(jobConfig)) {
    for (const ds of Array.isArray(proc?.datasets) ? proc.datasets : []) {
      for (const key of DATASET_PATH_KEYS) {
        if (ds && key in ds) ds[key] = swap(ds[key]);
      }
    }
    const sample = proc?.sample;
    if (sample && 'ctrl_img' in sample) sample.ctrl_img = swap(sample.ctrl_img);
    for (const item of Array.isArray(sample?.samples) ? sample.samples : []) {
      if (item && 'ctrl_img' in item) item.ctrl_img = swap(item.ctrl_img);
    }
  }
}

// ---------------------------------------------------------------- fs helpers

const isInside = (child: string, parent: string) => {
  const rel = path.relative(parent, child);
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel));
};

const toPosix = (p: string) => p.split(path.sep).join('/');

async function isDirectory(p: string) {
  try {
    return (await fsp.stat(p)).isDirectory();
  } catch {
    return false;
  }
}

async function isFile(p: string) {
  try {
    return (await fsp.stat(p)).isFile();
  } catch {
    return false;
  }
}

// checkpoints are written once under a step-specific name and never modified,
// so a hard link is a free, stable snapshot; copy where links are unsupported
async function linkOrCopy(src: string, dst: string) {
  try {
    await fsp.link(src, dst);
  } catch {
    await fsp.copyFile(src, dst);
  }
}

// ---------------------------------------------------------------- stopped job snapshot

/**
 * Snapshot a job that is not running (nothing writes its folder), in the same
 * layout the trainer writes: the latest save, aux state, the loss db WITH its
 * WAL (a db without its -wal misses every row since the last checkpoint), and
 * loss events cut at the save's step.
 */
export async function snapshotStoppedJob(jobFolder: string, id: string, fallbackStep: number): Promise<SnapshotDone> {
  const entries = await fsp.readdir(jobFolder, { withFileTypes: true });
  const saves = new Map<string, { files: string[]; mtimeMs: number }>();
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith('.safetensors')) continue;
    const stat = await fsp.stat(path.join(jobFolder, entry.name));
    const base = entry.name.replace(/\.safetensors$/, '').replace(EXPERT_SUFFIX, '');
    const save = saves.get(base) ?? { files: [], mtimeMs: 0 };
    save.files.push(entry.name);
    save.mtimeMs = Math.max(save.mtimeMs, stat.mtimeMs);
    saves.set(base, save);
  }
  const latest = [...saves.entries()].sort((a, b) => b[1].mtimeMs - a[1].mtimeMs)[0];
  if (!latest) throw new Error('No checkpoints to export yet');
  // <job>_000001500 -> 1500; the final save has no step suffix
  const m = latest[0].match(/_(\d{9})$/);
  const step = m ? parseInt(m[1], 10) : fallbackStep;

  const exportDir = exportDirOf(jobFolder);
  const finalDir = path.join(exportDir, SNAPSHOT_DIR);
  const tmpDir = `${finalDir}.tmp`;
  await fsp.rm(tmpDir, { recursive: true, force: true });
  await fsp.mkdir(tmpDir, { recursive: true });

  const files: string[] = [];
  for (const name of latest[1].files.sort()) {
    await linkOrCopy(path.join(jobFolder, name), path.join(tmpDir, name));
    files.push(name);
  }
  for (const name of [...AUX_FILES, LOSS_DB, `${LOSS_DB}-wal`]) {
    if (await isFile(path.join(jobFolder, name))) {
      await fsp.copyFile(path.join(jobFolder, name), path.join(tmpDir, name));
      files.push(name);
    }
  }
  // rows past the save are pruned by the UILogger on resume; events are
  // append-only and never pruned, so cut them here
  const eventsPath = path.join(jobFolder, LOSS_EVENTS);
  if (await isFile(eventsPath)) {
    await trimLossEvents(eventsPath, path.join(tmpDir, LOSS_EVENTS), step);
    files.push(LOSS_EVENTS);
  }

  const done: SnapshotDone = { id, step, files, created_at: Date.now() / 1000 };
  await writeJsonAtomic(path.join(tmpDir, DONE_FILE), done);
  await fsp.rm(finalDir, { recursive: true, force: true });
  await fsp.rename(tmpDir, finalDir);
  return done;
}

async function trimLossEvents(src: string, dst: string, step: number) {
  const out = fs.createWriteStream(dst);
  const rl = readline.createInterface({ input: fs.createReadStream(src, 'utf-8'), crlfDelay: Infinity });
  for await (const line of rl) {
    try {
      const recStep = JSON.parse(line)?.step;
      if (Number.isInteger(recStep) && recStep <= step) out.write(`${line}\n`);
    } catch {
      // torn or malformed line
    }
  }
  await new Promise<void>((resolve, reject) => out.end((err?: Error | null) => (err ? reject(err) : resolve())));
}

// ---------------------------------------------------------------- zip

type ArchiveItem = { abs: string; name: string };

async function walkDataset(root: string, archiveRoot: string, out: ArchiveItem[]) {
  const entries = await fsp.readdir(root, { withFileTypes: true });
  for (const entry of entries) {
    if (DATASET_SKIP_NAMES.has(entry.name)) continue;
    const abs = path.join(root, entry.name);
    // symlinks are skipped: following them could pull in arbitrary trees
    if (entry.isDirectory()) {
      await walkDataset(abs, `${archiveRoot}/${entry.name}`, out);
    } else if (entry.isFile()) {
      out.push({ abs, name: `${archiveRoot}/${entry.name}` });
    }
  }
}

/**
 * Resolve the dataset folders and sample inputs referenced by the config into
 * archive entries plus the original-path -> archive-path map the importer uses
 * to rewrite the config.
 */
async function collectInputs(jobConfig: any) {
  const { datasetPaths, sampleInputPaths } = collectConfigPaths(jobConfig);
  const items: ArchiveItem[] = [];
  const pathMap: PathMapEntry[] = [];
  const missing: string[] = [];

  // resolve dataset paths; nested folders (e.g. a control_path inside the
  // folder_path) ride along with their enclosing root instead of duplicating
  const resolved = new Map<string, string>(); // raw config string -> abs dir
  for (const raw of datasetPaths) {
    const abs = path.resolve(raw);
    if (await isDirectory(abs)) resolved.set(raw, abs);
    else missing.push(raw);
  }
  const roots: string[] = [];
  for (const abs of [...new Set(resolved.values())].sort((a, b) => a.length - b.length)) {
    if (!roots.some(r => isInside(abs, r))) roots.push(abs);
  }

  // the manifest join keys files as "<folder basename>/<file>" relative to the
  // folder's parent, so each root keeps its basename under a per-parent group
  const parents = [...new Set(roots.map(r => path.dirname(r)))];
  const groupOf = new Map<string, string>();
  parents.forEach((p, i) => groupOf.set(p, parents.length > 1 ? `set${i + 1}` : ''));
  const rootArchive = new Map<string, string>();
  for (const root of roots) {
    const group = groupOf.get(path.dirname(root))!;
    const archiveRoot = [ARCHIVE_DATASETS_DIR, group, path.basename(root)].filter(Boolean).join('/');
    rootArchive.set(root, archiveRoot);
    await walkDataset(root, archiveRoot, items);
  }
  for (const parent of parents) {
    const manifest = path.join(parent, MANIFEST_FILENAME);
    if (await isFile(manifest)) {
      const group = groupOf.get(parent)!;
      items.push({ abs: manifest, name: [ARCHIVE_DATASETS_DIR, group, MANIFEST_FILENAME].filter(Boolean).join('/') });
    }
  }
  for (const [raw, abs] of resolved) {
    const root = roots.find(r => isInside(abs, r))!;
    const rel = toPosix(path.relative(root, abs));
    pathMap.push({ original: raw, archive: rel ? `${rootArchive.get(root)}/${rel}` : rootArchive.get(root)! });
  }

  // sample control images (i2v cannot sample without them); flat, index-prefixed
  const seenInputs = new Map<string, string>();
  for (const raw of sampleInputPaths) {
    const abs = path.resolve(raw);
    if (!(await isFile(abs))) {
      missing.push(raw);
      continue;
    }
    let name = seenInputs.get(abs);
    if (!name) {
      name = `${ARCHIVE_SAMPLE_INPUTS_DIR}/${seenInputs.size}_${path.basename(abs)}`;
      seenInputs.set(abs, name);
      items.push({ abs, name });
    }
    pathMap.push({ original: raw, archive: name });
  }

  return { items, pathMap, missing };
}

export const stateZipName = (jobName: string, step: number) => `${jobName}_state_step${step}.zip`;

export type JobInfo = {
  name: string;
  gpu_ids: string;
  job_type: string;
  job_ref: string | null;
  job_config: string;
};

/**
 * Zip a finished snapshot plus the job's datasets. Written to a .partial file
 * and renamed, so a listed zip is always complete.
 */
export async function buildStateZip(jobFolder: string, job: JobInfo, done: SnapshotDone) {
  const jobConfig = JSON.parse(job.job_config);
  const snapshotDir = path.join(exportDirOf(jobFolder), SNAPSHOT_DIR);
  const inputs = await collectInputs(jobConfig);

  const state = {
    format: 'aitk_job_state',
    version: 2,
    exported_at: new Date().toISOString(),
    name: job.name,
    gpu_ids: job.gpu_ids,
    job_type: job.job_type,
    job_ref: job.job_ref,
    step: done.step,
    job_config: jobConfig,
    files: done.files,
    // config path strings -> archive locations; the importer rewrites them
    path_map: inputs.pathMap,
    input_file_count: inputs.items.length,
  };

  const fileName = stateZipName(job.name, done.step);
  const outputPath = path.join(exportDirOf(jobFolder), fileName);
  const partialPath = `${outputPath}.partial`;
  await fsp.rm(partialPath, { force: true });

  await new Promise<void>((resolve, reject) => {
    const output = fs.createWriteStream(partialPath);
    // checkpoints and video are already dense; store instead of deflate keeps
    // export fast and barely larger (archiver switches to zip64 past 4GB)
    const archive = archiver('zip', { store: true });

    output.on('close', () => resolve());
    output.on('error', reject);
    archive.on('error', reject);

    archive.pipe(output);
    archive.append(JSON.stringify(state, null, 2), { name: 'job_state.json' });
    for (const name of done.files) {
      const abs = path.join(snapshotDir, name);
      if (fs.statSync(abs).isDirectory()) archive.directory(abs, name);
      else archive.file(abs, { name });
    }
    for (const item of inputs.items) {
      archive.file(item.abs, { name: item.name });
    }

    archive.finalize().catch(reject);
  });

  await fsp.rename(partialPath, outputPath);

  // one state zip per job: drop older exports (each is checkpoint + datasets)
  for (const entry of await fsp.readdir(exportDirOf(jobFolder))) {
    if (entry !== fileName && /_state_step\d+\.zip(\.partial)?$/.test(entry)) {
      await fsp.rm(path.join(exportDirOf(jobFolder), entry), { force: true });
    }
  }
  // the zip owns the state now; the snapshot's optimizer copy is ~0.6GB
  await fsp.rm(snapshotDir, { recursive: true, force: true });

  const { size } = await fsp.stat(outputPath);
  return { zipPath: outputPath, fileName, size, missing: inputs.missing };
}
