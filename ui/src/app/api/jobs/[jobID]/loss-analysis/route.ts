import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

export const runtime = 'nodejs';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const trainingFolder = await getTrainingFolder();
  const snapshotPath = path.join(trainingFolder, job.name, 'loss_analysis.json');

  if (!fs.existsSync(snapshotPath)) {
    return NextResponse.json({ available: false });
  }

  try {
    const raw = fs.readFileSync(snapshotPath, 'utf-8');
    const data = JSON.parse(raw);
    return NextResponse.json({ available: true, ...data });
  } catch {
    return NextResponse.json({ available: false });
  }
}
