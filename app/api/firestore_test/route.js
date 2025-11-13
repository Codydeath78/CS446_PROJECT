export const runtime = 'nodejs';

import { NextResponse } from 'next/server';
import { Firestore, FieldValue } from '@google-cloud/firestore';

const GOOGLE_PROJECT_ID = process.env.GOOGLE_PROJECT_ID;
const firestore = GOOGLE_PROJECT_ID ? new Firestore({ projectId: GOOGLE_PROJECT_ID }) : null;

export async function POST() {
  try {
    if (!firestore) throw new Error('Firestore not initialized');
    const ref = await firestore.collection('safetyEvents').add({
      createdAt: FieldValue.serverTimestamp(),
      preview: 'manual test write',
      detector: { unsafe: true, scores: { neutral: 0, biased: 1 } },
      openaiModel: 'test',
    });
    return NextResponse.json({ ok: true, id: ref.id });
  } catch (e) {
    console.error(e);
    return NextResponse.json({ ok: false, error: String(e) }, { status: 500 });
  }
}