// app/api/chat/route.js  (Next.js app router style)
import { NextResponse } from "next/server";
import OpenAI from "openai";
import fetch from "node-fetch"; // node 18+ already has fetch, but keep for clarity
import { Firestore } from "@google-cloud/firestore";

require('dotenv').config();

const systemPrompt = 'Welcome to chatbot INC.! your go-to platform for real-time AI-powered conversations. Hello, how can I help you?';

// env vars
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const DETECTOR_URL = process.env.DETECTOR_URL; // e.g. https://<your-cloud-run>.run.app/detect
const GOOGLE_PROJECT_ID = process.env.GOOGLE_PROJECT_ID; // for Firestore logging

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// initialize Firestore (only if you're using it)
let firestore;
if (GOOGLE_PROJECT_ID) {
  firestore = new Firestore({ projectId: GOOGLE_PROJECT_ID });
}

export async function POST(req) {
  try {
    const data = await req.json(); // expect messages array from client
    const messages = Array.isArray(data) ? [{ role: "system", content: systemPrompt }, ...data] : [{ role: "system", content: systemPrompt }];

    // Request a streaming completion from OpenAI (gpt-4o stream)
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages,
      stream: true,
    });

    // Buffer all streamed tokens into `fullText`
    let fullText = '';
    for await (const chunk of completion) {
      // chunk.choices[0].delta.content may be undefined for some deltas
      const part = chunk.choices?.[0]?.delta?.content;
      if (part) fullText += part;
    }

    // Call detection service with fullText
    const detectorResp = await fetch(DETECTOR_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        response_text: fullText,
        // optional: send conversation context (anonymize user IDs first)
        // conversation: messages,
      }),
    });

    if (!detectorResp.ok) {
      // If detector failed, fallback to safe behavior: return original text
      console.error('Detector call failed', await detectorResp.text());
      return createStreamedResponse(fullText);
    }

    const det = await detectorResp.json();
    // expected det = { flagged: bool, action: 'pass'|'rephrase'|'block'|'human_review', confidence: 0.x, rephrase_text?: string, metadata?: {...} }

    // Log flag metadata to Firestore (if flagged)
    if (det.flagged && firestore) {
      try {
        const doc = {
          timestamp: new Date().toISOString(),
          flagged: det.flagged,
          action: det.action,
          confidence: det.confidence,
          // keep conversation anonymized
          preview: fullText.slice(0, 512),
          metadata: det.metadata || {},
        };
        await firestore.collection('bias_flags').add(doc);
      } catch (e) {
        console.error('Failed to log to Firestore', e);
      }
    }

    // Decide what to return
    if (det.action === 'rephrase' && det.rephrase_text) {
      return createStreamedResponse(det.rephrase_text);
    } else if (det.action === 'block') {
      const blockedMessage = "Sorry — this response was flagged as potentially inappropriate and cannot be shown.";
      return createStreamedResponse(blockedMessage);
    } else if (det.action === 'human_review') {
      const underReview = "This response is under review for safety. Please try again later or edit your question.";
      return createStreamedResponse(underReview);
    } else {
      // pass
      return createStreamedResponse(fullText);
    }

  } catch (err) {
    console.error(err);
    return NextResponse.json({ error: 'Server error' }, { status: 500 });
  }
}

// helper that streams a text as a ReadableStream (so client code expecting streaming will still work)
function createStreamedResponse(text) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(text));
      controller.close();
    },
  });
  return new NextResponse(stream);
}