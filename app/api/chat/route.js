export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

import { NextResponse } from "next/server";
import OpenAI from "openai";
import { Firestore, FieldValue } from "@google-cloud/firestore";

const OPENAI_API_KEY    = process.env.OPENAI_API_KEY || "";
const SECRET_STRING     = (process.env.SECRET_STRING || "").trim();
const DETECTOR_URL_ENV  = (process.env.DETECTOR_URL || "").trim();
const DETECTOR_URL      = DETECTOR_URL_ENV.endsWith("/detect")
  ? DETECTOR_URL_ENV
  : `${DETECTOR_URL_ENV.replace(/\/$/, "")}/detect`;

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

let firestore = null;
try {
  firestore = new Firestore();
  console.log("[Firestore] Initialized with ADC");
} catch (e) {
  console.error("[Firestore] init failed:", e);
}

const systemPrompt =
  "Welcome to chatbot INC.! your go-to platform for real-time AI-powered conversations. Hello, how can I help you?";

async function runDetector(text) {
  if (!DETECTOR_URL || !SECRET_STRING || !text?.trim()) return null;
  try {
    const r = await fetch(DETECTOR_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": SECRET_STRING,
      },
      body: JSON.stringify({ response_text: text }),
    });
    if (!r.ok) {
      console.warn("[Detector] Non-OK:", r.status, await r.text());
      return null;
    }
    return await r.json(); // { bias_detected, confidence_scores, temperature, threshold, ... }
  } catch (e) {
    console.warn("[Detector] failed:", e?.message || e);
    return null;
  }
}

async function rephraseUnsafeText(text) {
  const res = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    temperature: 0.3,
    messages: [
      {
        role: "system",
        content:
          "Rewrite the user's text to remove biased, discriminatory, or harmful phrasing while preserving intent and meaning. Keep it concise and neutral."
      },
      { role: "user", content: `Rewrite safely:\n\n${text}` }
    ]
  });
  return res.choices?.[0]?.message?.content ?? text;
}

async function logSafety({ preview, scores, source, rephrased }) {
  if (!firestore) return;
  try {
    await firestore.collection("safetyEvents").add({
      createdAt: FieldValue.serverTimestamp(),
      source: source || "assistant",                 // "user" | "assistant"
      preview: (preview || "").slice(0, 2000),
      rephrased: (rephrased || "").slice(0, 2000),
      detector: { unsafe: true, scores: scores || {} },
      openaiModel: "gpt-4o",
    });
    console.log("[Firestore] safetyEvents write OK");
  } catch (e) {
    console.error("[Firestore] safetyEvents write FAILED:", e?.message || e);
  }
}

function streamText(text, meta = {}) {
  const enc = new TextEncoder();
  const stream = new ReadableStream({
    start(c) { c.enqueue(enc.encode(text)); c.close(); }
  });

  const headers = {
    "Content-Type": "text/plain; charset=utf-8",
    "Cache-Control": "no-store",
  };


  if (meta.rephrased) headers["x-safety"] = meta.source || "assistant"; // "user"|"assistant"
  if (meta.scores)    headers["x-safety-scores"] = JSON.stringify(meta.scores);
  if (meta.reason)    headers["x-safety-reason"] = meta.reason; // short string


  //show the original text before rephrase (trim and sanitize)
  if (meta.original) {
    const original = String(meta.original).slice(0, 1200); // cap for safety/UI
    headers["x-safety-original"] = original.replace(/\r?\n/g, "\\n");
  }


  //pass through any additional x-* headers you included in meta
  for (const [k, v] of Object.entries(meta)) {
    if (k.toLowerCase().startsWith("x-")) headers[k] = v;
  }

  return new NextResponse(stream, { headers });
}

export async function POST(req) {
  try {
    const body = await req.json();
    const incoming = Array.isArray(body) ? body : body?.messages || [];
    const messages = [{ role: "system", content: systemPrompt }, ...incoming];

    // 1) Check the latest USER message
    const lastUser = [...incoming].reverse().find(m => m?.role === "user")?.content || "";
    const detUser = await runDetector(lastUser);
    // user message flagged
    if (detUser?.bias_detected) {
      const safeUser = await rephraseUnsafeText(lastUser);
      // log asynchronously; do not await to keep the UX snappy
      logSafety({
        source: "user",
        preview: lastUser,
        rephrased: safeUser,
        scores: detUser.confidence_scores
      });
      // Show the rephrased input back to the user as the assistant’s visible reply
      //return streamText(safeUser);
      return streamText(safeUser, {
        rephrased: true,
        source: "user",
        scores: detUser.confidence_scores,
        reason: "User message rephrased for safety",
        original: lastUser, //include original user text for UI display
        "x-moderation-action": "rephrased-user",
        "x-moderation-reason": "bias_detected",
      });
    }

    // 2) Generate assistant draft
    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages,
      stream: true,
    });

    let fullText = "";
    for await (const chunk of completion) {
      const part = chunk.choices?.[0]?.delta?.content;
      if (part) fullText += part;
    }

    // 3) Check assistant draft
    const detAsst = await runDetector(fullText);
    // assistant draft flagged
    if (detAsst?.bias_detected) {
      const safeReply = await rephraseUnsafeText(fullText);
      logSafety({
        source: "assistant",
        preview: fullText,
        rephrased: safeReply,
        scores: detAsst.confidence_scores
      });
      //return streamText(safeReply);
      return streamText(safeReply, {
        rephrased: true,
        source: "assistant",
        scores: detAsst.confidence_scores,
        reason: "Assistant reply rephrased for safety",
        original: fullText, //include original assistant text for UI display
        "x-moderation-action": "rephrased-assistant",
        "x-moderation-reason": "bias_detected",
      });
    }

    // 4) Safe reply
    return streamText(fullText);
  } catch (err) {
    console.error(err);
    return NextResponse.json({ error: "Server error" }, { status: 500 });
  }
}