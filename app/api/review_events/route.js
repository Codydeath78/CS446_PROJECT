export const runtime = "nodejs";
export const dynamic = "force-dynamic";

import { NextResponse } from "next/server";
import { Firestore, FieldValue } from "@google-cloud/firestore";

//initialize Firestore
let firestore = null;
//firestore initialization
try {
  firestore = new Firestore();
  console.log("[review-events] Firestore init OK");
} catch (e) {
  console.error("[review-events] Firestore init failed:", e);
}

// GET /api/review_events
//returns a page of events where humanLabel.isUnsafe == null
export async function GET(req) {
  if (!firestore) {
    return NextResponse.json({ error: "Firestore not available" }, { status: 500 });
  }

  //parse query params
  try {
    const { searchParams } = new URL(req.url);

    const page = Math.max(parseInt(searchParams.get("page") || "1", 10), 1);
    const pageSize = Math.min(
      Math.max(parseInt(searchParams.get("pageSize") || "20", 10), 1),
      100
    );
    //determine sorting
    const sortParam = (searchParams.get("sort") || "newest").toLowerCase();
    const source = (searchParams.get("source") || "all").toLowerCase(); // "all" | "user" | "assistant"

    //map sortParam to field and direction
    let sortField = "createdAt";
    let sortDir = "desc";

    if (sortParam === "oldest") {
      sortField = "createdAt";
      sortDir = "asc";
    } else if (sortParam === "confidence_high") {
      sortField = "detector.scores.biased";
      sortDir = "desc";
    } else if (sortParam === "confidence_low") {
      sortField = "detector.scores.biased";
      sortDir = "asc";
    }

    //build query
    let q = firestore
      .collection("safetyEventsPending") //pending collection
      .where("humanLabel.isUnsafe", "==", null); //only unreviewed

    if (source === "user" || source === "assistant") {
      q = q.where("source", "==", source);
    }

    q = q.orderBy(sortField, sortDir === "asc" ? "asc" : "desc");

    //simple offset-based pagination
    const offset = (page - 1) * pageSize;
    q = q.offset(offset).limit(pageSize);
    //execute query
    const snap = await q.get();
    //map results
    const items = snap.docs.map((doc) => ({
      id: doc.id,
      ...doc.data(),
    }));
    //determine if more pages
    const hasMore = items.length === pageSize;
    //return response
    return NextResponse.json({
      items,
      page,
      pageSize,
      hasMore,
      sort: sortParam,
      source,
    });
  } catch (err) {
    console.error("[review-events GET] error:", err);
    return NextResponse.json(
      { error: "Failed to fetch review events" },
      { status: 500 }
    );
  }
}

//POST /api/review_events
//body: { id, isUnsafe, category, notes }
export async function POST(req) {
  if (!firestore) {
    return NextResponse.json({ error: "Firestore not available" }, { status: 500 });
  }
  //parse body
  try {
    const { id, isUnsafe, category, notes } = await req.json();
    if (!id || typeof isUnsafe !== "boolean") {
      return NextResponse.json({ error: "Invalid payload" }, { status: 400 });
    }
    //references
    const pendingRef = firestore.collection("safetyEventsPending").doc(id);
    const reviewedRef = firestore.collection("safetyEventsReviewed").doc(id);
    //transaction: read from pending, write to reviewed, delete from pending
    await firestore.runTransaction(async (tx) => {
      //get pending doc
      const snap = await tx.get(pendingRef);
      if (!snap.exists) throw new Error("Document not found");
      //get data
      const data = snap.data();

      //write to reviewed
      tx.set(reviewedRef, {
        ...data,
        humanLabel: {
          isUnsafe,
          category: category || null,
          notes: notes || null,
          reviewedAt: FieldValue.serverTimestamp()
        }
      });

      //delete from pending
      tx.delete(pendingRef);
    });
    //return success
    return NextResponse.json({ ok: true });
  } catch (err) {
    console.error("[review-events POST] error:", err);
    return NextResponse.json({ error: "Failed to update" }, { status: 500 });
  }
}