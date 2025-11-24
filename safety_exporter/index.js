import express from "express";
import { Firestore } from "@google-cloud/firestore";
import { Storage } from "@google-cloud/storage";
import fs from "node:fs";
import path from "node:path";

//export logic
const firestore = new Firestore();
const storage = new Storage();

//configuration
const BUCKET_NAME = process.env.SAFETY_EXPORT_BUCKET;
const COLLECTION = "safetyEventsReviewed";

//this does the export
async function runExport() {
  if (!BUCKET_NAME) {
    throw new Error("SAFETY_EXPORT_BUCKET env var is required");
  }

  //create local temp file
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  //export path
  const localPath = path.join("/tmp", `safety_export_${ts}.jsonl`);
  //create write stream
  const writeStream = fs.createWriteStream(localPath, { encoding: "utf8" });

  console.log(`Exporting from collection "${COLLECTION}"...`);
  //fetch all documents
  const snap = await firestore.collection(COLLECTION).get();
  let count = 0;
  //process each document
  snap.forEach((doc) => {
    //extract data
    const data = doc.data();
    //structure row
    const row = {
      id: doc.id,
      text: data.preview ?? "",
      rephrased: data.rephrased ?? "",
      source: data.source ?? "user",
      biased_score: data.detector?.scores?.biased ?? null,
      neutral_score: data.detector?.scores?.neutral ?? null,
      isUnsafe: data.humanLabel?.isUnsafe ?? null,
      category: data.humanLabel?.category ?? null,
      notes: data.humanLabel?.notes ?? null,
      createdAt: data.createdAt?._seconds
        ? new Date(data.createdAt._seconds * 1000).toISOString()
        : null,
    };
    //write as JSON line
    writeStream.write(JSON.stringify(row) + "\n");
    count++;
  });
  //finalize write stream
  writeStream.end();
  //wait for finish
  await new Promise((res) => writeStream.on("finish", res));
  console.log(`Wrote ${count} rows to ${localPath}`);
  //upload to GCS
  const destFile = `exports/safety_export_${ts}.jsonl`;
  console.log(`Uploading to gs://${BUCKET_NAME}/${destFile}...`);
  //upload file
  await storage.bucket(BUCKET_NAME).upload(localPath, {
    destination: destFile,
    contentType: "application/jsonl",
  });

  console.log("Upload complete.");
  return { count, gcsPath: `gs://${BUCKET_NAME}/${destFile}` };
}

//express server for Cloud Run

//setup express app
const app = express();

//setup routes
app.get("/", (req, res) => {
  res.send("Safety exporter is up. Call /run-export to trigger an export.");
});

//route to trigger export
app.post("/run-export", async (req, res) => {
  //run export
  try {
    const result = await runExport();
    res.json({ ok: true, ...result });
  } catch (err) {
    console.error("[run-export] error", err);
    res.status(500).json({ ok: false, error: String(err) });
  }
});

//start server
const PORT = process.env.PORT || 8080;
//listen on port
app.listen(PORT, () => {
  console.log(`Exporter listening on port ${PORT}`);
});