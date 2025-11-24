import { Firestore } from "@google-cloud/firestore";
import { Storage } from "@google-cloud/storage";
import fs from "node:fs";
import path from "node:path";

//environment variable specifying the GCS bucket for export
const BUCKET_NAME = process.env.SAFETY_EXPORT_BUCKET;
const COLLECTION = "safetyEventsReviewed";

//main function to export safety data
async function main() {
  if (!BUCKET_NAME) {
    console.error("SAFETY_EXPORT_BUCKET env var is required");
    process.exit(1);
  }

  //initialize firestore and storage clients
  const firestore = new Firestore();
  const storage = new Storage();

  //create a timestamped local file for export
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const localPath = path.join(process.cwd(), `safety_export_${ts}.jsonl`);
  const writeStream = fs.createWriteStream(localPath, { encoding: "utf8" });

  console.log(`Exporting from collection "${COLLECTION}"...`);
  //fetch all documents from the specified collection
  const snap = await firestore.collection(COLLECTION).get();
  let count = 0;
  //process each document and write to the local file in JSONL format
  snap.forEach((doc) => {
    //extract data from the document
    const data = doc.data();
    //construct a row object with relevant fields
    const row = {
      id: doc.id,
      text: data.preview ?? "", //original text
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
    //write the JSON stringified row to the file
    writeStream.write(JSON.stringify(row) + "\n");
    count++;
  });
  //finalize the write stream
  writeStream.end();
  await new Promise((res) => writeStream.on("finish", res));
  console.log(`Wrote ${count} rows to ${localPath}`);
  //upload the local file to the specified GCS bucket
  const destFile = `exports/safety_export_${ts}.jsonl`;
  console.log(`Uploading to gs://${BUCKET_NAME}/${destFile}...`);
  //upload the file to GCS
  await storage.bucket(BUCKET_NAME).upload(localPath, {
    destination: destFile,
    contentType: "application/jsonl",
  });

  console.log("Upload complete.");
}
//execute the main function and handle errors
main().catch((err) => {
  console.error(err);
  process.exit(1);
});