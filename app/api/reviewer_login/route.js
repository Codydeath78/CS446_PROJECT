import { NextResponse } from "next/server";

const REVIEWER_PASSWORD = (process.env.REVIEWER_PASSWORD || "").trim();

//POST /api/reviewer_login
export async function POST(req) {
  //expecting JSON body: { password: string }
  try {
    //parse request body
    const { password } = await req.json();
    //validate password
    if (!REVIEWER_PASSWORD) {
      console.warn("[reviewer-login] REVIEWER_PASSWORD not set");
      return NextResponse.json({ ok: false, error: "Not configured" }, { status: 500 });
    }
    //check password
    if (password !== REVIEWER_PASSWORD) {
      return NextResponse.json({ ok: false, error: "Invalid password" }, { status: 401 });
    }
    //successful login
    return NextResponse.json({ ok: true });
  } catch (err) {
    console.error("[reviewer-login] error", err);
    return NextResponse.json({ ok: false, error: "Server error" }, { status: 500 });
  }
}

