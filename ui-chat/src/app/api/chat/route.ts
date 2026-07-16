import type { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_PROMPT_LENGTH = 20_000;
const MAX_HISTORY_MESSAGES = 200;

function isValidPayload(value: unknown): value is {
  prompt: string;
  prior_messages: unknown[];
} {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.prompt === "string" &&
    candidate.prompt.trim().length > 0 &&
    candidate.prompt.length <= MAX_PROMPT_LENGTH &&
    Array.isArray(candidate.prior_messages) &&
    candidate.prior_messages.length <= MAX_HISTORY_MESSAGES
  );
}

export async function POST(request: NextRequest) {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON request." }, { status: 400 });
  }

  if (!isValidPayload(payload)) {
    return Response.json(
      { error: "A non-empty prompt and valid conversation history are required." },
      { status: 400 },
    );
  }

  const baseUrl = (process.env.AGENT_API_URL ?? "http://127.0.0.1:8000").replace(
    /\/$/,
    "",
  );

  try {
    const upstream = await fetch(`${baseUrl}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
      signal: request.signal,
    });

    if (!upstream.ok || !upstream.body) {
      const detail = await upstream.json().catch(() => null);
      const message =
        typeof detail?.detail === "string"
          ? detail.detail
          : "The agent service rejected the request.";
      return Response.json({ error: message }, { status: upstream.status || 502 });
    }

    return new Response(upstream.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (error) {
    if (request.signal.aborted) {
      return new Response(null, { status: 499 });
    }
    console.error("Agent API request failed", error);
    return Response.json(
      { error: "Cannot reach the local agent service. Is FastAPI running?" },
      { status: 502 },
    );
  }
}
