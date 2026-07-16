import type {
  ChatRequest,
  CompleteEvent,
  ErrorEvent,
  TokenEvent,
} from "@/lib/chat-types";

type StreamHandlers = {
  onToken: (event: TokenEvent) => void;
  onComplete: (event: CompleteEvent) => void;
  onError: (event: ErrorEvent) => void;
};

function dispatchEvent(block: string, handlers: StreamHandlers) {
  if (!block.trim() || block.startsWith(":")) return;

  let eventName = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (!dataLines.length) return;

  const payload: unknown = JSON.parse(dataLines.join("\n"));
  if (eventName === "token") handlers.onToken(payload as TokenEvent);
  if (eventName === "complete") handlers.onComplete(payload as CompleteEvent);
  if (eventName === "error") handlers.onError(payload as ErrorEvent);
}

export async function streamChat(
  request: ChatRequest,
  signal: AbortSignal,
  handlers: StreamHandlers,
) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(
      typeof detail?.error === "string"
        ? detail.error
        : "The agent service is unavailable.",
    );
  }
  if (!response.body) throw new Error("The response stream was empty.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, "\n");

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      dispatchEvent(buffer.slice(0, boundary), handlers);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
    }
    if (done) break;
  }

  if (buffer.trim()) dispatchEvent(buffer, handlers);
}
