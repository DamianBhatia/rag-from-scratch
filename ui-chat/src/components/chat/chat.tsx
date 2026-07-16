"use client";

import { useEffect, useRef, useState } from "react";

import { streamChat } from "@/lib/chat-stream";
import type { AgentMessage, DisplayMessage } from "@/lib/chat-types";

import { Composer } from "./composer";
import { MessageContent } from "./message-content";

function createId() {
  return crypto.randomUUID();
}

export function Chat() {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [agentHistory, setAgentHistory] = useState<AgentMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage() {
    const prompt = draft.trim();
    if (!prompt || isStreaming) return;

    const assistantId = createId();
    const controller = new AbortController();
    let completed = false;
    let reportedError = false;

    controllerRef.current = controller;
    setDraft("");
    setIsStreaming(true);
    setMessages((current) => [
      ...current,
      { id: createId(), role: "user", content: prompt, state: "complete" },
      { id: assistantId, role: "assistant", content: "", state: "streaming" },
    ]);

    try {
      await streamChat(
        { prompt, prior_messages: agentHistory },
        controller.signal,
        {
          onToken: ({ content }) => {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, content: message.content + content }
                  : message,
              ),
            );
          },
          onComplete: ({ messages: history, final_answer: finalAnswer }) => {
            completed = true;
            setAgentHistory(history);
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: message.content || finalAnswer,
                      state: "complete",
                    }
                  : message,
              ),
            );
          },
          onError: ({ message: errorMessage }) => {
            reportedError = true;
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: message.content
                        ? `${message.content}\n\n${errorMessage}`
                        : errorMessage,
                      state: "error",
                    }
                  : message,
              ),
            );
          },
        },
      );

      if (!completed && !reportedError) {
        throw new Error("The response ended before the agent completed.");
      }
    } catch (error) {
      const stopped = controller.signal.aborted;
      const message =
        error instanceof Error ? error.message : "The request could not be completed.";
      setMessages((current) =>
        current.map((item) =>
          item.id === assistantId
            ? {
                ...item,
                content: item.content || (stopped ? "Generation stopped." : message),
                state: stopped ? "stopped" : "error",
              }
            : item,
        ),
      );
    } finally {
      controllerRef.current = null;
      setIsStreaming(false);
    }
  }

  function resetChat() {
    if (isStreaming) return;
    setMessages([]);
    setAgentHistory([]);
    setDraft("");
  }

  return (
    <main className="chat-shell">
      <header className="topbar">
        <button
          type="button"
          className="new-chat"
          onClick={resetChat}
          disabled={isStreaming || messages.length === 0}
          aria-label="Start a new chat"
          title="New chat"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 20h9M16.5 3.5a2.12 2.12 0 0 1 3 3L8 18l-4 1 1-4Z" />
          </svg>
        </button>
        <div className="product-name">ReAct <span>Local</span></div>
        <div className="topbar-spacer" />
      </header>

      <section
        className={`conversation ${messages.length === 0 ? "is-empty" : ""}`}
        aria-label="Conversation"
      >
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="agent-mark" aria-hidden="true">R</div>
            <h1>What can I help with?</h1>
          </div>
        ) : (
          <div className="transcript" aria-live="polite">
            {messages.map((message) => (
              <article
                key={message.id}
                className={`message ${message.role} ${message.state ?? ""}`}
              >
                <div className="message-inner">
                  {message.role === "assistant" ? (
                    <>
                      {message.content ? (
                        <div className="markdown-body">
                          <MessageContent content={message.content} />
                        </div>
                      ) : (
                        <div className="thinking" aria-label="Agent is thinking">
                          <span /><span /><span />
                        </div>
                      )}
                      {message.state === "error" && (
                        <span className="message-status">Response failed</span>
                      )}
                      {message.state === "stopped" && (
                        <span className="message-status">Stopped</span>
                      )}
                    </>
                  ) : (
                    <p>{message.content}</p>
                  )}
                </div>
              </article>
            ))}
            <div ref={transcriptEndRef} />
          </div>
        )}
      </section>

      <div className="composer-dock">
        <div className="composer-wrap">
          <Composer
            value={draft}
            busy={isStreaming}
            onChange={setDraft}
            onSubmit={sendMessage}
            onStop={() => controllerRef.current?.abort()}
          />
          <p className="disclaimer">Local models can make mistakes. Check important info.</p>
        </div>
      </div>
    </main>
  );
}
