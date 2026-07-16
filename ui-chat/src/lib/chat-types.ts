export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export type DisplayMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  state?: "streaming" | "complete" | "error" | "stopped";
};

export type AgentMessage = {
  role: "user" | "assistant" | "tool";
  content: string;
  [key: string]: JsonValue;
};

export type ChatRequest = {
  prompt: string;
  prior_messages: AgentMessage[];
};

export type TokenEvent = {
  content: string;
  iteration?: number;
};

export type CompleteEvent = {
  status: "completed" | "completed_with_errors";
  final_answer: string;
  messages: AgentMessage[];
};

export type ErrorEvent = {
  code: string;
  message: string;
};
