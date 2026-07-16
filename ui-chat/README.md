# ReAct Chat UI

A minimal, ChatGPT-like browser interface for the repository's canonical ReAct agent. The Next.js app keeps the visible conversation in browser memory, proxies a server-sent event stream, and displays tokens from the local Ollama-backed agent as they arrive.

## Architecture

```text
Browser -> Next.js /api/chat -> FastAPI /chat -> ReAct.run_agent() -> Ollama
```

The FastAPI service imports the existing agent from `../ReAct`; it does not maintain a second agent implementation. Completed Ollama message history is returned to the browser and sent back with the next turn. Refreshing or starting a new chat clears that in-memory history.

## Prerequisites

- Node.js 20 or newer and npm
- Python 3.11 or newer
- [Ollama](https://ollama.com/) running locally
- The default model installed:

  `ollama pull llama3.2:3b`

## Install

Run these commands from this `ui-chat` directory.

### Python backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

### Next.js frontend

```powershell
npm install
```

Copy `.env.example` to `.env.local` only when changing the defaults.

## Run

Use two terminals from this directory.

```powershell
# Terminal 1
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app:app --reload --port 8000
```

```powershell
# Terminal 2
npm run dev
```

Open <http://localhost:3000>. FastAPI health information is available at <http://127.0.0.1:8000/health>.

## Configuration

| Variable | Used by | Default | Purpose |
| --- | --- | --- | --- |
| `AGENT_API_URL` | Next.js server | `http://127.0.0.1:8000` | FastAPI sidecar URL |
| `AGENT_MODEL` | FastAPI | `llama3.2:3b` | Ollama model tag |
| `AGENT_MAX_ITERATIONS` | FastAPI | `4` | Maximum model turns per request |

`AGENT_API_URL` is intentionally server-only. Do not rename it with a `NEXT_PUBLIC_` prefix.

## Test and build

```powershell
python -m pytest backend\tests
npm run lint
npm run build
```

The backend tests inject a fake agent runner and do not require Ollama.

## MVP behavior

- Enter sends; Shift+Enter inserts a newline.
- Responses stream token by token and support Markdown/GFM.
- The stop button aborts browser consumption. The canonical Python agent is synchronous, so its active model invocation may continue in the backend until it returns.
- Only completed turns update agent context. Failed or stopped partial answers are excluded from the next request.
- Conversations are not persisted and disappear on refresh.
- The UI intentionally excludes authentication, a conversation sidebar, attachments, voice, model selection, RAG controls, and evaluation metrics.

## Troubleshooting

- **Cannot reach the local agent service:** start Uvicorn on port 8000, or set `AGENT_API_URL` to its actual URL.
- **Ollama connection error:** start Ollama and verify `ollama list` contains the configured model.
- **Model not found:** run `ollama pull llama3.2:3b`, or set `AGENT_MODEL` to an installed model before starting Uvicorn.
- **No tokens appear for a while:** local inference and tool calls can be quiet; the API sends keep-alive frames while it waits.
