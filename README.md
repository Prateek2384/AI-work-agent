# AI Task Worker

An LLM-driven agent that completes a back-office task end to end: it logs into a simulated vendor portal, finds the latest invoice from Acme Supplies, extracts the amount and due date from the PDF, enters it into a simulated internal ERP, and verifies the result.

The default task is:

> Find the latest invoice from Acme Supplies, extract the amount and due date, enter it into our internal system, and tell me once it is done.

## Project layout

| Path | Purpose |
|------|---------|
| `agent/` | Agent loop, planner, memory, verifier and run tracing |
| `llm/` | OpenAI-compatible LLM client |
| `tools/` | Browser (Playwright), file, and HTTP tools exposed to the agent |
| `env/` | Simulated vendor portal (port 8001) and ERP (port 8002) |
| `ui/` | Chat web UI (port 8000) |
| `tests/` | Pytest suite |
| `runs/` | Per-run evidence and traces (generated, git-ignored) |

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

### Configure the LLM

Create a `.env` file in the project root. Any OpenAI-compatible endpoint works.

**GroqCloud**

```env
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your_groq_key_here
LLM_MODEL=openai/gpt-oss-120b
```

**Local Ollama**

```bash
ollama create qwen3-agent -f Modelfile
```

```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen3-agent
```

`Modelfile.gptoss` builds an equivalent model from `gpt-oss:20b`.

Optional settings: `LLM_REASONING_EFFORT`, `LLM_FAST_EFFORT`, `LLM_MAX_TOKENS` (default 4000), `LLM_FAST_MAX_TOKENS` (default 700), `VERIFY=0` to disable the LLM verifier.

## Run

### Command line

```bash
python cli.py                          # run the default task
python cli.py "your own task"          # run a custom task
python cli.py --visible                # show the browser window
python cli.py --flaky                  # ERP fails the first save with a 500
python cli.py --variant b              # alternate UI label variant
python cli.py --allow-questions        # let the agent ask you questions
python cli.py --max-steps 40           # cap the number of agent steps (default 60)
```

The CLI starts the simulated portal and ERP automatically, prompts you before any irreversible action, and prints the final ERP state. Evidence for each run is saved under `runs/`. The exit code is 0 when the task finishes successfully.

### Chat UI

```bash
python -m ui.server
```

Open http://127.0.0.1:8000.

### Simulated apps only

```bash
python -m env.run
```

Vendor portal: http://127.0.0.1:8001 (login `ap-team` / `portal123`). ERP: http://127.0.0.1:8002.

## Tests

```bash
pytest
```
