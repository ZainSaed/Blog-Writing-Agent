# Blog Writing Agent

A multi-agent LLM system that generates technical blog posts end-to-end using LangGraph, with intelligent research, planning, parallel content generation, and image synthesis.

## Demo

**Live demo:** [Deploy to Streamlit Cloud](#deployment) (see instructions below)

## Overview

This project demonstrates a production-grade agentic AI workflow:

1. **Router** — analyzes the topic and decides whether web research is needed (closed-book / hybrid / open-book modes)
2. **Research** — fetches recent evidence from Tavily if the topic is time-sensitive
3. **Orchestrator** — plans the blog outline (3-4 sections with specific goals and word counts)
4. **Workers** — write each section in parallel using LangGraph's `Send`/fanout pattern
5. **Reducer** — merges sections, decides on images, generates and places them
6. **Storage** — saves every blog to SQLite with evidence links for portfolio/history

## Stack

| Component | Tool | Why |
|-----------|------|-----|
| **LLM (text)** | Groq (`gpt-oss-120b`) | Free tier (8k TPM), fast inference, structured output support |
| **Image generation** | Pollinations.ai | No API key required, free tier, flux model |
| **Web research** | Tavily | Free 1,000 searches/month, no credit card |
| **Agentic framework** | LangGraph | Native support for parallel workers, state management, conditional routing |
| **UI** | Streamlit | Fast iteration, minimal boilerplate, built-in dark mode |
| **Storage** | SQLite | Persistent blog history, local file (no DB setup) |

## Architecture

```
Topic Input
    ↓
[Router] → Decide: needs_research? closed_book/hybrid/open_book?
    ↓
    ├→ [Research] (if needed) → Tavily search → Evidence
    │
[Orchestrator] → Plan: 3-4 sections, goals, word counts
    ↓
[Fanout] → Send tasks to parallel workers
    ↓
[Worker₁] [Worker₂] [Worker₃] → Write sections concurrently
    ↓
[Merge] → Combine sections
    ↓
[Decide Images] → Determine if/where images help
    ↓
[Generate & Place] → Pollinations + insert into markdown
    ↓
[Save to SQLite] → Store markdown + evidence + metadata
    ↓
Blog Output (rendered in Streamlit)
```

## Features

- ✅ **Parallel section writing** — uses LangGraph's `Send` for concurrent workers
- ✅ **Smart research routing** — closed-book for evergreen, hybrid/open-book for timely topics
- ✅ **Auto-generated images** — detects where diagrams help and generates them via Pollinations
- ✅ **Local image rendering** — properly embeds generated images in the blog markdown
- ✅ **Blog history** — SQLite persistence, past blogs list in sidebar
- ✅ **Graceful error handling** — catches node failures, shows user-friendly errors instead of tracebacks
- ✅ **Evidence tracking** — stores research sources, shows them in the Evidence tab

## Installation

### Requirements

- Python 3.10+
- Free API keys (all optional, but recommended):
  - [Groq API key](https://console.groq.com) (free, no credit card)
  - [Tavily API key](https://tavily.com) (free 1,000 searches/month)
  - Pollinations.ai: **no key needed**

### Setup

```bash
git clone <your-repo-url>
cd Blog-Writing-Agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

If you skip `TAVILY_API_KEY`, the agent will default to closed-book (evergreen) mode.

## Running Locally

```bash
streamlit run frontend.py
```

Open http://localhost:8501 in your browser.

### Quick Test

1. Enter a topic: `"How to deploy LLMs in production"`
2. Click **Generate**
3. Watch the agent pipeline run (router → orchestrator → workers → reducer)
4. View the generated blog with images
5. Download as markdown or check the Evidence tab for sources

## Deployment

### Streamlit Cloud (1-click, recommended)

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → select your repo, branch, and `frontend.py`
4. Add secrets (Groq + Tavily keys) in the "Advanced settings" → "Secrets"
5. Click Deploy — live in ~2 min

**Note:** Streamlit Cloud runs serverless, so `blogs.db` is ephemeral (resets on redeploy). For persistent storage in production, upgrade to a Docker deployment or use a cloud database.

### Docker (for self-hosted)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t blog-agent .
docker run -p 8501:8501 -e GROQ_API_KEY=xxx -e TAVILY_API_KEY=yyy blog-agent
```

## Project Structure

```
Blog-Writing-Agent/
├── backend.py              # LangGraph agent + all nodes
├── frontend.py             # Streamlit UI
├── db.py                   # SQLite helpers
├── .env                    # API keys (gitignored)
├── blogs.db                # Generated (SQLite blog history)
├── images/                 # Generated (blog images)
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## How It Works

### Example: "AI Pulse: Key Developments from the Past Week"

1. **Router** sees "past week" → sets `mode=open_book`, `needs_research=True`
2. **Research** runs Tavily searches for recent AI news (7-day cutoff)
3. **Orchestrator** plans: "Weekly Highlights", "Key Releases", "Market Impact"
4. **Workers** write each section, citing evidence URLs
5. **Reducer** adds a diagram of the AI landscape, inserts it after "Key Releases"
6. **Result** → concise 3-section blog with sources and image, saved to SQLite

### Example: "How to Implement RAG Systems"

1. **Router** sees evergreen concept → sets `mode=closed_book`, `needs_research=False`
2. **Research** skipped
3. **Orchestrator** plans: "What is RAG", "Retrieval Strategies", "Production Patterns"
4. **Workers** write based on LLM knowledge (no web sources)
5. **Reducer** adds flowchart of retrieval pipeline
6. **Result** → stable educational blog, no time-dependency

## Key Design Decisions

- **Parallel workers** — LangGraph's `Send` allows all sections to write concurrently, reducing latency
- **No token-streaming in UI** — status updates show which *node* is running, not token-by-token text (simpler, clearer for demos)
- **Graceful image fallback** — if Pollinations fails, the blog still renders with an error note instead of crashing
- **Fuzzy section matching** — image placement uses `difflib.get_close_matches()` to handle slight LLM rewording of headings
- **SQLite over cloud DB** — keeps portfolio simple, no backend infrastructure; trades persistence for simplicity

## Limitations & Future Work

- **No real-time token streaming** — workers are silent until done; adding streaming would require LangChain's `LLMObserver`
- **Pollinations rate limit** — anonymous tier capped at ~1 request/15s; add an API key for higher volume
- **No user auth** — anyone with access can generate blogs; production would add per-user namespacing
- **Ephemeral storage on Streamlit Cloud** — `blogs.db` resets on redeploy; use PostgreSQL + `sqlalchemy` for production

## Contributing

This is a portfolio project. Feel free to fork, extend, and adapt for your own use case.


**Built with:** LangGraph, Groq, Pollinations, Tavily, Streamlit, SQLite

**Questions?** Check the [LangGraph docs](https://langchain-ai.github.io/langgraph/) or open an issue.
