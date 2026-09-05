from __future__ import annotations

import operator
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated
from urllib.parse import quote

import requests
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", max_retries=5)


class Task(BaseModel):
    id: int
    title: str
    goal: str
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = 5


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = ""
    filename: str
    alt: str
    caption: str
    prompt: str
    after_section: str = Field(..., description="Exact section heading text (without ##) to insert this image after.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list)


class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str


ROUTER_SYSTEM = """You are a routing module for a technical blog planner.
Decide whether web research is needed before planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/pricing/policy topics.

If needs_research=true, output 3-10 high-signal, scoped queries."""


def router_node(state: State) -> dict:
    decision = llm.with_structured_output(RouterDecision).invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
    ])
    recency_days = {"open_book": 7, "hybrid": 45}.get(decision.mode, 3650)
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        print(f"[tavily] TAVILY_API_KEY not set — skipping query: {query}")
        return []
    try:
        from langchain_tavily import TavilySearch
        results = TavilySearch(max_results=max_results).invoke({"query": query})
        return [
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
            for r in results or []
        ]
    except Exception as e:
        print(f"[tavily] search failed for '{query}': {e}")
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


RESEARCH_SYSTEM = """You are a research synthesizer. Convert raw search results into EvidenceItem objects.
Only include items with a non-empty url. Normalize published_at to ISO YYYY-MM-DD if reliably inferable, else null.
Keep snippets short. Deduplicate by URL."""


def research_node(state: State) -> dict:
    raw: List[dict] = []
    for q in (state.get("queries") or [])[:10]:
        raw.extend(_tavily_search(q))

    if not raw:
        return {"evidence": []}

    pack = llm.with_structured_output(EvidencePack).invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"As-of: {state['as_of']}\nRecency days: {state['recency_days']}\n\nRaw results:\n{raw}"),
    ])

    evidence = list({e.url: e for e in pack.evidence if e.url}.values())

    if state.get("mode") == "open_book":
        cutoff = date.fromisoformat(state["as_of"]) - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}


ORCH_SYSTEM = """You are a senior technical writer. Produce an actionable outline for a technical blog post.

- 3-5 tasks only, each with goal + 3-5 bullets + target_words (100-200 words each).
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True, requires_citations=True.
- open_book: set blog_kind="news_roundup", no tutorial content, don't invent events if evidence is weak.

Output must match the Plan schema."""


def orchestrator_node(state: State) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = llm.with_structured_output(Plan).invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\nMode: {mode}\n"
            f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
            f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
            f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
        )),
    ])
    if forced_kind:
        plan.blog_kind = "news_roundup"
    return {"plan": plan}


def fanout(state: State):
    plan = state["plan"]
    return [
        Send("worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "mode": state["mode"],
            "as_of": state["as_of"],
            "recency_days": state["recency_days"],
            "plan": plan.model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
        })
        for task in plan.tasks
    ]


WORKER_SYSTEM = """You are a senior technical writer. Write ONE section of a technical blog post in Markdown.

- Cover all bullets in order, concisely. Target words +-10%, do not exceed it. Output only markdown starting with "## <Section Title>".
- If blog_kind=="news_roundup", focus on events + implications, not tutorials.
- If mode=="open_book": only make claims supported by provided evidence URLs, cited as [Source](URL). Otherwise write "Not found in provided sources."
- If requires_citations==true, cite evidence URLs for external claims.
- If requires_code==true, include at least one minimal snippet."""


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}" for e in evidence[:20])

    response = llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Blog title: {plan.blog_title}\nAudience: {plan.audience}\nTone: {plan.tone}\n"
            f"Blog kind: {plan.blog_kind}\nConstraints: {plan.constraints}\n"
            f"Topic: {payload['topic']}\nMode: {payload.get('mode')}\n"
            f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
            f"Section title: {task.title}\nGoal: {task.goal}\nTarget words: {task.target_words}\n"
            f"Tags: {task.tags}\nrequires_research: {task.requires_research}\n"
            f"requires_citations: {task.requires_citations}\nrequires_code: {task.requires_code}\n"
            f"Bullets:{bullets_text}\n\nEvidence (ONLY cite these URLs):\n{evidence_text}\n"
        )),
    ])

    content = response.content
    if isinstance(content, list):
        section_md = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    else:
        section_md = content.strip()

    return {"sections": [(task.id, section_md)]}


def merge_content(state: State) -> dict:
    plan = state["plan"]
    ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    return {"merged_md": f"# {plan.blog_title}\n\n" + "\n\n".join(ordered).strip() + "\n"}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor. Decide if images/diagrams are needed for this blog.

- Max 3 images total. Each must materially improve understanding.
- For each image, set after_section to the EXACT section heading text it should follow (without the ## marks).
- If no images needed, return images=[].
Do NOT reproduce or repeat the blog body — only return image specs.
Return strictly GlobalImagePlan."""


import difflib

def decide_images(state: State) -> dict:
    plan = state["plan"]
    merged_md = state["merged_md"]
    section_titles = [t.title for t in plan.tasks]
    section_titles_text = "\n".join(f"- {t}" for t in section_titles)

    image_plan = llm.with_structured_output(GlobalImagePlan).invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=f"Blog kind: {plan.blog_kind}\nTopic: {state['topic']}\n\nSection headings:\n{section_titles_text}"),
    ])

    md = merged_md
    specs = []
    for i, img in enumerate(image_plan.images, start=1):
        placeholder = f"[[IMAGE_{i}]]"
        img.placeholder = placeholder

        match = difflib.get_close_matches(img.after_section, section_titles, n=1, cutoff=0.5)
        target_title = match[0] if match else (section_titles[0] if section_titles else None)
        if not target_title:
            continue

        heading = f"## {target_title}"
        if heading in md:
            start = md.index(heading)
            next_heading = md.find("\n## ", start + len(heading))
            insert_at = next_heading if next_heading != -1 else len(md)
            md = md[:insert_at] + f"\n\n{placeholder}\n\n" + md[insert_at:]
            specs.append(img.model_dump())

    return {
        "md_with_placeholders": md,
        "image_specs": specs,
    }

def _pollinations_generate_image_bytes(prompt: str) -> bytes:
    encoded_prompt = quote(prompt)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width=1024&height=1024&nologo=true&model=flux"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def _safe_slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9 _-]+", "", title.strip().lower())
    return re.sub(r"\s+", "_", s).strip("_") or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    md = state.get("md_with_placeholders") or state["merged_md"]
    specs = state.get("image_specs", []) or []

    if specs:
        images_dir = Path("images")
        images_dir.mkdir(exist_ok=True)
        for spec in specs:
            out_path = images_dir / spec["filename"]
            if not out_path.exists():
                try:
                    out_path.write_bytes(_pollinations_generate_image_bytes(spec["prompt"]))
                except Exception as e:
                    md = md.replace(spec["placeholder"], f"> **[IMAGE GENERATION FAILED]** {e}\n")
                    continue
            md = md.replace(spec["placeholder"], f"![{spec['alt']}](images/{spec['filename']})\n*{spec['caption']}*")

    Path(f"{_safe_slug(plan.blog_title)}.md").write_text(md, encoding="utf-8")
    return {"final": md}


reducer = StateGraph(State)
reducer.add_node("merge_content", merge_content)
reducer.add_node("decide_images", decide_images)
reducer.add_node("generate_and_place_images", generate_and_place_images)
reducer.add_edge(START, "merge_content")
reducer.add_edge("merge_content", "decide_images")
reducer.add_edge("decide_images", "generate_and_place_images")
reducer.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer.compile()

g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()