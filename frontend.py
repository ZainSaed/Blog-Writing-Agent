from datetime import date

import pandas as pd
import streamlit as st

from backend import app
from db import init_db, save_blog, list_blogs, load_blog, delete_blog
from pathlib import Path
import re as _re

def render_markdown_with_images(md: str):
    pattern = _re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    last_end = 0
    for m in pattern.finditer(md):
        st.markdown(md[last_end:m.start()])
        alt, src = m.group(1), m.group(2)
        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=alt or None, use_container_width=True)
        elif Path(src).exists():
            st.image(src, caption=alt or None, use_container_width=True)
        else:
            st.warning(f"Image not found: {src}")
        last_end = m.end()
    st.markdown(md[last_end:])

init_db()

st.set_page_config(page_title="Blog Writing Agent", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #000000; color: #ffffff; }
    h1, h2, h3, p, label, span, .stMarkdown, .stRadio label { color: #ffffff !important; }
    .stTextInput > div, .stTextInput > div > div,
    div[data-baseweb="base-input"], div[data-baseweb="input"],
    .stTextInput input {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    .stTextInput input:hover, .stTextInput input:focus, .stTextInput input:active {
        background-color: #111111 !important;
    }
    .stButton button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 600;
}
.stButton button p {
    color: #000000 !important;
}
    .stTabs [data-baseweb="tab"] { color: #ffffff !important; }
    .stDataFrame { background-color: #111111; }
    [data-testid="stSidebar"] { background-color: #0a0a0a; }
</style>
""", unsafe_allow_html=True)

st.title("Blog Writing Agent")

# ── Sidebar: past blogs ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Past Blogs")
    past = list_blogs()

    if not past:
        st.caption("No blogs yet. Generate one first.")
    else:
        for blog in past:
            col1, col2 = st.columns([4, 1])
            with col1:
                label = f"**{blog['title'][:40]}**\n{blog['created_at'][:10]}"
                if st.button(label, key=f"load_{blog['id']}", use_container_width=True):
                    st.session_state["loaded"] = load_blog(blog["id"])
            with col2:
                if st.button("🗑", key=f"del_{blog['id']}"):
                    delete_blog(blog["id"])
                    st.rerun()

# ── Main: generate ───────────────────────────────────────────────────────
topic = st.text_input("Enter a blog topic")
run = st.button("Generate", type="primary")

if "loaded" not in st.session_state:
    st.session_state["loaded"] = None

if run:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": date.today().isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    NODE_LABELS = {
        "router":       ("🧭", "Routing the request"),
        "research":     ("🔍", "Researching the web"),
        "orchestrator": ("🗂️", "Planning the blog outline"),
        "worker":       ("✍️", "Writing section"),
        "reducer":      ("🎨", "Merging content & adding images"),
    }

    final_state: dict = {}
    worker_count = 0

    with st.status("Running agents...", expanded=True) as status_box:
        try:
            for update in app.stream(inputs, stream_mode="updates", config={"max_concurrency": 2}):
                node = next(iter(update))
                final_state.update(update[node])
                if node == "worker":
                    worker_count += 1
                    status_box.write(f"✍️ Writing section {worker_count}")
                else:
                    icon, label = NODE_LABELS.get(node, ("⚙️", node))
                    status_box.write(f"{icon} {label}")
            status_box.update(label="✅ Blog generated", state="complete", expanded=False)
        except Exception as e:
            status_box.update(label="❌ Generation failed", state="error", expanded=False)
            st.error(f"Error: {e}")
            st.stop()

    final_md = final_state.get("final", "")
    plan = final_state.get("plan")
    evidence = final_state.get("evidence", []) or []

    if final_md:
        title = plan.blog_title if plan and hasattr(plan, "blog_title") else topic.strip()
        evidence_list = [e.model_dump() if hasattr(e, "model_dump") else e for e in evidence]
        save_blog(topic.strip(), title, final_md, evidence_list)
        st.session_state["loaded"] = {
            "topic": topic.strip(),
            "title": title,
            "markdown": final_md,
            "evidence": evidence_list,
        }
        st.rerun()

# ── Results view ─────────────────────────────────────────────────────────
out = st.session_state.get("loaded")
if out:
    st.subheader(out.get("title", "Blog"))

    tab_blog, tab_evidence = st.tabs(["📝 Blog", "🔎 Evidence"])

    with tab_blog:
        render_markdown_with_images(out["markdown"])
        st.download_button(
            "⬇️ Download Markdown",
            data=out["markdown"].encode("utf-8"),
            file_name="blog.md",
            mime="text/markdown",
        )

    with tab_evidence:
        evidence = out.get("evidence", [])
        if not evidence:
            st.info("No evidence collected (closed-book topic or no Tavily key).")
        else:
            rows = [
                {
                    "Title": e.get("title", ""),
                    "Source": e.get("source", ""),
                    "Published": e.get("published_at", ""),
                    "URL": e.get("url", ""),
                }
                for e in evidence
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("Enter a topic and click **Generate**.")