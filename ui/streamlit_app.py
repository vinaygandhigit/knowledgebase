from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

import requests
import streamlit as st

# Image formats a browser can render inline. Vector formats (wmf/emf), commonly
# emitted by Office documents, are offered as a download link instead.
DISPLAYABLE_IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}


def _asset_url(api_base_url: str, path: str) -> str:
    return f"{api_base_url}/api/v1/assets?path={quote(str(path), safe='')}"


def _ext(path: str) -> str:
    return path.rsplit(".", 1)[-1].lower() if "." in path else ""


def _render_chunk_text(text: str) -> None:
    """Render chunk text, turning ``[TABLE]`` pipe-blocks into real tables.
    Native Markdown tables (from .md files) are rendered by st.markdown directly."""
    if not text:
        return

    text_buffer: list[str] = []
    table_rows: list[list[str]] = []
    in_table = False

    def flush_text() -> None:
        if text_buffer:
            st.markdown("\n".join(text_buffer))
            text_buffer.clear()

    def flush_table() -> None:
        if not table_rows:
            return
        table = _table_from_rows(table_rows)
        if table:
            st.table(table)
        table_rows.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        # Internal placeholder markers — the real image/diagram is rendered
        # separately from visual_refs, so don't echo the marker as text.
        if stripped.startswith("[IMAGE]") or stripped.startswith("[DIAGRAM_SHAPE]"):
            continue
        if stripped.startswith("[TABLE]"):
            flush_text()
            flush_table()
            in_table = True
            continue
        if in_table:
            if "|" in line:
                table_rows.append([cell.strip() for cell in line.split("|")])
                continue
            if not stripped:
                continue
            flush_table()
            in_table = False
        text_buffer.append(line)

    flush_table()
    flush_text()


def _table_from_rows(rows: list[list[str]]) -> list[dict[str, str]] | None:
    if not rows:
        return None
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else rows

    columns: list[str] = []
    seen: dict[str, int] = {}
    for index, raw in enumerate(header):
        name = raw or f"col{index + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        columns.append(name)

    data: list[dict[str, str]] = []
    for row in body:
        data.append({columns[i]: (row[i] if i < len(row) else "") for i in range(len(columns))})
    return data or None


def _visual_key(visual: dict[str, Any]) -> str:
    """Stable identity for a visual so the same image/diagram is shown once per
    answer even though it's attached to every chunk of its source section."""
    return (
        visual.get("path")
        or visual.get("url")
        or visual.get("source")
        or f"{visual.get('name', '')}|{visual.get('shape_type', '')}"
    )


def _render_chunk_visuals(
    chunk: dict[str, Any], api_base_url: str, seen: set[str]
) -> None:
    # Skip visuals already rendered for an earlier source chunk of this answer.
    visual_refs = [
        v for v in (chunk.get("visual_refs", []) or []) if _visual_key(v) not in seen
    ]
    if not visual_refs:
        return

    st.markdown("**Visuals:**")
    for visual in visual_refs:
        seen.add(_visual_key(visual))
        visual_type = visual.get("type", "")
        name = visual.get("name") or "Visual"

        if visual_type in ("image", "diagram") and (visual.get("path") or visual.get("url")):
            url = visual.get("url") or _asset_url(api_base_url, visual["path"])
            source_ext = _ext(visual.get("url") or visual.get("path", ""))
            if visual.get("url") or source_ext in DISPLAYABLE_IMAGE_EXTS:
                st.image(url, caption=name, use_container_width=True)
            else:
                # Non-displayable (e.g. .wmf/.emf): offer the original instead.
                st.markdown(f"🖼️ {name} — [open / download]({url}) (`.{source_ext}` not renderable inline)")
        elif visual_type == "diagram":
            shape_type = visual.get("shape_type", "")
            st.caption(f"Diagram shape detected: {name} ({shape_type})")
        elif visual.get("source"):
            st.caption(f"Image reference (unresolved): {name} → {visual['source']}")


def _render_source(
    index: int, chunk: dict[str, Any], api_base_url: str, seen: set[str]
) -> None:
    badge = (
        "🔵" if chunk["search_type"] == "vector"
        else "🟠" if chunk["search_type"] == "keyword"
        else "🟢"
    )
    st.markdown(
        f"**{index}. {badge} `{chunk['file_name']}`** &nbsp; "
        f"score `{chunk['score']:.4f}` &nbsp; type `{chunk['search_type']}`"
    )

    location = chunk.get("source_location") or {}
    location_bits = []
    if location.get("page_number"):
        location_bits.append(f"page {location['page_number']}")
    if location.get("slide_number"):
        location_bits.append(f"slide {location['slide_number']}")
    if location.get("section_title"):
        location_bits.append(str(location["section_title"]))
    if location_bits:
        st.caption(" · ".join(location_bits))

    if chunk.get("chunk_text"):
        _render_chunk_text(chunk["chunk_text"])
    _render_chunk_visuals(chunk, api_base_url, seen)
    st.divider()


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_base = st.text_input(
        "API Base URL",
        value="http://localhost:8080",
        help="Base URL of the RAG ingestion API",
    )
    hybrid_alpha = st.slider(
        "Hybrid Search Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0 = pure keyword (BM25) · 1 = pure semantic (vector)",
    )

    alpha_label = (
        "Pure keyword (BM25)" if hybrid_alpha == 0.0
        else "Pure semantic (vector)" if hybrid_alpha == 1.0
        else f"Semantic {int(hybrid_alpha * 100)}% · Keyword {int((1 - hybrid_alpha) * 100)}%"
    )
    st.caption(f"🔍 {alpha_label}")

    st.divider()
    st.subheader("🏥 API Health")

    if st.button("Check Health", use_container_width=True):
        with st.spinner("Checking …"):
            try:
                r = requests.get(f"{api_base}/actuator/health", timeout=5)
                h = r.json()
                col_a, col_b = st.columns(2)
                col_a.metric("App", h.get("status", "—"))
                col_b.metric("DB", h.get("db", "—"))
            except Exception as exc:
                st.error(f"Cannot reach API: {exc}")

    st.divider()
    st.subheader("🔄 Reindex")
    if st.button("Trigger Full Reindex", use_container_width=True, type="secondary"):
        with st.spinner("Submitting …"):
            try:
                r = requests.post(f"{api_base}/api/v1/ingestion/reindex", timeout=10)
                data = r.json()
                st.success(f"Job started: `{data.get('job_id')}`")
                st.session_state["last_job_id"] = data.get("job_id", "")
            except Exception as exc:
                st.error(f"Error: {exc}")

    if "last_job_id" in st.session_state and st.session_state["last_job_id"]:
        job_id = st.session_state["last_job_id"]
        if st.button("Refresh Job Status", use_container_width=True):
            try:
                r = requests.get(
                    f"{api_base}/api/v1/ingestion/jobs/{job_id}", timeout=5
                )
                job = r.json()
                status_colour = {
                    "COMPLETED": "🟢",
                    "RUNNING": "🟡",
                    "FAILED": "🔴",
                }.get(job.get("status", ""), "⚪")
                st.markdown(
                    f"**{status_colour} {job.get('status')}** — "
                    f"{job.get('processed_files')}/{job.get('total_files')} files · "
                    f"{job.get('total_chunks')} chunks"
                )
                if job.get("message"):
                    st.caption(job["message"])
            except Exception as exc:
                st.error(f"Error: {exc}")

# ── Chat History ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # list[dict] with role / content / meta

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🧠 Knowledge Intelligence Platform")
st.caption(
    "Ask questions about your documents. Powered by hybrid vector + keyword "
    "search and Anthropic Claude. Retrieved images, diagrams, and tables are "
    "shown under each answer's sources."
)

# ── Render previous messages ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            meta: dict[str, Any] = msg["meta"]
            with st.expander(
                f"📎 {len(meta['chunks'])} source(s) · {meta['execution_time_ms']:.0f} ms",
                expanded=False,
            ):
                seen_visuals: set[str] = set()
                for i, chunk in enumerate(meta["chunks"], start=1):
                    _render_source(i, chunk, api_base, seen_visuals)

# ── Chat input ────────────────────────────────────────────────────────────────
if query := st.chat_input("Ask a question about your documents …"):
    # Render user bubble
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating response …"):
            try:
                t0 = time.perf_counter()
                response = requests.post(
                    f"{api_base}/api/v1/retrieval/query",
                    json={"query": query, "hybrid_alpha": hybrid_alpha},
                    timeout=300,
                )
                response.raise_for_status()
                data = response.json()
                elapsed_ui = round((time.perf_counter() - t0) * 1000, 1)

                answer = data.get("response", "No response received.")
                chunks = data.get("retrieved_chunks", [])
                api_ms = data.get("execution_time_ms", elapsed_ui)

                st.markdown(answer)

                meta = {"chunks": chunks, "execution_time_ms": api_ms}
                with st.expander(
                    f"📎 {len(chunks)} source(s) · {api_ms:.0f} ms", expanded=True
                ):
                    seen_visuals: set[str] = set()
                    for i, chunk in enumerate(chunks, start=1):
                        _render_source(i, chunk, api_base, seen_visuals)

                # Persist to history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "meta": meta}
                )

            except requests.exceptions.ConnectionError:
                err = f"❌ Cannot connect to `{api_base}`. Is the API running?"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except requests.exceptions.HTTPError as exc:
                err = f"❌ API error {exc.response.status_code}: {exc.response.text}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as exc:
                err = f"❌ Unexpected error: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

# ── Clear chat button ─────────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ Clear chat", key="clear"):
        st.session_state.messages = []
        st.rerun()
