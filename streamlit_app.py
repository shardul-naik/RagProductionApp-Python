import asyncio
import logging
import time
import traceback

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

from storage import upload_pdf

load_dotenv()
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def _inngest_is_production() -> bool:
    configured_mode = os.getenv("INNGEST_IS_PRODUCTION")
    if configured_mode is not None:
        return configured_mode.strip().lower() in {"1", "true", "yes", "on"}
    return bool(os.getenv("INNGEST_SIGNING_KEY"))


def run_async(coro):
    import asyncio

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()


st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    api_base_url = os.getenv("INNGEST_API_BASE", "https://api.inngest.com")
    st.write(f"DEBUG api_base_url={api_base_url}")
    st.write(f"DEBUG is_production={_inngest_is_production()}")
    return inngest.Inngest(
        api_base_url=api_base_url,
        app_id="rag_app",
        event_key=os.environ["INNGEST_EVENT_KEY"],
        signing_key=os.getenv("INNGEST_SIGNING_KEY"),
        is_production=_inngest_is_production(),
        request_timeout=120_000,
    )


async def send_rag_ingest_event(object_key: str, source_id: str) -> None:
    client = get_inngest_client()
    try:
        await client.send(
            inngest.Event(
                name="rag/ingest_pdf",
                data={
                    "object_key": object_key,
                    "source_id": source_id,
                },
            )
        )
    except Exception as e:
        logger.exception("Inngest ingestion event send failed")
        st.error(f"DEBUG: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        object_key = upload_pdf(uploaded.name, uploaded.getvalue())
        # Kick off the event and block until the send completes
        run_async(send_rag_ingest_event(object_key, uploaded.name))
        # Small pause for user feedback continuity
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {uploaded.name}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int) -> None:
    client = get_inngest_client()
    try:
        result = await client.send(
            inngest.Event(
                name="rag/query_pdf_ai",
                data={
                    "question": question,
                    "top_k": top_k,
                },
            )
        )
        return result[0]
    except Exception:
        logger.exception("Inngest query event send failed")
        raise


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "https://api.inngest.com").rstrip("/")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/v1/events/{event_id}/runs"
    headers = {
        "Authorization": f"Bearer {os.environ['INNGEST_SIGNING_KEY']}",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            # Fire-and-forget event to Inngest for observability/workflow
            event_id = run_async(send_rag_query_event(question.strip(), int(top_k)))
            # Poll the local Inngest API for the run's output
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")