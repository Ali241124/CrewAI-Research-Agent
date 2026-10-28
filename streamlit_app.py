import threading
from pathlib import Path
from datetime import datetime
from database import (
    add_research_run,
    get_recent_runs,
    search_research_runs,
    delete_research_run,
)
import streamlit as st
from research_crew import run_research

@st.cache_resource
def get_research_lock():
    return threading.Lock()


research_lock = get_research_lock()
# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="CrewAI Research Agent",
    page_icon="🔎",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# -----------------------------
# Page styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #111318;
        color: #E8EAF0;
    }

    [data-testid="stSidebar"] {
        background-color: #191C23;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #A6ADBB;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    div.stButton > button[kind="primary"] {
        background-color: #4F46E5;
        border: none;
        border-radius: 8px;
        min-height: 44px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Session state
# -----------------------------
if "research_result" not in st.session_state:
    st.session_state.research_result = None

if "research_topic" not in st.session_state:
    st.session_state.research_topic = ""


# -----------------------------
# Sidebar: Research History and Previous Reports
# -----------------------------
with st.sidebar:
    st.title("🔎 Research Agent")
    st.caption("AI-powered web research")
    st.divider()

    st.subheader("Research History")

    history_query = st.text_input(
        "Search past topics",
        placeholder="e.g. healthcare, RAG, agents",
        key="history_search",
    )

    history_runs = search_research_runs(
        query=history_query,
        limit=20,
    )

    if not history_runs:
        st.info("Your completed research runs will appear here.")

    else:
        for run in history_runs:
            date_label = run["created_at"][:10]

            with st.expander(
                f"#{run['id']} · {run['topic'][:35]} · {date_label}"
            ):
                st.caption(f"Sources: {run['sources_count']}")

                if run["validation_issues"]:
                    st.warning(
                        f"{len(run['validation_issues'])} "
                        "citation validation warning(s)"
                    )
                else:
                    st.caption(
                        "No citation validation warnings recorded."
                    )

                saved_path = Path(run["report_path"])

                if saved_path.is_file():
                    if st.button(
                        "Open saved report",
                        key=f"open_history_{run['id']}",
                        use_container_width=True,
                    ):
                        st.session_state.research_result = (
                            saved_path.read_text(encoding="utf-8")
                        )
                        st.session_state.research_topic = run["topic"]
                        st.rerun()
                else:
                    st.caption("Report file not found.")

                if st.button(
                    "Delete history entry",
                    key=f"delete_history_{run['id']}",
                    use_container_width=True,
                ):
                    deleted = delete_research_run(run["id"])

                    if deleted:
                        st.toast("History entry deleted.")
                    st.rerun()

    st.divider()
    st.subheader("Previous Reports")

    report_files = sorted(
        REPORTS_DIR.glob("*.md"),
        key=lambda file: file.stat().st_mtime,
        reverse=True,
    )

    if not report_files:
        st.info("No saved reports yet.")
    else:
        for report_file in report_files:
            with st.expander(
                report_file.stem.replace("_", " ")
            ):
                st.caption(
                    f"Modified: "
                    f"{datetime.fromtimestamp(report_file.stat().st_mtime):%d %b %Y, %I:%M %p}"
                )

                if st.button(
                    "Open report",
                    key=f"open_{report_file.name}",
                    use_container_width=True,
                ):
                    st.session_state.research_result = (
                        report_file.read_text(encoding="utf-8")
                    )
                    st.session_state.research_topic = report_file.stem
                    st.rerun()

# -----------------------------
# Main page
# -----------------------------
st.markdown(
    '<div class="main-title">Research Workspace</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Research a topic, evaluate sources, and generate a structured report.'
    '</div>',
    unsafe_allow_html=True,
)

with st.form("research_form"):
    topic = st.text_area(
        "Research topic",
        placeholder=(
            "Example: Practical applications of agentic AI "
            "in software engineering"
        ),
        height=100,
    )

    submitted = st.form_submit_button(
        "🚀 Start Research",
        type="primary",
        use_container_width=True,
    )


# -----------------------------
# Run research
# -----------------------------

if submitted:
    topic = topic.strip()

    if not topic:
        st.warning("Please enter a research topic.")

    elif not research_lock.acquire(blocking=False):
        st.warning(
            "Research is already running. Please wait for the current task to finish."
        )

    else:
        try:
            st.session_state.research_topic = topic
            st.session_state.research_result = None

            with st.status(
                "Research agent is working...",
                expanded=True,
            ) as status:
                st.write("Connecting to the local language model...")
                st.write("Researching the topic and evaluating sources...")
                st.write("Analyzing findings and preparing the report...")

                result = run_research(topic)

            report_text = result["report"]
            report_path = Path(result["report_path"])

            run_id = add_research_run(
                topic=topic,
                report_path=str(report_path),
                sources_count=len(result.get("sources", [])),
                validation_issues=result.get("validation_issues", []),
            )

            st.session_state.research_result = report_text
            st.session_state.research_topic = topic

            st.success(f"Research saved to history. Run ID: {run_id}")

            status.update(
                label="Research completed successfully!",
                state="complete",
                expanded=False,
            )

            st.success(f"Report saved: {report_path.name}")

        except Exception as exc:
            st.error(
                "Research failed. Check Ollama, your .env configuration, "
                "and the terminal error details."
            )
            st.exception(exc)

        finally:
            research_lock.release()



# -----------------------------
# Display final or previous report
# -----------------------------
if st.session_state.research_result:
    st.divider()
    st.subheader("Research Report")

    st.markdown(st.session_state.research_result)

    st.download_button(
        label="⬇️ Download Markdown Report",
        data=st.session_state.research_result,
        file_name="research_report.md",
        mime="text/markdown",
        use_container_width=False,
    )
