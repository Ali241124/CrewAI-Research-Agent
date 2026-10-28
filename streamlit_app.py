import threading
from pathlib import Path
from datetime import datetime

import streamlit as st
from research_crew import research_crew

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
# Sidebar: Previous reports
# -----------------------------
with st.sidebar:
    st.title("🔎 Research Agent")
    st.caption("AI-powered web research")
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
            with st.expander(report_file.stem.replace("_", " ")):
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

                result = research_crew.kickoff(
                    inputs={"topic": topic}
                )

                report_text = str(result)

                safe_topic = "".join(
                    char if char.isalnum() else "_"
                    for char in topic
                ).strip("_")[:60]

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                report_path = (
                    REPORTS_DIR
                    / f"{safe_topic}_{timestamp}_ui.md"
                )

                report_path.write_text(
                    report_text,
                    encoding="utf-8",
                )

                st.session_state.research_result = report_text

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
