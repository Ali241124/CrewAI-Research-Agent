
import os
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

from dotenv import load_dotenv
from tavily import TavilyClient
from crewai import Agent, Task, Crew, LLM, Process

load_dotenv()

# --------------------------------------------------
# 1. Configure the local Ollama language model
# --------------------------------------------------

llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.1,
    timeout=300,
    max_retries=1,
)

# --------------------------------------------------
# 2. Search the web directly using Tavily
# --------------------------------------------------

def search_web(topic: str) -> list[dict]:
    """Search once and return source metadata and summaries."""

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is missing. Check your .env file."
        )

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=f"{topic} official documentation research papers reputable sources",
        max_results=3,
        search_depth="basic",
    )

    results = response.get("results", [])
    sources = []

    for index, item in enumerate(results, start=1):
        url = item.get("url") or ""
        parsed_url = urlparse(url)

        sources.append({
            "label": f"S{index}",
            "title": item.get("title") or "Untitled",
            "url": url,
            "domain": parsed_url.netloc,
            "score": item.get("score"),
            "published_date": (
                item.get("published_date") or "Not provided"
            ),
            "summary": (
                item.get("content") or "No summary available"
            )[:400],
        })

    return sources


def format_sources(sources: list[dict]) -> str:
    """Format the search results for the report-writing agent."""

    if not sources:
        return (
            "NO SOURCES WERE FOUND. State this limitation clearly "
            "and do not invent sources or research findings."
        )

    formatted = []

    for source in sources:
        formatted.append(
            f"[{source['label']}]\n"
            f"Title: {source['title']}\n"
            f"URL: {source['url']}\n"
            f"Domain: {source['domain']}\n"
            f"Published date: {source['published_date']}\n"
            f"Search relevance score: {source['score']}\n"
            f"Search-result summary: {source['summary']}"
        )

    return "\n\n".join(formatted)


# --------------------------------------------------
# 3. Create the report-writing agent
# --------------------------------------------------

writer = Agent(
    role="Evidence-Based Research Writer",
    goal=(
        "Create clear, balanced research reports grounded in "
        "the supplied web-search evidence."
    ),
    backstory=(
        "You are a careful technical researcher. You distinguish "
        "search-result summaries from complete articles, avoid "
        "unsupported claims, and preserve exact source URLs. "
        "You never invent references or claim to have independently "
        "verified a webpage."
    ),
    llm=llm,
    max_iter=1,
    max_retry_limit=1,
    verbose=True,
    allow_delegation=False,
)

# --------------------------------------------------
# 4. Define the report-writing task
# --------------------------------------------------

report_task = Task(
    description="""
    Write a concise Markdown research report about: {topic}

    Use only the supplied source material below.

    SOURCE MATERIAL:
    {source_material}

    Rules:
    - Write approximately 250–350 words.
    - Use only claims supported by the supplied summaries.
    - Cite factual claims using [S1], [S2], or [S3] where appropriate.
    - Preserve source titles and URLs exactly.
    - Do not invent statistics, dates, references, or quotations.
    - Do not claim that you independently verified webpages.
    - Explain important evidence gaps and limitations.

    Use these headings:
    # Title
    ## Introduction
    ## Key Findings and Applications
    ## Benefits and Limitations
    ## Conclusion
    ## Sources

    In Sources, list each available source label, exact title, URL,
    and publication date. If evidence is insufficient, say so.
    """,
    expected_output=(
        "A concise Markdown report with evidence-based claims, "
        "citations, limitations, and exact source URLs."
    ),

    agent=writer,
)

# --------------------------------------------------
# 5. Assemble the CrewAI writing crew
# --------------------------------------------------

research_crew = Crew(
    agents=[writer],
    tasks=[report_task],
    process=Process.sequential,
    verbose=True,
)

# --------------------------------------------------
# 6. Run research and save the report
# --------------------------------------------------

if __name__ == "__main__":
    topic = input("Enter a research topic: ").strip()

    if not topic:
        print("Please enter a research topic.")
        raise SystemExit(1)

    try:
        print("\nSearching the web with Tavily...")
        sources = search_web(topic)

        print(f"Retrieved {len(sources)} source(s).")

        if sources:
            for source in sources:
                print(f"- [{source['label']}] {source['title']}")
                print(f"  {source['url']}")

        source_material = format_sources(sources)

        print("\nGenerating the report with local Ollama...")
        result = research_crew.kickoff(
            inputs={
                "topic": topic,
                "source_material": source_material,
            }
        )

        report = result.raw if hasattr(result, "raw") else str(result)

        print("\n" + "=" * 60)
        print("FINAL RESEARCH REPORT")
        print("=" * 60)
        print(report)

        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        safe_topic = re.sub(r'[<>:"/\\|?*]', "", topic)
        safe_topic = re.sub(r"\s+", "_", safe_topic.strip())
        safe_topic = safe_topic[:70] or "research_report"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"{safe_topic}_{timestamp}.md"

        report_path.write_text(report, encoding="utf-8")

        print(f"\nReport saved to: {report_path.resolve()}")

    except Exception as error:
        print(f"\nResearch failed: {type(error).__name__}: {error}")
        raise SystemExit(1)
