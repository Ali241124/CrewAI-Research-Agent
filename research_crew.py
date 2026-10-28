
import os
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, urldefrag

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


def classify_source(source: dict) -> dict:
    """Classify a source for review; this is not proof of credibility."""

    domain = (source.get("domain") or "").lower()
    title = (source.get("title") or "").lower()
    summary = (source.get("summary") or "").strip()

    academic_domains = (
        "pubmed.ncbi.nlm.nih.gov",
        "jmir.org",
        "nature.com",
        "sciencedirect.com",
        "springer.com",
        "wiley.com",
        "ieee.org",
        "acm.org",
        "mdpi.com",
        "plos.org",
        "arxiv.org",
    )

    aggregator_domains = (
        "paperguide.ai",
        "researchgate.net",
        "medium.com",
    )

    if any(
        domain == trusted or domain.endswith("." + trusted)
        for trusted in academic_domains
    ):
        source_type = "Academic publisher/repository"
        review_status = "Candidate for verification"
    elif any(
        domain == site or domain.endswith("." + site)
        for site in aggregator_domains
    ):
        source_type = "Aggregator/community platform"
        review_status = "Review original source"
    elif ".edu" in domain or domain.endswith(".gov"):
        source_type = "Educational/government domain"
        review_status = "Candidate for verification"
    else:
        source_type = "Other website"
        review_status = "Needs review"

    if len(summary) < 100:
        review_status = "Insufficient evidence excerpt"

    source["source_type"] = source_type
    source["review_status"] = review_status
    source["evidence_length"] = len(summary)

    return source

# --------------------------------------------------
# 2. Search the web directly using Tavily
# --------------------------------------------------

def search_web(topic: str) -> list[dict]:
    """Search Tavily and return filtered, deduplicated sources."""

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is missing. Check your .env file."
        )

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=(
            f"{topic} official documentation research papers "
            "reputable sources"
        ),
        max_results=5,
        search_depth="basic",
    )

    results = response.get("results", [])
    sources = []
    seen_urls = set()

    for item in results:
        url = (item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        summary = (item.get("content") or "").strip()

        # Ignore incomplete search results.
        if not url or not title or not summary:
            continue

        # Only accept HTTP/HTTPS URLs with a valid hostname.
        parsed_url = urlparse(url)

        if (
            parsed_url.scheme not in ("http", "https")
            or not parsed_url.hostname
        ):
            continue

        # Remove URL fragments and normalize the hostname.
        clean_url = urldefrag(url).url
        parsed_url = urlparse(clean_url)
        normalized_host = (parsed_url.hostname or "").lower()

        # Remove common tracking parameters for duplicate detection.
        normalized_path = parsed_url.path.rstrip("/") or "/"
        normalized_query = parsed_url.query

        tracking_markers = (
            "utm_", "gclid", "fbclid", "mc_cid", "mc_eid"
        )

        if normalized_query:
            kept_params = []

            for parameter in normalized_query.split("&"):
                key = parameter.split("=", 1)[0].lower()

                if not key.startswith(tracking_markers):
                    kept_params.append(parameter)

            normalized_query = "&".join(kept_params)

        dedupe_key = (
            normalized_host,
            normalized_path,
            normalized_query,
        )

        if dedupe_key in seen_urls:
            continue

        seen_urls.add(dedupe_key)

        sources.append({
            "title": title,
            "url": clean_url,
            "domain": normalized_host,
            "score": item.get("score"),
            "published_date": (
                item.get("published_date") or "Not provided"
            ),
            "summary": summary[:700],
        })

    # Assign labels only after filtering and deduplication.
    for index, source in enumerate(sources, start=1):
        source["label"] = f"S{index}"
        classify_source(source)

    return sources 

def format_sources(sources: list[dict]) -> str:
    """Format source evidence for the report-writing agent."""

    if not sources:
        return (
            "NO USABLE SOURCES WERE FOUND. State this limitation "
            "clearly. Do not invent sources or research findings."
        )

    formatted = []

    for source in sources:
        formatted.append(
            f"[{source['label']}]\n"
            f"Title: {source['title']}\n"
            f"URL: {source['url']}\n"
            f"Domain: {source['domain']}\n"
            f"Source type: {source['source_type']}\n"
            f"Review status: {source['review_status']}\n"
            f"Published date: {source['published_date']}\n"
            f"Search relevance score: {source['score']}\n"
            f"Evidence excerpt: {source['summary']}"
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
    
    SOURCE QUALITY AND EVIDENCE RULES

    1. Use the supplied source material as the evidence base for the report.
    2. Treat "Candidate for verification" as an unverified candidate, not a confirmed trustworthy source.
    3. Treat "Review original source" and "Needs review" as lower-confidence material. Prefer stronger sources when their evidence supports the claim.
    4. Do not claim that a source is peer-reviewed, authoritative, or independently verified based only on its domain classification.
    5. Cite factual claims using the supplied source labels, such as [S1] or [S4].
    6. Only make claims supported by the relevant source excerpt. Do not infer a paper's findings from its title.
    7. If the available excerpts do not support a requested detail, explicitly state that the evidence is insufficient.
    8. Preserve the exact source URLs supplied in the evidence. Never invent references.
    9. Distinguish findings reported by a source from your own interpretation.
    10. End the report with a References section listing only sources actually cited, using their exact titles and URLs.

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
