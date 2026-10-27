import os
from pathlib import Path
from datetime import datetime
import re
from dotenv import load_dotenv
from tavily import TavilyClient
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool

load_dotenv()

# --------------------------------------------------
# 1. Configure our local language model
# --------------------------------------------------

llm = LLM(
    model="ollama/qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0.3,
)


@tool("Tavily Web Search")
def tavily_web_search(query: str) -> str:
    """Search the live web for relevant information.
    Return source titles, URLs, and text summaries
    that can be used to research a topic.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return (
            "Search unavailable: TAVILY_API_KEY is missing "
            "from the .env file."
        )

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        max_results=5,
        search_depth="basic",
    )

    results = response.get("results", [])

    if not results:
        return "No search results were returned."

    formatted_results = []

    for index, result in enumerate(results, start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "No URL")
        content = result.get("content", "No summary available")

        formatted_results.append(
            f"Source {index}: {title}\n"
            f"URL: {url}\n"
            f"Summary: {content[:1500]}"
        )

    return "\n\n".join(formatted_results)

# --------------------------------------------------
# 2. Create the Research Agent
# --------------------------------------------------

researcher = Agent(
    role="AI Researcher",
    goal="Research topics using current web sources",
    backstory=(
        "You are a careful AI researcher. "
        "You use web search to find relevant information, "
        "record source URLs, and distinguish evidence "
        "from assumptions. Never invent references."
    ),
    llm=llm,
    tools=[tavily_web_search],
    max_iter=4,
    max_retry_limit=1,
    verbose=True,
    allow_delegation=False,
)

# --------------------------------------------------
# 3. Create the Analysis Agent
# --------------------------------------------------

analyst = Agent(
    role="Research Analyst",
    goal="Evaluate research findings and extract useful insights",
    backstory=(
        "You are an analytical thinker who identifies "
        "patterns, compares ideas, and distinguishes "
        "established facts from speculation."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# --------------------------------------------------
# 4. Create the Source Verification Agent
# --------------------------------------------------

verifier = Agent(
    role="Source Verification Specialist",
    goal=(
        "Evaluate the credibility, relevance, and evidence "
        "quality of sources collected during research"
    ),
    backstory=(
        "You are a careful research verification specialist. "
        "You examine source titles, URLs, and available summaries. "
        "You distinguish peer-reviewed research, official "
        "documentation, reputable reporting, and opinion pieces "
        "from marketing material, job listings, and unsupported "
        "claims. You identify duplicate sources and limitations. "
        "You never claim to have opened a URL, checked a fact, "
        "or verified a claim unless the available evidence "
        "actually supports that statement."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# --------------------------------------------------
# 4. Create the Report Writer Agent
# --------------------------------------------------

writer = Agent(
    role="Technical Report Writer",
    goal="Produce a clear and well-organized research report",
    backstory=(
        "You write professional reports for students and "
        "technology professionals. Your writing is structured, "
        "readable, and based on the supplied research."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# --------------------------------------------------
# 5. Define the research task
# --------------------------------------------------

research_task = Task(
    description=(
        "Research the topic: {topic}. "
        "Use the Tavily Web Search tool to find relevant "
        "and reasonably recent online sources. "
        "Search for the definition, major concepts, practical "
        "applications, benefits, limitations, and open questions. "
        "Include source titles and URLs in your findings. "
        "Distinguish source-supported claims from assumptions. "
        "Do not fabricate sources or claim that a source "
        "supports something you have not verified."
    ),
    expected_output=(
        "A structured collection of research findings covering "
        "the definition, concepts, applications, benefits, "
        "limitations, and uncertainties."
    ),
    agent=researcher,
)


# --------------------------------------------------
# 6. Define the source verification task
# --------------------------------------------------

verification_task = Task(
    description=(
        "Evaluate the research findings about {topic} produced "
        "by the research task. Examine each source's title, URL, "
        "and available summary. For every source, report:\n"
        "1. Source title and URL.\n"
        "2. Source category: academic research, official "
        "documentation, reputable reporting, opinion, marketing, "
        "job listing, or other.\n"
        "3. Relevance to the research topic.\n"
        "4. What evidence the available summary actually provides.\n"
        "5. Potential bias, limitations, or unsupported claims.\n"
        "6. Whether the source appears duplicated.\n\n"
        "Assign a qualitative evidence assessment: stronger, "
        "limited, or weak, and explain the reason. Do not treat "
        "a source as reliable solely because it appears in search "
        "results. Do not claim to have opened URLs, checked "
        "publication details, or independently verified facts "
        "when you have not done so. If information is insufficient, "
        "explicitly say 'Unable to determine from the available "
        "information.'\n\n"
        "Finish with a concise assessment of the overall source "
        "quality and identify claims that require further checking."
    ),
    expected_output=(
        "A source-by-source verification report listing each "
        "source's title, URL, category, relevance, available "
        "evidence, limitations, possible duplication, and "
        "qualitative evidence assessment, followed by an overall "
        "source-quality summary."
    ),
    agent=verifier,
    context=[research_task],
)

# --------------------------------------------------
# 6. Define the analysis task
# --------------------------------------------------

analysis_task = Task(
    description=(
        "Analyze the research findings about {topic}, using both "
        "the original research task and the source verification "
        "report. Prioritize claims supported by stronger evidence. "
        "Identify important insights, practical implications, "
        "limitations, conflicting evidence, and unanswered questions. "
        "Clearly distinguish supported findings from weak evidence "
        "and assumptions. Do not introduce unsupported facts."
    ),
    expected_output=(
        "An analytical summary containing key insights, "
        "practical implications, limitations, and open questions."
    ),
    agent=analyst,
    context=[research_task, verification_task],
)

# --------------------------------------------------
# 7. Define the report-writing task
# --------------------------------------------------

writing_task = Task(
    description=(
        "Write a professional report about {topic} using "
        "the research findings and analysis. Include a title, "
        "introduction, main findings, practical applications, "
        "limitations, conclusion, and verification notes. "
        "Do not present unverified claims as confirmed facts."
    ),
    expected_output=(
        "A complete, readable Markdown research report with "
        "clear headings and a balanced conclusion."
    ),
    agent=writer,
    context=[research_task, verification_task, analysis_task],
)

# --------------------------------------------------
# 8. Assemble the crew
# --------------------------------------------------

research_crew = Crew(
    agents=[researcher, verifier, analyst, writer],
    tasks=[
        research_task,
        verification_task,
        analysis_task,
        writing_task,
    ],
    process=Process.sequential,
    verbose=True,
)

# --------------------------------------------------
# 9. Run the crew
# --------------------------------------------------

if __name__ == "__main__":
    topic = input("Enter a research topic: ").strip()

    if not topic:
        print("Please enter a research topic.")
    else:
        # Run the research crew
        result = research_crew.kickoff(inputs={"topic": topic})

        # Display the final report
        print("\n" + "=" * 60)
        print("FINAL RESEARCH REPORT")
        print("=" * 60)

        report = result.raw if hasattr(result, "raw") else str(result)
        print(report)

        # Create the reports folder if it does not exist
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        # Make a safe filename from the research topic
        safe_topic = re.sub(r'[<>:"/\\|?*]', "", topic)
        safe_topic = re.sub(r"\s+", "_", safe_topic.strip())
        safe_topic = safe_topic[:70] or "research_report"

        # Add a timestamp to prevent overwriting earlier reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"{safe_topic}_{timestamp}.md"

        # Save the report as a Markdown file
        report_path.write_text(report, encoding="utf-8")

        print(f"\nReport saved successfully to: {report_path.resolve()}")