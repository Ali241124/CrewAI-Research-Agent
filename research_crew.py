import os

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
# 6. Define the analysis task
# --------------------------------------------------

analysis_task = Task(
    description=(
        "Analyze the research findings about {topic}. "
        "Identify the most important insights, practical "
        "implications, limitations, and unanswered questions. "
        "Use the research task's output as your main evidence. "
        "Do not introduce unsupported facts."
    ),
    expected_output=(
        "An analytical summary containing key insights, "
        "practical implications, limitations, and open questions."
    ),
    agent=analyst,
    context=[research_task],
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
    context=[research_task, analysis_task],
)

# --------------------------------------------------
# 8. Assemble the crew
# --------------------------------------------------

research_crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
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
        result = research_crew.kickoff(
            inputs={"topic": topic}
        )

        print("\n" + "=" * 60)
        print("FINAL RESEARCH REPORT")
        print("=" * 60)
        print(result)
