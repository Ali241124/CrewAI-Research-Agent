
import os
from dotenv import load_dotenv
from tavily import TavilyClient
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

load_dotenv()

llm = LLM(
    model="ollama/qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0.1,
    timeout=300,
    max_retries=1,
)

@tool("Tavily Web Search")
def tavily_web_search(query: str) -> str:
    """Search the web and return up to three source titles, URLs, and summaries."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY is missing."

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=3,
        search_depth="basic",
    )

    results = response.get("results", [])
    return "\n\n".join(
        f"Source {i}: {item.get('title', 'Untitled')}\n"
        f"URL: {item.get('url', 'Unavailable')}\n"
        f"Summary: {item.get('content', '')[:400]}"
        for i, item in enumerate(results, start=1)
    ) or "No search results found."


researcher = Agent(
    role="AI Researcher",
    goal="Find useful web sources and summarize their findings.",
    backstory="You are a concise researcher who uses search results carefully.",
    llm=llm,
    tools=[tavily_web_search],
    max_iter=2,
    max_retry_limit=1,
    verbose=True,
    allow_delegation=False,
)

task = Task(
    description=(
        "Research the topic: {topic}. "
        "Search the web once. Summarize the returned results in "
        "three to five bullet points and list the source titles and URLs. "
        "Do not invent references or claim you independently verified pages."
    ),
    expected_output="A short summary with source titles and URLs.",
    agent=researcher,
)

if __name__ == "__main__":
    result = Crew(
        agents=[researcher],
        tasks=[task],
        verbose=True,
    ).kickoff(
        inputs={"topic": "Official Python documentation for machine learning"}
    )
    print("\nFINAL RESULT:\n")
    print(result)
