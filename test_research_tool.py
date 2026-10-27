
from crewai import Agent, Task, Crew
from research_crew import llm, tavily_web_search

print("Step 1: Creating a tool-enabled agent...", flush=True)

research_agent = Agent(
    role="AI Research Tester",
    goal="Find and summarize relevant information from web search",
    backstory=(
        "You are a careful researcher who uses web search "
        "and distinguishes source evidence from assumptions."
    ),
    llm=llm,
    tools=[tavily_web_search],
    verbose=True,
    max_iter=4,
    allow_delegation=False,
)

research_task = Task(
    description=(
        "Research the topic '{topic}'. You MUST use the "
        "Tavily Web Search tool at least once. Summarize "
        "the results in three short bullet points and include "
        "the source URLs returned by the tool. Do not invent "
        "sources or claim independent verification."
    ),
    expected_output=(
        "Three concise research bullet points with source URLs."
    ),
    agent=research_agent,
)

test_crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    verbose=True,
)

print("Step 2: Starting the research test...", flush=True)

result = test_crew.kickoff(
    inputs={"topic": "practical applications of agentic AI in software engineering"}
)

print("\nStep 3: TEST COMPLETED", flush=True)
print(result, flush=True)
