
from crewai import Agent, Task, Crew, LLM

print("Step 1: Configuring Ollama...", flush=True)

llm = LLM(
    model="ollama/qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0,
)

print("Step 2: Creating agent...", flush=True)

agent = Agent(
    role="Simple AI Assistant",
    goal="Answer questions briefly and accurately",
    backstory="You are a helpful assistant.",
    llm=llm,
    verbose=True,
    max_iter=1,
    allow_delegation=False,
)

task = Task(
    description="In one sentence, explain what Python is.",
    expected_output="One short sentence.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
)

print("Step 3: Sending request to CrewAI...", flush=True)

result = crew.kickoff()

print("\nStep 4: SUCCESS!", flush=True)
print(result, flush=True)
