
from crewai import Agent, Task, Crew, LLM

llm = LLM(
    model="ollama/qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0.1,
    timeout=300,
    max_retries=1,
)

writer = Agent(
    role="Research Writer",
    goal="Write a short, evidence-based summary.",
    backstory="You write concise technical summaries.",
    llm=llm,
    max_iter=1,
    verbose=True,
    allow_delegation=False,
)

task = Task(
    description=(
        "Write a summary of about 150 words about deep learning "
        "in healthcare, using only this evidence: "
        "Source S1 says deep learning is used for medical imaging "
        "and clinical decision support. It also identifies privacy "
        "and interpretability as challenges. "
        "Explain the applications and limitations without adding "
        "unsupported statistics. Cite the evidence as [S1]."
    ),
    expected_output="A short Markdown summary with an [S1] citation.",
    agent=writer,
)

result = Crew(
    agents=[writer],
    tasks=[task],
    verbose=True,
).kickoff()

print("\nRESULT:\n")
print(result.raw)
